import json
import re
import uuid
import asyncio
import threading
from typing import List, Dict, Any, Optional, Union, Literal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mlx_lm import load, generate, stream_generate
from config import config

# Global variables for model and tokenizer
model = None
tokenizer = None


# ============================================================
# Pydantic Models (Data shapes for request/response validation)
# ============================================================
# Junior tip: Pydantic models define the "shape" of data.
# FastAPI uses them to auto-validate incoming JSON and serialize responses.


class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class ContentBlockThinking(BaseModel):
    """Claude API thinking content block — returned when thinking is enabled."""
    type: Literal["thinking"] = "thinking"
    thinking: str


class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    # Why Union: response can contain text, tool_use, or thinking blocks
    content: List[Dict[str, Any]]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


class MessageStreamResponse(BaseModel):
    type: str
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None


# ============================================================
# Startup / Shutdown
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading MLX model: {config.MODEL_NAME}")

    tokenizer_config = {}
    if config.TRUST_REMOTE_CODE:
        tokenizer_config["trust_remote_code"] = True
    if config.EOS_TOKEN:
        tokenizer_config["eos_token"] = config.EOS_TOKEN

    model, tokenizer = load(config.MODEL_NAME, tokenizer_config=tokenizer_config)
    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


# ============================================================
# Content Extraction Helpers
# ============================================================


def _flatten_tool_result_content(content: Any) -> str:
    """Convert tool_result content (str, dict, list) to a flat string.

    Why a separate function: tool_result.content can be a string, a dict,
    a list of dicts, or anything. We normalize to a string the model can read.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        if "text" in content:
            return content["text"]
        return json.dumps(content)
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def extract_text_from_content(
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ],
) -> str:
    """Extract text content from Claude-style content blocks.

    Now also converts tool_use and tool_result blocks to human-readable
    text so the model sees the full conversation context.
    """
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if hasattr(block, "type"):
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Why JSON dump: model needs to see what tool was called with what params
                text_parts.append(
                    f'[Calling tool: {block.name}({json.dumps(block.input)})]'
                )
            elif block.type == "tool_result":
                result_text = _flatten_tool_result_content(block.content)
                text_parts.append(
                    f'[Tool result for {block.tool_use_id}: {result_text}]'
                )
        elif isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                text_parts.append(
                    f'[Calling tool: {block["name"]}({json.dumps(block.get("input", {}))})]'
                )
            elif block.get("type") == "tool_result":
                result_text = _flatten_tool_result_content(block.get("content", ""))
                text_parts.append(
                    f'[Tool result for {block.get("tool_use_id", "?")}: {result_text}]'
                )

    return " ".join(text_parts)


def extract_system_text(
    system: Optional[Union[str, List[SystemContent]]],
) -> Optional[str]:
    """Extract system text from system parameter"""
    if isinstance(system, str):
        return system
    elif isinstance(system, list):
        return " ".join([content.text for content in system])
    return None


# ============================================================
# Tool Calling Adapter: Claude JSON <-> Qwen XML
# ============================================================
# Design Pattern: Adapter
# ELI5: Translates between two "languages" — Claude's JSON tool format
#        and Qwen's XML tool format.
# Real analogy: A USB-C to Lightning adapter.


def format_tools_for_chat_template(tools: Optional[List[Tool]]) -> Optional[List[dict]]:
    """Convert Claude API tool definitions into OpenAI-style function defs.

    Why OpenAI format: Qwen's chat template (Jinja2) expects the same
    shape as OpenAI function-calling — a list of dicts with
    {"type": "function", "function": {name, description, parameters}}.
    """
    if not tools:
        return None

    formatted = []
    for tool in tools:
        formatted.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        })
    return formatted


def parse_tool_calls_from_response(response_text: str) -> tuple:
    """Parse Qwen 3.5's XML tool-call output into Claude API tool_use blocks.

    Qwen outputs:  <function=read_file><parameter=path>foo.py</parameter></function>
    Claude expects: {"type":"tool_use","id":"toolu_...","name":"...","input":{...}}

    Also handles JSON-based tool calls some Qwen variants produce:
        <function=read_file>{"path": "foo.py"}</function>

    Returns: (clean_text_without_tool_xml, list_of_tool_use_dicts)
    """
    tool_calls = []

    # Match <function=name>...inner...</function>
    xml_pattern = r'<function=(\w+)>(.*?)</function>'

    for match in re.finditer(xml_pattern, response_text, re.DOTALL):
        func_name = match.group(1)
        inner = match.group(2).strip()

        params = {}

        # Try XML parameter tags first: <parameter=key>value</parameter>
        param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
        param_matches = list(re.finditer(param_pattern, inner, re.DOTALL))

        if param_matches:
            for pm in param_matches:
                params[pm.group(1)] = pm.group(2).strip()
        else:
            # Fallback: inner might be raw JSON
            try:
                params = json.loads(inner)
            except (json.JSONDecodeError, TypeError):
                if inner:
                    params = {"input": inner}

        tool_calls.append({
            "type": "tool_use",
            # Why uuid: Claude API requires each tool_use block to have a unique ID
            # so the client can match tool_result back to it.
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": func_name,
            "input": params,
        })

    # Remove tool-call XML from the text
    clean_text = re.sub(xml_pattern, '', response_text, flags=re.DOTALL).strip()

    return clean_text, tool_calls


# ============================================================
# Thinking Block Handler: Qwen <think> -> Claude thinking blocks
# ============================================================


def parse_thinking_blocks(response_text: str) -> tuple:
    """Extract <think>...</think> blocks from Qwen 3.5 output.

    Qwen 3.5 outputs chain-of-thought reasoning inside <think> tags.
    Claude API has a `thinking` content block type for this purpose.

    Returns: (clean_text_without_think_tags, thinking_text_or_None)
    """
    think_pattern = r'<think>(.*?)</think>'
    thinking_parts = re.findall(think_pattern, response_text, re.DOTALL)
    clean_text = re.sub(think_pattern, '', response_text, flags=re.DOTALL).strip()

    thinking_text = '\n'.join(p.strip() for p in thinking_parts).strip() if thinking_parts else None
    return clean_text, thinking_text


# ============================================================
# Process Model Response: Parse thinking + tool calls into content blocks
# ============================================================


def process_model_response(
    response_text: str,
    thinking_enabled: bool = False,
) -> tuple:
    """Process raw model output into Claude API content blocks and stop_reason.

    This is the main "output adapter" — takes raw text from Qwen 3.5 and
    converts it into the structured format Claude Code expects.

    Returns: (content_blocks_list, stop_reason_string)
    """
    content_blocks = []

    # Step 1: Extract thinking blocks
    text, thinking_text = parse_thinking_blocks(response_text)

    # Step 2: Extract tool calls
    text, tool_calls = parse_tool_calls_from_response(text)

    # Build content blocks in Claude API order:
    # 1. thinking (if enabled and present)
    # 2. text (if any remaining)
    # 3. tool_use blocks (if any)

    if thinking_enabled and thinking_text:
        content_blocks.append({"type": "thinking", "thinking": thinking_text})

    if text:
        content_blocks.append({"type": "text", "text": text})

    if tool_calls:
        content_blocks.extend(tool_calls)
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    # Ensure at least one text block exists (Claude API requires non-empty content)
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return content_blocks, stop_reason


# ============================================================
# Message Formatting: Claude API -> Chat Template
# ============================================================


def format_messages_for_model(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
    tools: Optional[List[Tool]] = None,
) -> str:
    """Convert Claude-style messages to the format expected by the model's chat template.

    This is the core input adapter. It:
    1. Builds a list of {role, content} dicts from Claude's complex content blocks
    2. Converts tool definitions for the chat template if present
    3. Calls tokenizer.apply_chat_template() (model's Jinja2 template)
    4. Falls back to manual formatting if the template fails
    """
    formatted_messages = []

    # Add system message
    system_text = extract_system_text(system)
    if system_text:
        formatted_messages.append({"role": "system", "content": system_text})

    # Convert each Claude message to simple {role, content}
    for message in messages:
        content_text = extract_text_from_content(message.content)
        formatted_messages.append({"role": message.role, "content": content_text})

    # Convert Claude tool defs to OpenAI-style for chat template
    formatted_tools = format_tools_for_chat_template(tools)

    # Apply chat template if available
    if tokenizer.chat_template is not None:
        try:
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            # Why pass tools separately: Qwen's Jinja2 template accepts a
            # `tools` kwarg and auto-injects tool descriptions into the
            # system prompt. This is how the model learns what tools exist.
            if formatted_tools:
                template_kwargs["tools"] = formatted_tools

            result = tokenizer.apply_chat_template(
                formatted_messages, **template_kwargs
            )
            if isinstance(result, str):
                return result
        except Exception as e:
            print(f"Chat template failed, using fallback: {e}")

    # Fallback formatting (used if no template or template fails)
    prompt = ""
    for msg in formatted_messages:
        role = msg["role"]
        content = msg["content"]
        prompt += "<|" + role + "|>" + "\n" + content + "\n" + "<|end|>" + "\n"
    prompt += "<|assistant|>\n"
    return prompt


# ============================================================
# Token Counting
# ============================================================


def count_tokens(text: str) -> int:
    """Count tokens in text using the loaded tokenizer."""
    try:
        if isinstance(text, str) and text.strip():
            # Try the tokenizer's __call__ method first
            try:
                result = tokenizer(text, return_tensors=False, add_special_tokens=False)
                if isinstance(result, dict) and "input_ids" in result:
                    return len(result["input_ids"])
                elif hasattr(result, "__len__"):
                    return len(result)
            except (AttributeError, TypeError, ValueError):
                pass

            # Try direct encode
            try:
                encoded = tokenizer.encode(text)
                return len(encoded) if hasattr(encoded, "__len__") else len(list(encoded))
            except (AttributeError, TypeError, ValueError):
                pass

            # Try with explicit string conversion
            try:
                tokens = tokenizer.encode(str(text), add_special_tokens=False)
                return len(tokens)
            except (AttributeError, TypeError, ValueError):
                pass

        # Fallback: character-based estimation (~4 chars per token)
        return max(1, len(str(text)) // 4)

    except Exception as e:
        print(f"Token counting failed with error: {e}")
        return max(1, len(str(text)) // 4)


# ============================================================
# Context Monitoring
# ============================================================


def get_max_context_length() -> int:
    """
    Safely try to determine the maximum context length of the loaded model.
    Checks user config first, then inspects model config attributes.
    """
    if config.DEFAULT_MAX_TOKENS is not None:
        return config.DEFAULT_MAX_TOKENS

    if model is None:
        return 8192  # Safe default if model not loaded

    # Try common MLX model config attributes
    try:
        # Some MLX models store config in 'config', others in 'text_model.config'
        m_config = model.config if hasattr(model, "config") else getattr(model, "text_model", model).config

        # Check standard attributes that represent context length
        for attr in ["max_position_embeddings", "sliding_window", "n_positions", "max_seq_len"]:
            if hasattr(m_config, attr) and getattr(m_config, attr) is not None:
                return int(getattr(m_config, attr))
    except Exception:
        pass

    return 8192  # Fallback if we absolutely cannot find it


# ============================================================
# API Route Handlers
# ============================================================


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format messages using the model's chat template
        prompt = format_messages_for_model(request.messages, request.system, request.tools)

        input_tokens = await asyncio.to_thread(count_tokens, prompt)
        max_context = get_max_context_length()
        remaining = max_context - input_tokens

        # Log to server console
        print(f"[{request.model}] Context used: {input_tokens} / {max_context} tokens. Remaining: {remaining}")

        # Check if thinking is enabled in the request
        thinking_enabled = (
            request.thinking is not None and request.thinking.type in ("enabled", "adaptive")
        )

        if request.stream:
            return StreamingResponse(
                stream_generate_response(request, prompt, input_tokens, thinking_enabled),
                media_type="text/event-stream",
            )
        else:
            return await generate_response(request, prompt, input_tokens, thinking_enabled)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: TokenCountRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prompt = format_messages_for_model(request.messages, request.system, request.tools)
        token_count = await asyncio.to_thread(count_tokens, prompt)
        return {"input_tokens": token_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Response Generation (Non-Streaming)
# ============================================================


async def generate_response(
    request: MessagesRequest,
    prompt: str,
    input_tokens: int,
    thinking_enabled: bool,
):
    """Generate non-streaming response with tool call and thinking support."""
    # Build generation kwargs
    gen_kwargs = {
        "max_tokens": request.max_tokens,
        "verbose": config.VERBOSE,
    }
    # Pass repetition_penalty if configured (helps Qwen avoid loops)
    if config.REPETITION_PENALTY is not None:
        gen_kwargs["repetition_penalty"] = config.REPETITION_PENALTY

    response_text = await asyncio.to_thread(
        generate,
        model,
        tokenizer,
        prompt=prompt,
        **gen_kwargs,
    )

    # Process the raw output: extract thinking + tool calls
    content_blocks, stop_reason = process_model_response(response_text, thinking_enabled)

    output_tokens = await asyncio.to_thread(count_tokens, response_text)

    response = MessageResponse(
        id="msg_" + str(abs(hash(prompt)))[:8],
        content=content_blocks,
        model=request.model,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )

    return response


# ============================================================
# Response Generation (Streaming via SSE)
# ============================================================


async def stream_generate_response(
    request: MessagesRequest,
    prompt: str,
    input_tokens: int,
    thinking_enabled: bool,
):
    """Generate streaming response following Claude's SSE protocol.

    How it works:
    1. Send message_start event
    2. Start a background thread to generate tokens (Producer)
    3. Read tokens from a queue in the main thread (Consumer)
    4. Send content_block_delta events as tokens arrive
    5. After streaming completes, process full text for tool calls / thinking
    6. If tool calls found, send additional tool_use content blocks
    7. Send message_delta with final stop_reason and usage
    8. Send message_stop

    Why threads: mlx_lm.stream_generate is blocking and CPU-heavy.
    Running it in a thread keeps the FastAPI event loop responsive.
    """
    response_id = "msg_" + str(abs(hash(prompt)))[:8]
    full_text = ""

    # Build generation kwargs
    gen_kwargs = {
        "max_tokens": request.max_tokens,
    }
    if config.REPETITION_PENALTY is not None:
        gen_kwargs["repetition_penalty"] = config.REPETITION_PENALTY

    # Producer-Consumer queue setup
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer():
        try:
            for response in stream_generate(
                model, tokenizer, prompt=prompt, **gen_kwargs
            ):
                asyncio.run_coroutine_threadsafe(queue.put(response), loop)
            # Poison pill
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(e), loop)

    # Start the generator thread
    threading.Thread(target=producer, daemon=True).start()

    # --- message_start ---
    message_start = {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": request.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # --- content_block_start for text (index 0) ---
    content_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_start)}\n\n"

    # --- Stream tokens from the queue ---
    while True:
        response = await queue.get()
        if response is None:
            break
        if isinstance(response, Exception):
            raise response

        full_text += response.text

        content_delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": response.text},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"

    # --- content_block_stop for text ---
    content_stop = {"type": "content_block_stop", "index": 0}
    yield f"event: content_block_stop\ndata: {json.dumps(content_stop)}\n\n"

    # --- Post-process: check for tool calls in the completed text ---
    _, tool_calls = parse_tool_calls_from_response(full_text)

    # If tool calls found, emit them as additional content blocks
    for i, tool_call in enumerate(tool_calls):
        block_index = i + 1  # text was index 0

        # content_block_start for tool_use
        tool_start = {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["name"],
                "input": {},
            },
        }
        yield f"event: content_block_start\ndata: {json.dumps(tool_start)}\n\n"

        # Send tool input as a single delta
        tool_input_delta = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(tool_call["input"]),
            },
        }
        yield f"event: content_block_delta\ndata: {json.dumps(tool_input_delta)}\n\n"

        # content_block_stop for tool_use
        tool_stop = {"type": "content_block_stop", "index": block_index}
        yield f"event: content_block_stop\ndata: {json.dumps(tool_stop)}\n\n"

    # --- Determine stop reason ---
    stop_reason = "tool_use" if tool_calls else "end_turn"

    output_tokens = await asyncio.to_thread(count_tokens, full_text)

    # --- message_delta with final usage ---
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # --- message_stop ---
    message_stop = {"type": "message_stop"}
    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"


# ============================================================
# Health / Status Endpoints
# ============================================================


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "model_loaded": model is not None,
    }


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":
    print(f"Starting Claude Code MLX Proxy on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)

# Idea 8: Asynchronous Pre-filling Implementation Plan

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* **MLX KV Cache API:** How exactly does `mlx_lm` allow us to pass in a pre-computed Key-Value (KV) cache to the `generate` or `stream_generate` functions? We will need to investigate if we can evaluate the model on chunks of prompt tokens and build the cache state before calling the final generation loop.
* **Streaming JSON Parser:** Standard `json.loads` blocks until the entire string is loaded. We will need a streaming JSON parser (like `ijson`) to read the `MessagesRequest` as it's being downloaded by FastAPI.

### ⚠️ Potential Effects of Uncertainty

| Question | Impact |
|---|---|
| Can `mlx_lm` accept pre-computed caches? | If MLX doesn't expose a clean way to inject a KV cache into its high-level `generate` functions, we may have to write a custom generation loop using the lower-level `model(tokens, cache)` API. |
| How complex is the JSON streaming? | If `ijson` is too slow or complex to integrate with FastAPI's async request body stream, we might have to write a custom ASGI middleware or accept a slight delay. |

---

## Big Picture: What Does This Phase Do?

> **Junior tip:** When dealing with large payloads (like sending a whole codebase to an AI), downloading the payload over the network takes time. If we wait for the download to finish *before* we start thinking about it, we waste time. Asynchronous pre-filling means we start thinking about the first paragraph while the second paragraph is still downloading.

**The Goal:**
Reduce the "Time-To-First-Token" (TTFT) by processing the Claude Code request in chunks as it arrives over the network, rather than waiting for the entire request to download before starting the MLX tokenization and evaluation.

**The Approach:**
1. Bypass FastAPI's strict Pydantic parsing, which forces the whole request body to load into memory.
2. Stream the incoming HTTP request bytes.
3. Use a streaming JSON parser (`ijson`) to extract the system prompt and conversation messages *as they arrive*.
4. Send extracted chunks of text to a background thread.
5. In the background thread, tokenize the text and evaluate the MLX model to build up the Key-Value (KV) cache mathematically.
6. When the full request finishes downloading, take the fully primed KV cache, append any final tokens, and start generation immediately.

### Patterns Used

- **Pattern Name:** Pipeline / Stream Processing Pattern
- **One-Line ELI5:** An assembly line where station 2 starts working on a part as soon as station 1 finishes its piece, instead of waiting for the whole batch.
- **Why Here:** We have a sequential bottleneck: Network I/O -> JSON Parsing -> Tokenization -> Model Evaluation. Streaming lets us overlap these steps.
- **Real Analogy:** Watching a YouTube video. You don't wait for the entire 1GB video file to download to your computer before you start watching. You watch the first minute while the second minute downloads.

- **Pattern Name:** State Accumulator (KV Cache)
- **One-Line ELI5:** Keeping a running total on a calculator instead of calculating everything from scratch at the end.
- **Why Here:** The MLX model can build its internal state (cache) progressively.
- **Real Analogy:** Reading a book chapter by chapter and remembering the plot, rather than trying to read the whole book in a single sitting on the day of the book report.

---

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Add Streaming JSON Dependencies
We will need to add `ijson` to the project to parse JSON iteratively.
- Update `pyproject.toml` / dependencies to include `ijson`.

#### 2. Introduce a Custom Streaming Route
Instead of letting FastAPI parse `MessagesRequest` directly, we will read the raw request stream.

```python
import ijson
from fastapi import Request

@app.post("/v1/messages")
async def create_message(request: Request):
    # We will read chunks of data from `request.stream()`
    # We will use ijson to asynchronously parse the messages array as it arrives.
    pass
```

#### 3. Implement the Cache Builder Thread
We need a background thread that accepts strings (messages), tokenizes them, and runs a forward pass on the model to update the KV cache.

```python
# Junior Tip: A queue is used to send data safely between the web server thread and the AI processing thread.
text_chunk_queue = queue.Queue()

def cache_builder_worker():
    # Initialize empty MLX cache
    # Loop over queue:
    #   Get text chunk
    #   Tokenize
    #   Run model forward pass: model.forward(tokens, cache)
    #   Wait for next chunk
```

#### 4. Fallback/Integrate with `generate`
Once the HTTP request is fully finished and parsed, we check if the cache is ready. If we have a partially or fully built cache, we pass it to `stream_generate` or our own custom generation loop.

```python
    # Final step when all data is received
    final_cache = get_built_cache()
    # Resume generation from the pre-filled cache state
```

## User Review Required
> [!WARNING]
> This is a highly advanced optimization that requires dropping down to lower-level MLX APIs (managing `KVCache` manually instead of just calling `mlx_lm.generate`). It also requires changing how FastAPI validates the request. 

Are we aligned on pursuing this advanced architectural change, or should we investigate a simpler optimization first?

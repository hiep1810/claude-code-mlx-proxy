# Fix MLX Generate Cache Argument

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* The exact keyword argument expected by different versions of `mlx_lm` to pass an existing cache. It might be `prompt_cache`, `cache`, or we may need to drop down to `mlx_lm.generate_step` manually if the high-level wrappers don't expose it correctly.

### ⚠️ Potential Effects of Uncertainty
* If we use the wrong keyword, `stream_generate` continues to crash.

---

## Big Picture: Why is it crashing?

In our attempt to pass the warmed-up memory (KV Cache) to the model, we guessed the keyword argument was `existing_cache`:

```python
return generate(model, tokenizer, prompt=prompt, existing_cache=kv_cache)
```

However, the MLX `generate_step` function (which `stream_generate` and `generate` call under the hood) does not accept a parameter named `existing_cache`. It threw a `TypeError`.

Looking at the MLX framework documentation and source patterns, when you provide a custom cache to the generation functions, the proper parameter name is usually **`prompt_cache`**.

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Change `existing_cache` to `prompt_cache` in `generate_response`
We will rename the parameter in our wrapper function to match our intent, and update the kwargs logic.

```python
async def generate_response(
    request: MessagesRequest,
    prompt: str,
    input_tokens: int,
    thinking_enabled: bool,
    prompt_cache: Optional[List[Any]] = None,
):
    # inside thread_generate:
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
```

#### 2. Change `existing_cache` to `prompt_cache` in `stream_generate_response`
Similarly, update the streaming function to pass the cache using the correct keyword.

```python
async def stream_generate_response(
    request: MessagesRequest,
    prompt: str,
    input_tokens: int,
    thinking_enabled: bool,
    prompt_cache: Optional[List[Any]] = None,
):
    # inside producer:
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
```

#### 3. Update the `create_message` caller
Update the route handler to pass `prompt_cache=kv_cache`.

```python
    if typed_req.stream:
        return StreamingResponse(
            stream_generate_response(typed_req, prompt, input_tokens, thinking_enabled, prompt_cache=kv_cache),
            media_type="text/event-stream",
        )
    else:
        return await generate_response(typed_req, prompt, input_tokens, thinking_enabled, prompt_cache=kv_cache)
```

## Verification Plan
1. Apply the changes to `main.py`.
2. As the user is testing natively, run `uv run main.py` on the Mac.
3. Make an API request to the proxy to verify the tokens successfully generate using the pre-filled cache without hitting the `TypeError`.

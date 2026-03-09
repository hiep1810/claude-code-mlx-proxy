# Fix Metal `malloc` Buffer Limit Exceeded Crash

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* The optimal chunk size for the pre-fill loop. Too small = slow, too large = crash. A safe default for 8-9B models is usually between 512 and 1024 tokens. We will configure it to 512 for safety.

### ⚠️ Potential Effects of Uncertainty
* If 512 is still too large, it might crash on very small VRAM Macs, but since it's an 8GB limit we hit, 512 tokens will definitely fit.

---

## Big Picture: Why is it crashing?

> **Junior tip:** Imagine asking a moving company to transport an entire 3-story mansion in a single flatbed truck. The truck screams, "I can't hold 12,000 tons at once!" The GPU is doing the exact same thing. We asked it to memorize a massive codebase (the prompt) in a *single* computational step, and Apple's Metal framework has a strict rule: "No single memory bag can be larger than 8GB."

**The Problem:**
In our `prefill()` logic, we do this:
```python
model(tokens[None], kv_cache)
```
If `tokens` is a massive codebase (e.g., 60,000 tokens), MLX attempts to compute and allocate the *entire* Key-Value Cache matrices for all 32 layers of the Qwen 9B model in a single memory buffer. For an 8B/9B model, 60k tokens might require 12.6 GB of continuous VRAM for the KV Cache alone. 

Macs with unified memory have plenty of total RAM, but Apple's Metal limits the *maximum size of a single allocation* (usually 4GB, 8GB, or sometimes up to 75% of total RAM via a sysctl flag, but heavily restricted by default).

**The Solution:**
We need to **chunk** the pre-filling. Instead of evaluating 60,000 tokens in one forward pass, we evaluate them in chunks (e.g., 512 tokens at a time) and tell MLX to finalize the memory (`mx.eval`) after each chunk. The overall KV Cache will still build up correctly, but the GPU only has to juggle 512 tokens' worth of active computation at any given millisecond.

---

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Implement chunking in the `prefill()` logic
Instead of passing the entire `tokens` array to the model at once, we will slice the array into chunks of 512 tokens and run a loop.

```python
    # Pre-fill logic: evaluate the prompt in the background once before starting stream_generate
    # This warms up the KV cache safely in chunks to avoid Metal malloc limits.
    def prefill():
        if kv_cache is None:
            return
            
        with mlx_lock:
            tokens = mx.array(tokenizer.encode(prompt))
            chunk_size = 512 # Safe chunk size to stay under 8GB Metal buffer limits
            
            for i in range(0, tokens.shape[0], chunk_size):
                chunk = tokens[i:i + chunk_size]
                # Forward pass for the chunk
                model(chunk[None], kv_cache)
                # Force Metal to finalize calculations and free intermediate graphs
                mx.eval([c.state for c in kv_cache])
```

## Verification Plan
1. Approve this plan.
2. I will apply the chunking code.
3. You run `uv run main.py` on your Mac.
4. Test the proxy with the same large codebase prompt. The pre-filling phase should take slightly longer (a few seconds) but should successfully complete without throwing the `[metal::malloc]` limit error.

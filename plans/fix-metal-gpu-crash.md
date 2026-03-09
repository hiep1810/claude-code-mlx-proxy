# Fix Metal GPU Thread Crash

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* None identified. The crash log `A command encoder is already encoding to this command buffer` combined with our recent multithreading changes makes it clear: MLX (specifically Apple Metal) stringently forbids concurrent GPU access across multiple threads.

### ⚠️ Potential Effects of Uncertainty
* N/A

---

## Big Picture: Why is the GPU crashing?

> **Junior tip:** Imagine the GPU as a single, incredibly fast cash register. Our recent "Server Lockup Fix" hired multiple cashiers (Threads) so the line (FastAPI) would keep moving. But right now, two cashiers are trying to punch buttons on the exact same cash register at the exact same millisecond. The register panics and crashes.

**The Problem:**
In our previous fix to stop FastAPI from freezing, we moved `predict` and `tokenization` workloads into background threads using `asyncio.to_thread` and `threading.Thread`. 

However, Claude Code often sends a request to stream text, and *while that text is streaming*, it might instantly send another request to just count tokens (`/v1/messages/count_tokens`). Because these run in separate threads, Thread A is using `mlx_lm.stream_generate` on the GPU, while Thread B suddenly asks `tokenizer.encode` to use the GPU. The Apple Metal framework throws a fatal assertion because multiple command encoders are trying to write to the GPU simultaneously.

**The Approach:**
We will implement the **Mutex (Mutual Exclusion) Pattern**.

- **Pattern Name:** Mutex / Global Lock
- **One-Line ELI5:** A bathroom key at a gas station. Only the person holding the physical key can go inside. Everyone else has to wait in line until the key is returned.
- **Why Here:** The MLX model and tokenizer are global resources that are *not* thread-safe on Apple Silicon. A single lock ensures that even if FastAPI receives 10 concurrent requests, the threads take turns accessing the MLX GPU without overlapping.
- **Real Analogy:** An airport runway. Even if five planes arrive at the exact same time, Air Traffic Control (the Lock) forces them to land one by one to avoid a crash.

---

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Define the Global Lock
Create a global `threading.Lock()` at the top of the file near the model definitions.

```python
# Global variables for model and tokenizer
model = None
tokenizer = None
mlx_lock = threading.Lock() # <-- Prevents Metal GPU crashes from concurrent access
```

#### 2. Synchronize Token Counting
Wrap the heavy parts of `count_tokens` (where the tokenizer is invoked) inside the lock. 
*(Note: Since `count_tokens` is run inside `asyncio.to_thread`, a standard `threading.Lock` is correct and safe to block the worker thread).*

```python
def count_tokens(text: str) -> int:
    try:
        if isinstance(text, str) and text.strip():
            with mlx_lock: # <-- Wait for the bathroom key
                # Run the tokenizer...
                pass
```

#### 3. Synchronize Non-Streaming Generation
Wrap the `generate` call in `generate_response`.

```python
# Use a separate wrapper to run generate inside the lock
def thread_generate():
    with mlx_lock:
        return generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        
response_text = await asyncio.to_thread(thread_generate)
```

#### 4. Synchronize Streaming Generation
Wrap the entire `stream_generate` loop inside the producer thread in `stream_generate_response`.

```python
    def producer():
        try:
            with mlx_lock: # <-- Keep the lock for the entire response generation
                for response in stream_generate(model, tokenizer, prompt=prompt, **gen_kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(response), loop)
            # Poison pill
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(e), loop)
```

## Verification Plan

### Manual Verification
1. Run the server using `uv run main.py` on the Mac server.
2. In one terminal window, start a very long streaming generation via `curl` or Claude Code.
3. While it is generating, immediately fire a `curl` request to the `/v1/messages/count_tokens` endpoint from a second terminal window.
4. Verify that:
   - The server does **not** crash with the `AGXG16GFamilyCommandBuffer` error.
   - The token counting request naturally waits (hangs) until the streaming generation finishes, and then returns successfully.

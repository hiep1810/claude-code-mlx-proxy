# Fix Server Lockup (Unkillable FastAPI)

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* None identified. The symptoms ("server stops responding", "can't ctrl+c", "timeout from claude code") are textbook examples of blocking the `asyncio` event loop.

### ⚠️ Potential Effects of Uncertainty
* N/A

---

## Big Picture: Why is the server freezing?

> **Junior tip:** FastAPI runs on an "event loop" running on a single main thread. It acts like a waiter serving multiple tables. If the waiter goes to the kitchen and cooks the meal themselves (a blocking, synchronous task), they stop checking on the other tables and ignore the manager telling them to go home (Ctrl+C).

**The Problem:**
In `main.py`, our API routes are defined as `async def create_message(...)`. 
Inside this route, we call `mlx_lm.generate(...)` and `mlx_lm.stream_generate(...)`. 
These are **synchronous**, massively CPU/GPU-heavy functions. Because there is no `await` keyword releasing control, Python's main thread gets permanently stuck calculating numbers for the AI. 
When the event loop is blocked:
1. It stops accepting new requests (Claude Code gets a timeout).
2. It stops listening to OS signals (Ctrl+C stops working).

**The Approach:**
We need to move the heavy MLX calculations off the main event loop and into background threads. 
1. For non-streaming, we can simply wrap the function in `asyncio.to_thread`.
2. For streaming, we will spin up a background thread that puts tokens into a queue, while the main thread safely reads from the queue and sends them to the client.

---

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Imports
We need to import `asyncio` and `threading` at the top of the file.

#### 2. Fix the Non-Streaming Generation
Update `generate_response` to run the model calculation in a separate thread.

```python
    # Pass generate to a background thread to prevent lockup
    response_text = await asyncio.to_thread(
        generate,
        model,
        tokenizer,
        prompt=prompt,
        **gen_kwargs,
    )
```

#### 3. Fix Token Counting
Token counting on a 100,000 token codebase can take a full second. We should thread this too in `create_message()` and `count_tokens_endpoint()`:
```python
        input_tokens = await asyncio.to_thread(count_tokens, prompt)
```

#### 4. Fix Streaming Generation (The tricky part)
We can't just `to_thread` a generator. We need to implement the **Producer-Consumer Pattern**.

- **Pattern Name:** Producer-Consumer Pattern
- **One-Line ELI5:** A chef (Producer) makes food and puts it on a counter, and a waiter (Consumer) picks it up when ready to serve it to the customer.
- **Why Here:** The MLX model can only spit out tokens synchronously (the Chef). But our web server needs to wait for them asynchronously (the Waiter) without getting blocked. They communicate safely through an `asyncio.Queue` (the counter).
- **Real Analogy:** A sushi restaurant conveyor belt. The chef makes the sushi in the back and puts it on the belt; the customer takes it off when it reaches them.

```python
async def stream_generate_response(...):
    response_id = "msg_" + str(abs(hash(prompt)))[:8]
    full_text = ""
    
    # 1. Setup the Queue
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    # 2. Define the Producer (runs in background thread)
    def producer():
        try:
            for response in stream_generate(model, tokenizer, prompt=prompt, **gen_kwargs):
                # Thread-safe way to put items in an async queue
                asyncio.run_coroutine_threadsafe(queue.put(response), loop)
            # Send a "None" poison pill to signal completion
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(e), loop)

    # 3. Start the Producer Thread
    threading.Thread(target=producer, daemon=True).start()

    # ... send SSE start events ...

    # 4. The Consumer (runs safely on main thread)
    while True:
        # Await safely yields control of the event loop back to FastAPI!
        response = await queue.get()
        
        if response is None: # Generation is done
            break
        if isinstance(response, Exception):
            raise response # Re-raise any errors from the thread
            
        full_text += response.text
        # ... yield SSE delta events ...
```

## Verification Plan

### Manual Verification
1. Run the server using `uv run main.py`.
2. In another terminal, trigger a huge prompt using `curl` or Claude Code.
3. While the model is generating, try pressing `Ctrl+C` in the server terminal. It should immediately shut down instantly, rather than locking up.
4. Check that Claude Code streams tokens normally without timeouts.

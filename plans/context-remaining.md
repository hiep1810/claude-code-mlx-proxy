# Context Remaining Feature

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty

| # | Question | Impact |
|---|----------|--------|
| 1 | How does the `mlx_lm` model object store max context length? | Different architectures (Llama, Qwen) store it differently (e.g. `max_position_embeddings`, `sliding_window`, etc.). Relying purely on the model object might be fragile. |
| 2 | Do you want this information logged to the terminal console where the server runs, or returned in the API response metadata? | If terminal, a simple `print()` is enough. If API response, we have to modify the Claude API schema slightly or add a custom header. |

### ⚠️ Potential Effects of Uncertainty
- If we extract the context length from the model object and it fails for a specific model family, the proxy might log "Unknown" or crash.
- I will assume you want it **logged to the terminal console**, and I will add a `.env` fallback variable `MAX_CONTEXT_LENGTH` just in case the model object doesn't report it properly.

---

## Big Picture: What Does This Feature Do?

> **Junior tip:** Context window is the "short-term memory limit" of an AI model. If you pass more tokens than its limit, it "forgets" the beginning of the conversation or throws an error. Knowing how much memory is left helps you avoid sending too much code at once.

**The Goal:**
Before the model generates a response, we want to calculate how many tokens the prompt takes, subtract that from the model's total context limit, and log the "context remaining" to the server console.

**The Approach:**
1. Try to read the context limit dynamically from the loaded MLX model.
2. Provide a fallback `MAX_CONTEXT_LENGTH` in `.env` if the model's architecture hides this value.
3. In `create_message()`, immediately after calculating `input_tokens`, compute the remaining tokens and print a helpful log message to the terminal.

---

## Proposed Changes

### 1. Configuration (`config.py` & `.env.example`)
Add a new optional setting so the user can hardcode the context limit if they want to override the model's default.

**`config.py`**
```python
    # Context settings
    MAX_CONTEXT_LENGTH: Optional[int] = (
        int(os.getenv("MAX_CONTEXT_LENGTH")) if os.getenv("MAX_CONTEXT_LENGTH") else None
    )
```

### 2. Context Length Extraction (`main.py`)
Add a helper function to safely extract the max context length from the `mlx_lm` model object.

```python
def get_max_context_length() -> int:
    """
    Safely try to determine the maximum context length of the loaded model.
    Checks user config first, then inspects model config attributes.
    """
    if config.MAX_CONTEXT_LENGTH is not None:
        return config.MAX_CONTEXT_LENGTH
        
    if model is None:
        return 8192 # Safe default
        
    # Try common MLX model config attributes
    try:
        m_config = model.config if hasattr(model, "config") else getattr(model, "text_model", model).config
        
        # Check standard attributes
        for attr in ["max_position_embeddings", "sliding_window", "n_positions", "max_seq_len"]:
            if hasattr(m_config, attr) and getattr(m_config, attr) is not None:
                return int(getattr(m_config, attr))
    except Exception:
        pass
        
    return 8192 # Fallback if we absolutely cannot find it
```

### 3. Server Logging (`main.py` -> `create_message`)
Right before deciding to stream or not, calculate and print the stats.

```python
        input_tokens = count_tokens(prompt)
        max_context = get_max_context_length()
        remaining = max_context - input_tokens
        
        # Log to server console
        print(f"[{request.model}] Context used: {input_tokens} / {max_context} tokens. Remaining: {remaining}")
```

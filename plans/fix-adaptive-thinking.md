# Fix 'adaptive' thinking error in claude-code-mlx-proxy

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty

| # | Question | Impact |
|---|----------|--------|
| 1 | Should "adaptive" thinking still enable the `<think>` block extraction, or turn it off? | Determines `thinking_enabled` logic. We will assume "adaptive" means we *should* enable extraction so we capture Qwen's thinking if it decides to think. |

### ⚠️ Potential Effects of Uncertainty
- If we assume wrong, we might drop useful thinking blocks or include unwanted ones, but either way it won't crash the proxy anymore.

---

## Big Picture: What Does This Fix Do?

> **Junior tip:** When an API adds a new feature (like a new option for a field), older code breaks because it was strict about what it expected. Here, FastAPI is doing its job *too well* by rejecting the new `"adaptive"` value for thinking types because our Pydantic model told it only `"enabled"` and `"disabled"` were allowed.

**The Problem:**
Claude Code recently updated how it requests thinking capabilities. Instead of just `"enabled"` or `"disabled"`, it now sends `"adaptive"`. Our `ThinkingConfig` Pydantic model in `main.py` rigidly enforces:
```python
class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled"]
```
When it receives `"adaptive"`, FastAPI throws a `422 Unprocessable Entity` error before our code even runs.

**The Solution:**
1. Update `ThinkingConfig` to accept `"adaptive"`.
2. Update the logic that checks if thinking is enabled to treat `"adaptive"` as enabled (since local Qwen models are inherently free to write chain-of-thought, and we want to capture and pass it back).

---

## Proposed Changes

### [MODIFY] [main.py](file:///d:/H%20Drive/git/claude-code-mlx-proxy/main.py)

**1. Update the Pydantic Model:**

```python
class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None
```

**2. Update `create_message()` logic:**

Modify the `thinking_enabled` check to include `"adaptive"`:

```python
        # Check if thinking is enabled or adaptive in the request
        thinking_enabled = (
            request.thinking is not None 
            and request.thinking.type in ("enabled", "adaptive")
        )
```

## Verification Plan

### Automated Tests
Currently, we don't have tests for the FastAPI route parsing, but we can verify the fix by running the server and sending a `curl` request that matches what Claude Code sends:

```bash
curl -X POST http://localhost:8888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "max_tokens": 100,
    "thinking": {"type": "adaptive", "budget_tokens": 1024},
    "messages": [
      {"role": "user", "content": "Test adaptive thinking"}
    ]
  }'
```
We expect a `200 OK` response instead of a `422 Unprocessable Entity` error.

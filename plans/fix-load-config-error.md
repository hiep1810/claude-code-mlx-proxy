# Fix `load_config` TypeError Crash

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* None identified. The stack trace clearly shows `mlx_lm.utils.load_config` expects a `pathlib.Path` or a local directory path, but we are passing the string model ID (e.g., `mlx-community/Qwen3.5-9B-OptiQ-4bit`). In older or different versions of `mlx_lm`, it handled strings gracefully. In the user's specific Mac environment, it's attempting to use the `/` operator on a string and throwing `TypeError: unsupported operand type(s) for /: 'str' and 'str'`.

### ⚠️ Potential Effects of Uncertainty
* N/A

---

## Big Picture: Why is it crashing?

> **Junior tip:** Imagine you give a delivery driver an address like "New York". The driver's GPS software expects a specific street address, so it tries to append a street name to "New York" using a special GPS merging tool. But because "New York" is just a raw string of text and not a structured map object, the GPS crashes.

**The Problem:**
In our `main.py` startup function (`lifespan`), we added this line to support Idea 8 (Async Pre-filling):

```python
model_config = load_config(config.MODEL_NAME)
```

`config.MODEL_NAME` is a string like `"mlx-community/Qwen3.5-9B-OptiQ-4bit"`.
Inside `mlx_lm` on the Mac server, the `load_config` function is doing this:

```python
with open(model_path / "config.json", "r") as f:
```

It expects `model_path` to be a `pathlib.Path` object (where the `/` operator joins paths). Because `model_path` is a string, Python throws an error when it tries to divide a string by a string (`/`).

**The Solution:**
Instead of removing `load_config` (which we need for the cache builder in Idea 8), we need to feed it the correct local file path. The `mlx_lm.load` function actually handles downloading the model and figuring out the path automatically. We can use the `huggingface_hub` tools, or simply rely on the fact that `model.config` is already available after `mlx_lm.load` is called!

Wait, `mlx_lm.load` already gives us the model. We can just extract the configuration directly from the loaded model instead of trying to hit the file system again.

---

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Remove `load_config` import.
Since we don't need to parse the JSON file manually, we can remove the explicit `load_config` import that was added.

```diff
-from mlx_lm.utils import load_config
```

#### 2. Remove `load_config` call in `lifespan`.
Remove the line that causes the crash during server startup.

```diff
-    model_config = load_config(config.MODEL_NAME)
```

#### 3. Determine `num_layers` dynamically inside the route.
In Idea 8's Async Pre-filling logic inside the `create_message` route, we need `num_layers` to initialize the `KVCache`. We will safely inspect the loaded model object instead of relying on a raw JSON config dictionary.

```python
    # KV cache state
    from mlx_lm.models.base import KVCache
    
    # Safely determine the number of layers
    if hasattr(model, "layers"):
        num_layers = len(model.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "text_model") and hasattr(model.text_model, "layers"):
        num_layers = len(model.text_model.layers)
    else:
        print("Warning: Could not dynamically determine num_layers. Defaulting to 32.")
        num_layers = 32
        
    kv_cache = [KVCache() for _ in range(num_layers)]
```

## Verification Plan

### Manual Verification
1. Approve this plan.
2. We will apply the code changes.
3. You will run `uv run main.py` on your Mac server.
4. Verify the server successfully passes the `Loading MLX model:` phase and logs `Started server process`.

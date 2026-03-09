# Fix `KVCache` ImportError

## 🛠️ Execution Status: Draft

### 🔍 Points of Uncertainty
* MLX `KVCache` internal pathing changes frequently between minor `mlx_lm` versions (e.g., it might be in `mlx_lm.models.cache` or dynamically attached to the model).
* Because we are not on the deployment environment (Mac), we cannot easily inspect the exact library structure via python CLI to find where `KVCache` moved.

### ⚠️ Potential Effects of Uncertainty
* If we hardcode another import path, it might break on the next update.

---

## Big Picture: Why is it crashing?

> **Junior tip:** When building a racing car, you have two choices. You can buy the pre-assembled engine from the manufacturer (`generate`), or you can try to buy the individual pistons and valves from the factory floor (`KVCache`). The problem with buying individual parts is that the factory reorganizes their shelves every month, so your map to the pistons (`import KVCache from base`) breaks!

**The Problem:**
In our Async Pre-filling implementation (Idea 8), we tried to manually build the model's memory (the KV Cache) by importing `KVCache` directly from the internal files of the `mlx_lm` library:

```python
from mlx_lm.models.base import KVCache
```

However, the MLX team is moving fast, and in the version installed on your Mac server, `KVCache` is not located in `mlx_lm.models.base.py`. This causes Python to throw an `ImportError` when the `/v1/messages` endpoint is hit.

**The Solution:**
Instead of trying to guess where the MLX maintainers hid the `KVCache` class, we should ask the model to create its own cache. 

In recent MLX versions, `mlx_lm.generate_step` or similar APIs automatically handle caching, OR the model itself has a method to initialize its cache, such as `mlx_lm.models.cache.make_prompt_cache(model)`.

Even better, we can actually use MLX's built-in caching helper. In `mlx_lm`, the standard way to get a cache for a prompt without hardcoding the class is:

```python
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
```

By using the official helper function, we let the library worry about where the classes are stored and how many layers the model has!

---

## Proposed Changes

### [MODIFY] `main.py`

#### 1. Import `make_prompt_cache`
In the `create_message` route, inside the `cache_builder_worker` thread, we will replace the hardcoded `KVCache` loop with the official helper function.

```python
    def cache_builder_worker():
        """Background thread that tokenizes and evaluates prompt chunks."""
        nonlocal kv_cache
        
        # Use MLX's official helper to build the cache shape dynamically
        try:
            from mlx_lm.models.cache import make_prompt_cache
            kv_cache = make_prompt_cache(model)
        except ImportError:
            # Fallback for older versions if make_prompt_cache is missing
            try:
                from mlx_lm.models.base import KVCache
                num_layers = len(model.layers) if hasattr(model, "layers") else 32
                kv_cache = [KVCache() for _ in range(num_layers)]
            except ImportError:
                print("Warning: Could not initialize MLX KV Cache. Pre-filling disabled.")
                kv_cache = None
                return
```

#### 2. Handle API edge-cases
If `make_prompt_cache` works, it returns an initialized array of caches for the layers.
We also need to make sure we don't crash the server if *both* imports fail; we should simply gracefully degrade back to standard generation without pre-filling by setting `kv_cache = None`.

## Verification Plan

### Manual Verification
1. Approve this plan.
2. I will apply the changes to `main.py`.
3. You run `uv run main.py` on your Mac server.
4. Test the proxy with a regular prompt. It should successfully import the cache module and run the async pre-fill logic without an `ImportError`.

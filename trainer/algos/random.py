import numpy as np


def init(rng: np.random.Generator, action_spec: dict, **kwargs):
    return {"rng": rng, "action_spec": action_spec}


def act(state, obs):
    rng = state["rng"]
    s = state["action_spec"]
    kind = s.get("kind", "discrete")

    if kind == "discrete":
        n = int(s["n"])
        return np.array(rng.integers(0, n), dtype=np.int64)

    if kind == "box":
        shape = tuple(int(x) for x in s["shape"])
        dtype = np.dtype(s.get("dtype", "float32"))
        low = float(s.get("low", -1.0))
        high = float(s.get("high", 1.0))
        return rng.uniform(low, high, size=shape).astype(dtype, copy=False)

    raise ValueError(f"unknown action kind: {kind}")

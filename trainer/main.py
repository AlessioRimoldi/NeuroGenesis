import argparse
import json
import os
import time
import importlib

import grpc
import mlflow
import numpy as np
import redis

from trainer import env_pb2, env_pb2_grpc


def obs_from_proto(msg):
    arr = np.frombuffer(msg.tensor, dtype=np.dtype(msg.dtype))
    if msg.shape:
        arr = arr.reshape(list(msg.shape))
    return arr


def action_to_proto(arr: np.ndarray):
    arr = np.asarray(arr)
    return env_pb2.Action(
        tensor=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def sample_action(action_spec: dict):
    kind = action_spec.get("kind", "discrete")
    if kind == "discrete":
        n = int(action_spec["n"])
        a = np.array(np.random.randint(0, n), dtype=np.int64)
        return a
    if kind == "box":
        shape = tuple(int(x) for x in action_spec["shape"])
        dtype = np.dtype(action_spec.get("dtype", "float32"))
        low = action_spec.get("low", -1.0)
        high = action_spec.get("high", 1.0)
        a = np.random.uniform(low, high, size=shape).astype(dtype, copy=False)
        return a
    raise ValueError(f"unknown action kind: {kind}")


def load_spec(path: str | None, spec_json: str | None):
    if spec_json:
        return json.loads(spec_json)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def connect_redis(url: str):
    r = redis.Redis.from_url(url, decode_responses=True)
    r.ping()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=str, default=None)
    p.add_argument("--spec-json", type=str, default=None)
    p.add_argument("--rerun", type=str, default=None)
    p.add_argument("--override", type=str, action="append", default=[])
    args = p.parse_args()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://registry:5000")
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    env_addr = os.environ.get("ENV_ADDR", "env:50051")

    r = connect_redis(redis_url)

    if args.rerun:
        raw = r.get(f"trial:{args.rerun}:spec")
        if raw is None:
            raise SystemExit(f"missing spec in redis for run_id={args.rerun}")
        spec = json.loads(raw)
        spec.setdefault("meta", {})["rerun_of"] = args.rerun
    else:
        spec = load_spec(args.spec, args.spec_json)

    for ov in args.override:
        k, v = ov.split("=", 1)
        spec[k] = json.loads(v) if (v.startswith("{") or v.startswith("[") or v in ("true", "false", "null") or v.replace(".", "", 1).isdigit()) else v

    task = spec.get("task", "gym:CartPole-v1")
    seed = int(spec.get("seed", 0))
    episodes = int(spec.get("episodes", 10))
    max_steps = int(spec.get("max_steps", 500))
    action_spec = spec.get("action", {"kind": "discrete", "n": 2})
    experiment = spec.get("experiment", "neurogenesis")
    run_name = spec.get("run_name", task)
    algo_name = spec.get("algo", "random")
    algo_kwargs = spec.get("algo_kwargs", {})
    algo_mod = importlib.import_module(f"trainer.algos.{algo_name}")
    rng = np.random.default_rng(seed)
    algo_state = algo_mod.init(rng=rng, action_spec=action_spec, **algo_kwargs)


    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    channel = grpc.insecure_channel(env_addr)
    stub = env_pb2_grpc.EnvStub(channel)

    stub.Seed(env_pb2.SeedRequest(seed=seed))
    stub.Reset(env_pb2.ResetRequest())

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        spec_out = dict(spec)
        spec_out.setdefault("env_addr", env_addr)
        spec_out.setdefault("mlflow_tracking_uri", mlflow_uri)

        mlflow.log_params(
            {
                "task": task,
                "seed": seed,
                "episodes": episodes,
                "max_steps": max_steps,
                "env_addr": env_addr,
                "action_kind": action_spec.get("kind", "unknown"),
            }
        )
        mlflow.log_text(json.dumps(spec_out, indent=2), "trial_spec.json")

        r.set(f"trial:{run_id}:spec", json.dumps(spec_out))
        r.lpush("trials", run_id)

        t0 = time.time()
        for ep in range(episodes):
            obs = obs_from_proto(stub.Reset(env_pb2.ResetRequest()).observation)
            ep_ret = 0.0
            ep_len = 0

            for _ in range(max_steps):
                a = algo_mod.act(algo_state, obs)
                resp = stub.Step(env_pb2.StepRequest(action=action_to_proto(a)))
                obs = obs_from_proto(resp.observation)
                ep_ret += float(resp.reward)
                ep_len += 1
                if resp.done:
                    break

            mlflow.log_metric("episode_return", ep_ret, step=ep)
            mlflow.log_metric("episode_length", ep_len, step=ep)

        mlflow.log_metric("wall_time_sec", time.time() - t0)

        print("run_id", run_id)


if __name__ == "__main__":
    main()

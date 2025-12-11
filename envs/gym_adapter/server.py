import argparse
from concurrent import futures

import grpc
import numpy as np
import gymnasium.spaces as spaces

import env_pb2
import env_pb2_grpc
from wrappers.gym_adapter import GymAdapter


def obs_to_proto(obs):
    arr = np.asarray(obs)
    return env_pb2.Observation(
        tensor=arr.tobytes(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def action_from_proto(msg):
    arr = np.frombuffer(msg.tensor, dtype=msg.dtype)
    if msg.shape:
        arr = arr.reshape(msg.shape)
    return arr


def space_to_spec(space):
    if isinstance(space, spaces.Discrete):
        return env_pb2.SpaceSpec(shape=[1], dtype="int64")
    if hasattr(space, "shape"):
        return env_pb2.SpaceSpec(shape=list(space.shape), dtype="float32")
    return env_pb2.SpaceSpec()


class EnvServicer(env_pb2_grpc.EnvServicer):
    def __init__(self, env_id: str):
        self._env = GymAdapter(env_id)

    def Reset(self, request, context):
        obs = self._env.reset()
        return env_pb2.ResetResponse(observation=obs_to_proto(obs))

    def Step(self, request, context):
        action_arr = action_from_proto(request.action)
        space = self._env.action_space()
        if isinstance(space, spaces.Discrete):
            action = int(action_arr.item())
        else:
            action = action_arr
        obs, reward, done, info = self._env.step(action)
        return env_pb2.StepResponse(
            observation=obs_to_proto(obs),
            reward=float(reward),
            done=bool(done),
        )

    def Seed(self, request, context):
        self._env.seed(request.seed)
        return env_pb2.SeedResponse()

    def Spec(self, request, context):
        return env_pb2.SpecResponse(
            observation=space_to_spec(self._env.observation_space()),
            action=space_to_spec(self._env.action_space()),
        )


def serve(env_id: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    env_pb2_grpc.add_EnvServicer_to_server(EnvServicer(env_id), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="gym:CartPole-v1")
    p.add_argument("--port", type=int, default=50051)
    args = p.parse_args()
    _, env_id = args.task.split(":", 1)
    serve(env_id, args.port)

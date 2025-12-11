import grpc
import numpy as np

import env_pb2
import env_pb2_grpc


def obs_from_proto(msg):
    arr = np.frombuffer(msg.tensor, dtype=msg.dtype)
    if msg.shape:
        arr = arr.reshape(msg.shape)
    return arr


def action_to_proto(a: int):
    arr = np.array([a], dtype=np.int64)
    return env_pb2.Action(
        tensor=arr.tobytes(),
        shape=[],
        dtype=str(arr.dtype),
    )


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = env_pb2_grpc.EnvStub(channel)

    stub.Seed(env_pb2.SeedRequest(seed=0))
    reset_resp = stub.Reset(env_pb2.ResetRequest())
    obs = obs_from_proto(reset_resp.observation)
    print("reset obs", obs)

    for t in range(1000):
        action = np.random.randint(0, 2)
        step_resp = stub.Step(env_pb2.StepRequest(action=action_to_proto(action)))
        obs = obs_from_proto(step_resp.observation)
        print(t, action, obs, step_resp.reward, step_resp.done)
        if step_resp.done:
            reset_resp = stub.Reset(env_pb2.ResetRequest())
            obs = obs_from_proto(reset_resp.observation)


if __name__ == "__main__":
    main()

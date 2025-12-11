from wrappers.gym_adapter import GymAdapter


def main():
    env = GymAdapter("CartPole-v1",render_mode="human")
    obs = env.reset()
    for t in range(1000):
        action = env.action_space().sample()
        obs, reward, done, info = env.step(action)
        print(t, obs, reward, done)
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
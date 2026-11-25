"""Lockstep throughput: steps/s and agent-ticks/s against agent count (random MineRL-style actions)."""

import argparse
import random
import time

from neurogenesis.mc import McClient
from neurogenesis.mc.actions import KEYS, action, agent, arena


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25570)
    p.add_argument("--agents", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--ticks", type=int, default=1, help="game ticks per step")
    p.add_argument("--craftable", action="store_true", help="also compute the craftable-recipe mask")
    args = p.parse_args()
    rng = random.Random(0)
    with McClient(args.host, args.port) as c:
        print(f"{'agents':>6} {'steps/s':>9} {'agent-ticks/s':>14} {'server ms/step':>15}")
        for n in args.agents:
            c.reset(seed=0, arena=arena((48, 48), food_count=8, smell=True),
                    agents=[agent(f"bench{i}", inventory=[("minecraft:oak_log", 8), ("minecraft:cobblestone", 16)])
                            for i in range(n)],
                    craftable=args.craftable)
            server_us = 0
            t0 = time.perf_counter()
            for _ in range(args.steps):
                acts = [action(i, [k for k in KEYS[:7] if rng.random() < 0.3],
                               camera=(rng.uniform(-10, 10), rng.uniform(-5, 5))) for i in range(n)]
                server_us += c.step(acts, ticks=args.ticks).server_step_us
            dt = time.perf_counter() - t0
            rate = args.steps / dt
            print(f"{n:>6} {rate:>9.0f} {rate * n * args.ticks:>14.0f} {server_us / args.steps / 1000:>15.2f}")


if __name__ == "__main__":
    main()

"""Watch agents in real time with a Minecraft: Java Edition 26.3 client.

Runs smell-following "worms" (the steering breakthrough) in a food arena, paced to real time with `/ng pace`
(set over RCON, restored on exit). Join the server at localhost:25565 *before* or *while* this runs, then e.g.
`/tp @s 12 72 12` or `/spectate worm0` (you need op: `docker exec neurogenesis-mc rcon-cli op <name>`).

    python scripts/watch.py --agents 3 --tps 20
"""

from __future__ import annotations

import argparse
import time

from neurogenesis.mc import McClient
from neurogenesis.mc.actions import action, agent, arena
from neurogenesis.mc.rcon import Rcon


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25570, help="control port")
    p.add_argument("--rcon-port", type=int, default=25575)
    p.add_argument("--rcon-password", default="neurogenesis")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--food", type=int, default=4, help="food items in the arena (respawn when picked up)")
    p.add_argument("--size", type=int, default=24, help="arena side in blocks")
    p.add_argument("--tps", type=float, default=20.0, help="ticks per second (20 = real time, 0 = unlimited)")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    rcon = None
    try:
        rcon = Rcon(args.host, args.rcon_port, args.rcon_password)
        print(rcon.command(f"ng pace {args.tps:g}"))
    except OSError as e:
        print(f"RCON unavailable ({e}); pacing in Python instead")

    half = args.size / 2
    agents = [agent(f"worm{i}", pos=(half + 0.5 + 2 * (i - args.agents // 2), 64, half + 0.5), food=10)
              for i in range(args.agents)]
    collected = [0] * args.agents
    try:
        with McClient(args.host, args.port) as mc:
            r = mc.reset(seed=args.seed, agents=agents,
                         arena=arena((args.size, args.size), food_count=args.food, food_respawn=True,
                                     food_respawn_delay=40, smell=True))
            print(f"{args.agents} worm(s) running at {args.tps:g} ticks/s - Ctrl+C to stop")
            last_print = time.monotonic()
            while True:
                acts = []
                for obs in r.obs:
                    s = obs.smell
                    turn = -15.0 if s.left > s.right else 15.0   # turn toward the stronger antenna
                    acts.append(action(obs.agent_id, ["forward"], camera=(turn, 0)))
                t0 = time.monotonic()
                r = mc.step(acts)
                for obs in r.obs:
                    collected[obs.agent_id] += sum(1 for e in obs.events if e.type == e.PICKED_UP)
                if rcon is None and args.tps > 0:                # fallback pacing
                    time.sleep(max(0.0, 1.0 / args.tps - (time.monotonic() - t0)))
                if time.monotonic() - last_print > 5:
                    last_print = time.monotonic()
                    print(f"tick {r.episode_tick:6d}  food collected: {collected}")
    except KeyboardInterrupt:
        pass
    finally:
        if rcon is not None:
            try:
                print(rcon.command("ng pace 0"))
            finally:
                rcon.close()


if __name__ == "__main__":
    main()

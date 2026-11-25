# NeuroGenesis

An implementation of the ideas in Max Bennett's *A Brief History of Intelligence*: the five breakthroughs
(**steering, reinforcing, simulating, mentalizing, speaking**) grown as learning agents inside Minecraft.

Agents are **server-side players** on a Fabric server. They are simulated by the game's own physics and act
through human-like keyboard actions. Python steps the world in **lockstep** and receives each agent's state, plus
custom senses such as a diffusing *smell* field.

```
 Python (RL code)                           Docker: Fabric 26.3 dedicated server
 ┌──────────────────────┐  TCP + protobuf   ┌──────────────────────────────────────────────┐
 │ neurogenesis.mc      │ ───────────────►  │ NeuroGenesis mod                             │
 │  McClient.reset/step │  Step{ticks,      │  Lockstep: server waits between ticks        │
 │  actions / configs   │   actions[]}      │  AgentPlayer × N (vanilla physics)           │
 │                      │ ◄───────────────  │  ClientEmulator: keys → vanilla handlers     │
 └──────────────────────┘  StepResult{obs}  │  Arena · food · SmellField · observations    │
                                            └──────────────────────────────────────────────┘
```

## How it works

- **Lockstep.** While a Python client is connected, the server stops at every tick boundary until the next
  request authorises more ticks. The game runs exactly as fast as Python steps it, with no wall-clock pacing.
  Console and RCON keep working while it is paused. With no client attached, the server ticks normally.
- **Agents** (`mod/.../agent`). Each agent is a `ServerPlayer` with a connection that has no client behind it.
  Unlike a human player, whose movement is simulated by their own client, agents are fully simulated on the
  server.
- **Actions** (`mod/.../input`). These are MineRL-style held keys (`forward back left right jump sneak sprint
  attack use`), plus camera, hotbar, drop, swap-hands, inventory open/close and `click_slot`. A small *client
  emulator* reproduces what the vanilla client does with key presses: crosshair targeting, block-breaking
  timing, use-item repeat, sprint rules, sneaking and item-use slowdown. It then passes the resulting "packets"
  to the server's own handlers, so agents get exactly the server-side validation that human players get.
- **Observations** (`mod/.../obs`). Python receives the full state of each agent and decides what each agent
  may perceive:
  - pose and velocity, health, food, saturation and air;
  - the inventory (41 slots) and any open container;
  - the crosshair target and the two smell "antennas";
  - events: damage, death, consumed, picked up, block broken, crafted, attacked.
- **Arena** (`mod/.../arena`). Each reset builds a seeded walled arena in a void world and places food (items or
  blocks). Food can respawn when eaten.
- **Smell** (`mod/.../sense`). This is a 2-D diffusion and decay field emitted by food, with walls that let no
  smell through. It is sampled at two antennas to the left and right of the head, which is enough for
  bilateral chemotaxis: the "steering" breakthrough.

## Quick start

Prerequisites: Docker, a JDK ≥ 17 to start Gradle (Gradle downloads the JDK 25 that Minecraft 26.3 needs), and
Python ≥ 3.11.

```bash
make py          # pip install -e python[dev]
make up          # build the mod, start the server in Docker, wait for the control port
make itest       # integration tests against the running server
make bench       # lockstep throughput
make down
```

`make up` sets `EULA=TRUE` in `docker/mc/compose.yaml`. That means you accept
[Mojang's EULA](https://aka.ms/MinecraftEULA). The server and the agents need no Minecraft account. Watching
the world with the game client requires a purchased Minecraft: Java Edition. Connect to `localhost:25565`;
humans join in spectator mode.

```python
from neurogenesis.mc import McClient
from neurogenesis.mc.actions import action, agent, arena

with McClient() as mc:
    r = mc.reset(seed=3, arena=arena((24, 24), food_count=1, food_min_dist=8, smell=True),
                 agents=[agent("worm", pos=(12.5, 64, 12.5), food=10)])
    for _ in range(600):                      # follow the smell gradient to the food
        s = r.obs[0].smell
        turn = -15 if s.left > s.right else 15
        r = mc.step([action(0, ["forward"], camera=(turn, 0))], ticks=2)
        if any(e.type == e.PICKED_UP for e in r.obs[0].events):
            break
```

## Repository layout

| Path | Contents |
|---|---|
| `proto/neurogenesis/mc/v1/` | Protocol schema (length-prefixed protobuf over TCP) |
| `mod/` | Fabric mod (Gradle, Java 25): `net`, `sim` (lockstep, episodes), `agent`, `input`, `arena`, `sense`, `obs`, `mixin` |
| `python/` | `neurogenesis` package: client, action/config helpers, generated protobuf, tests |
| `docker/mc/` | Docker Compose for the dedicated server (`itzg/minecraft-server`) |
| `scripts/` | Protobuf generation, benchmark |

The development server runs without Docker: `make dev-server` (or `cd mod && ./gradlew runServer`, plus
`-PwithCarpet` to load Carpet for debugging).

## Validation

- **Java unit tests:** the smell field (`./gradlew test`): mass conservation, symmetry, the analytic decay
  length and stability.
- **Integration tests** (`make itest`) against a live server, compared with Minecraft Wiki values:

| Check | Result |
|---|---|
| Walk / sprint / sneak speed | 0.216 / 0.281 / 0.065 blocks/tick |
| Sprint when food ≤ 6 | Refused |
| Jump apex | 1.252 blocks |
| Grass block mined by hand | 18 ticks |
| Apple | Eaten in 32 ticks, +4 food |
| Crafting | Planks crafted with `click_slot` in the 2×2 grid |
| Death | Terminal, or respawn if configured |
| Punch between agents | Damage and knockback |
| End-to-end chemotaxis | A smell-following agent reaches the food |

- **Throughput** (laptop, 1 tick per step, random actions, Docker):

| Agents | Steps/s | Agent-ticks/s |
|---|---|---|
| 1 | ≈ 460–530 | ≈ 500 |
| 16 | ≈ 240–300 | ≈ 3,900–4,800 |
| 64 | ≈ 130 | ≈ 8,200 |

## Roadmap

- **M1 (this):** headless simulation core: lockstep, agents, actions, arena, smell, Docker.
- **M2:** vision. First the "host client": the same mod inside a client's integrated server, rendering every
  agent's first-person view per tick. Measure its limits, then decide whether separate camera clients are needed.
- **M3:** RL framework integration: Gymnasium and PettingZoo wrappers, task modules with rewards (starting with
  steering), and vectorisation across servers.

Known gaps: piercing weapons (spears) and custom attack ranges are not emulated yet, bit-exact replay is not
guaranteed (vanilla has asynchronous chunk and lighting work), and human spectators may time out if Python
pauses for more than about 30 s.

## Licences

Parts of the agent code are adapted from [Carpet](https://github.com/gnembon/fabric-carpet) (MIT); see
`mod/THIRD_PARTY_NOTICES`. Minecraft is © Mojang; do not redistribute Minecraft jars. The Docker image downloads
them at startup.

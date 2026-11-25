"""Agent behaviour against a live server (pytest -m server). Reference values: Minecraft Wiki."""

import math

import pytest

from neurogenesis.mc.actions import action, agent, arena

pytestmark = pytest.mark.server

CENTER = (16.5, 64.0, 16.5)


def _id(vocab, name):
    return list(vocab).index(name)


def _events(obs, kind):
    from neurogenesis.mc.v1 import observation_pb2

    t = observation_pb2.Event.Type.Value(kind)
    return [e for e in obs.events if e.type == t]


def _speed(client, keys, ticks=20, **agent_kw):
    """Blocks/tick along +Z after accelerating for `ticks` ticks (yaw 0 faces +Z)."""
    client.reset(seed=1, arena=arena((32, 64)), agents=[agent("a", pos=(16.5, 64, 2.5), yaw=0, **agent_kw)])
    client.step([action(0, keys)], ticks=ticks)
    z0 = client.step([action(0, keys)], ticks=1).obs[0].pos.z
    r = client.step([action(0, keys)], ticks=ticks)
    return (r.obs[0].pos.z - z0) / ticks, r.obs[0]


def test_walk_sprint_sneak_speeds(client):
    walk, _ = _speed(client, ["forward"])
    sprint, obs = _speed(client, ["forward", "sprint"])
    assert obs.sprinting
    sneak, obs = _speed(client, ["forward", "sneak"])
    assert obs.crouching
    assert walk == pytest.approx(0.2158, rel=0.02)
    assert sprint == pytest.approx(0.2806, rel=0.02)
    assert sneak == pytest.approx(0.0648, rel=0.05)


def test_no_sprint_when_hungry(client):
    speed, obs = _speed(client, ["forward", "sprint"], food=6)
    assert not obs.sprinting
    assert speed == pytest.approx(0.2158, rel=0.02)


def test_jump_height(client):
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER)])
    client.step(ticks=5)
    y0 = client.step(ticks=1).obs[0].pos.y
    apex = y0
    r = client.step([action(0, ["jump"])], ticks=1)
    for _ in range(15):
        apex = max(apex, r.obs[0].pos.y)
        r = client.step(ticks=1)
    assert apex - y0 == pytest.approx(1.2522, abs=0.01)
    assert r.obs[0].on_ground


def test_mine_grass_by_hand(client):
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER, yaw=0)])
    r = client.step([action(0, camera=(0, 90))], ticks=1)  # look straight down
    grass = _id(client.spec.block_vocab, "minecraft:grass_block")
    assert r.obs[0].crosshair.block_id == grass
    for tick in range(1, 60):
        r = client.step([action(0, ["attack"])], ticks=1)
        broken = _events(r.obs[0], "BLOCK_BROKEN")
        if broken:
            break
    assert broken and broken[0].item_id == grass
    # Grass block, hardness 0.6, by hand: 0.9 s = 18 ticks.
    assert tick == pytest.approx(18, abs=1)


def test_eat_apple(client):
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER, food=10, inventory=[("minecraft:apple", 2, 0)])])
    used = None
    for tick in range(1, 60):
        r = client.step([action(0, ["use"])], ticks=1)
        if _events(r.obs[0], "CONSUMED"):
            used = tick
            break
    assert used == pytest.approx(32, abs=2)  # eating takes 1.6 s
    assert r.obs[0].food == 14
    apple = _id(client.spec.item_vocab, "minecraft:apple")
    assert r.obs[0].inv_item[0] == apple and r.obs[0].inv_count[0] == 1


def test_craft_planks_with_click_slot(client):
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER, inventory=[("minecraft:oak_log", 1, 0)])])
    # Player inventory menu: 0 = result, 1-4 = 2x2 grid, 9-35 = main, 36-44 = hotbar.
    r = client.step([action(0, inventory="open", clicks=[(36, 0), (1, 0)])])
    planks = _id(client.spec.item_vocab, "minecraft:oak_planks")
    c = r.obs[0].container
    assert c.open and c.menu_type == -1
    assert c.slot_item[0] == planks and c.slot_count[0] == 4
    r = client.step([action(0, clicks=[(0, 0, "quick_move")])])
    crafted = _events(r.obs[0], "CRAFTED")
    assert crafted and crafted[0].item_id == planks and crafted[0].amount == 4
    r = client.step([action(0, inventory="close")])
    assert not r.obs[0].container.open
    assert planks in r.obs[0].inv_item


def test_fall_death_is_terminal(client):
    r = client.reset(seed=1, arena=arena(), agents=[agent("a", pos=(16.5, 84, 16.5), health=4)], settle_ticks=1)
    for _ in range(60):
        r = client.step(ticks=1)
        if _events(r.obs[0], "DEATH"):
            break
    assert _events(r.obs[0], "DEATH")
    r = client.step(ticks=1)
    assert r.obs[0].terminated and not r.obs[0].alive


def test_respawn_on_death(client):
    client.reset(
        seed=1, arena=arena(), agents=[agent("a", pos=(16.5, 84, 16.5), health=4, respawn_on_death=True)], settle_ticks=1
    )
    deaths = 0
    for _ in range(60):
        r = client.step(ticks=1)
        deaths += len(_events(r.obs[0], "DEATH"))
    assert deaths == 1
    assert r.obs[0].alive and not r.obs[0].terminated and r.obs[0].health == 20


def test_punch_knocks_back_other_agent(client):
    client.reset(
        seed=1,
        arena=arena(),
        agents=[agent("a", pos=(16.5, 64, 15.0), yaw=0), agent("b", pos=(16.5, 64, 17.0), yaw=180)],
    )
    client.step(ticks=5)
    z0 = client.step(ticks=1).obs[1].pos.z
    r = client.step([action(0, ["attack"])], ticks=1)
    assert _events(r.obs[1], "DAMAGE")
    r = client.step(ticks=5)
    assert r.obs[1].pos.z > z0 + 0.2  # pushed away from the attacker
    assert r.obs[1].health < 20


def test_follow_smell_to_food(client):
    """Bilateral chemotaxis (the 'steering' breakthrough): turn toward the stronger antenna, walk forward."""
    r = client.reset(
        seed=3,
        arena=arena((24, 24), food_count=1, food_min_dist=8, smell=True),
        agents=[agent("worm", pos=(12.5, 64, 12.5), yaw=0, food=10)],
    )
    for _ in range(600):
        s = r.obs[0].smell
        # Minecraft yaw increases clockwise seen from above, so turning left is a negative dyaw.
        turn = -15.0 if s.left > s.right else 15.0
        r = client.step([action(0, ["forward"], camera=(turn, 0))], ticks=2)
        if _events(r.obs[0], "PICKED_UP"):
            break
    assert _events(r.obs[0], "PICKED_UP"), "agent never reached the food"

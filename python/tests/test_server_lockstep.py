"""Lockstep behaviour against a live server (pytest -m server)."""

import re
import time

import pytest

pytestmark = pytest.mark.server


def _gametime(rcon) -> int:
    return int(re.search(r"(\d+)", rcon.command("time query gametime")).group(1))


def test_hello_spec(client):
    spec = client.spec
    assert spec.minecraft_version == "26.3"
    assert "minecraft:apple" in spec.item_vocab
    assert spec.block_vocab[0] == "minecraft:air"


def test_step_advances_exact_ticks(client):
    first = client.reset(seed=1, settle_ticks=0)
    tick = first.tick
    for n in (1, 1, 5, 20, 1):
        result = client.step(ticks=n)
        assert result.tick == tick + n
        tick = result.tick
    assert result.episode_tick == 28


def test_paused_while_python_idles(client, rcon):
    client.reset(seed=1, settle_ticks=0)
    before = _gametime(rcon)          # RCON is served while the server waits at the barrier
    time.sleep(1.0)
    assert _gametime(rcon) == before  # no ticks without a step
    client.step(ticks=10)
    assert _gametime(rcon) == before + 10


def test_step_before_reset_is_rejected():
    from neurogenesis.mc import McClient, McError

    from conftest import HOST, PORT

    with McClient(HOST, PORT) as c:
        with pytest.raises(McError) as err:
            c.step()
        assert err.value.code_name == "NOT_READY"


def test_throughput_single_ticks(client):
    client.reset(seed=1, settle_ticks=0)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        client.step(ticks=1)
    rate = n / (time.perf_counter() - t0)
    print(f"\nempty-world lockstep: {rate:.0f} steps/s")
    assert rate > 200

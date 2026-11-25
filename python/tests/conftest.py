import os

import pytest

HOST = os.environ.get("NG_HOST", "127.0.0.1")
PORT = int(os.environ.get("NG_CONTROL_PORT", "25570"))
RCON_PORT = int(os.environ.get("NG_RCON_PORT", "25575"))
RCON_PASSWORD = os.environ.get("NG_RCON_PASSWORD", "neurogenesis")


@pytest.fixture
def client():
    from neurogenesis.mc import McClient

    c = McClient(HOST, PORT, connect_timeout=10.0, timeout=60.0)
    yield c
    c.close()


@pytest.fixture
def rcon():
    from neurogenesis.mc.rcon import Rcon

    r = Rcon(HOST, RCON_PORT, RCON_PASSWORD)
    yield r
    r.close()

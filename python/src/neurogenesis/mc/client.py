"""Blocking lockstep client for the NeuroGenesis server mod."""

from __future__ import annotations

import socket
import time
from collections.abc import Iterable

from neurogenesis.mc.framing import read_frame, write_frame
from neurogenesis.mc.v1 import action_pb2, arena_pb2, control_pb2

PROTOCOL_VERSION = 2
DEFAULT_PORT = 25570


class McError(RuntimeError):
    """The server answered a request with an Error."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.code_name = control_pb2.Error.Code.Name(code)
        super().__init__(f"{self.code_name}: {message}")


class McClient:
    """One lockstep session. Every call blocks until the server has simulated the requested ticks."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        connect_timeout: float = 60.0,
        timeout: float | None = 300.0,
        client_name: str = "neurogenesis-python",
    ):
        self._sock = _connect_with_retry(host, port, connect_timeout)
        self._sock.settimeout(timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._next_id = 1
        self.spec: control_pb2.Spec = self._call(
            control_pb2.Request(hello=control_pb2.Hello(protocol_version=PROTOCOL_VERSION, client_name=client_name))
        ).spec

    # ------------------------------------------------------------------ requests

    def reset(
        self,
        *,
        seed: int = 0,
        arena: arena_pb2.ArenaConfig | None = None,
        agents: Iterable[arena_pb2.AgentConfig] = (),
        settle_ticks: int = 5,
        max_episode_ticks: int = 0,
        craftable: bool = False,
    ) -> control_pb2.StepResult:
        """Starts an episode. ``craftable=True`` fills ``AgentObs.craftable`` (recipe-book highlights)."""
        reset = control_pb2.Reset(
            seed=seed,
            agents=list(agents),
            settle_ticks=settle_ticks,
            max_episode_ticks=max_episode_ticks,
            obs=control_pb2.ObsConfig(craftable=craftable),
        )
        if arena is not None:
            reset.arena.CopyFrom(arena)
        return self._call(control_pb2.Request(reset=reset)).step_result

    def step(
        self, actions: Iterable[action_pb2.AgentAction] = (), *, ticks: int = 1
    ) -> control_pb2.StepResult:
        step = control_pb2.Step(ticks=ticks, actions=list(actions))
        return self._call(control_pb2.Request(step=step)).step_result

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._call(control_pb2.Request(close=control_pb2.Close()))
        except (OSError, ConnectionError, McError):
            pass
        finally:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> McClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ plumbing

    def _call(self, request: control_pb2.Request) -> control_pb2.Response:
        if self._sock is None:
            raise ConnectionError("client is closed")
        request.request_id = self._next_id
        self._next_id += 1
        write_frame(self._sock, request.SerializeToString())
        response = control_pb2.Response.FromString(read_frame(self._sock))
        if response.request_id != request.request_id:
            raise ConnectionError(
                f"response id {response.request_id} does not match request id {request.request_id}"
            )
        if response.WhichOneof("body") == "error":
            raise McError(response.error.code, response.error.message)
        return response


def _connect_with_retry(host: str, port: int, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=5.0)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)

"""Minimal Source-RCON client (Minecraft's remote console), for debugging and tests."""

from __future__ import annotations

import socket
import struct

_AUTH, _EXEC = 3, 2


class Rcon:
    def __init__(self, host: str = "127.0.0.1", port: int = 25575, password: str = "", timeout: float = 10.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._id = 0
        if self._request(_AUTH, password)[0] == -1:
            raise PermissionError("RCON authentication failed")

    def command(self, cmd: str) -> str:
        return self._request(_EXEC, cmd)[1]

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> Rcon:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _request(self, kind: int, body: str) -> tuple[int, str]:
        self._id += 1
        data = struct.pack("<ii", self._id, kind) + body.encode() + b"\x00\x00"
        self._sock.sendall(struct.pack("<i", len(data)) + data)
        (length,) = struct.unpack("<i", self._recv(4))
        payload = self._recv(length)
        req_id, _ = struct.unpack("<ii", payload[:8])
        return req_id, payload[8:-2].decode(errors="replace")

    def _recv(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("RCON connection closed")
            buf += chunk
        return buf

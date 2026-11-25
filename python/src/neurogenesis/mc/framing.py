"""Length-prefixed framing: 4-byte big-endian length, then the payload (mirrors mod net/FrameCodec)."""

from __future__ import annotations

import socket
import struct

MAX_FRAME_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct(">I")


def write_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed by server")
        got += k
    return bytes(buf)


def read_frame(sock: socket.socket) -> bytes:
    (length,) = _HEADER.unpack(_read_exact(sock, _HEADER.size))
    if length > MAX_FRAME_BYTES:
        raise ConnectionError(f"frame too large: {length} bytes")
    return _read_exact(sock, length)

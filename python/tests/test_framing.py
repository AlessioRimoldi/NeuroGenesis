import socket
import threading

import pytest

from neurogenesis.mc.framing import read_frame, write_frame
from neurogenesis.mc.v1 import control_pb2


def test_roundtrip_over_socketpair():
    a, b = socket.socketpair()
    with a, b:
        payloads = [b"", b"x", bytes(range(256)) * 1000]

        def writer():
            for p in payloads:
                write_frame(a, p)

        # Write from a thread: the large frame exceeds the socket buffer and would block without a reader.
        t = threading.Thread(target=writer)
        t.start()
        got = [read_frame(b) for _ in payloads]
        t.join()
        assert got == payloads


def test_closed_mid_frame_raises():
    a, b = socket.socketpair()
    with b:
        a.sendall(b"\x00\x00\x00\x10abc")
        a.close()
        with pytest.raises(ConnectionError):
            read_frame(b)


def test_protobuf_request_roundtrip():
    a, b = socket.socketpair()
    with a, b:
        req = control_pb2.Request(request_id=7, step=control_pb2.Step(ticks=3))
        t = threading.Thread(target=write_frame, args=(a, req.SerializeToString()))
        t.start()
        got = control_pb2.Request.FromString(read_frame(b))
        t.join()
        assert got.request_id == 7 and got.step.ticks == 3

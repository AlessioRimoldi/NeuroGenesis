package neurogenesis.mc.net;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.proto.v1.Response;

/** One connected Python client. Responses are written from the server thread. */
public final class ControlConnection {
	private final Socket socket;
	private final DataOutputStream out;
	private volatile boolean closed;

	ControlConnection(Socket socket) throws IOException {
		this.socket = socket;
		this.out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream(), 1 << 16));
	}

	public synchronized void send(Response response) {
		if (closed) {
			return;
		}
		try {
			FrameCodec.write(out, response.toByteArray());
		} catch (IOException e) {
			NeuroGenesisMod.LOGGER.warn("Failed to send response to {}: {}", describe(), e.toString());
			close();
		}
	}

	public void close() {
		closed = true;
		try {
			socket.close();
		} catch (IOException ignored) {
		}
	}

	public boolean isClosed() {
		return closed;
	}

	public String describe() {
		return String.valueOf(socket.getRemoteSocketAddress());
	}
}

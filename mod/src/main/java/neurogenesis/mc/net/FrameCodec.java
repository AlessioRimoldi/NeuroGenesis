package neurogenesis.mc.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;

/** Length-prefixed framing: 4-byte big-endian length, then the payload. */
public final class FrameCodec {
	public static final int MAX_FRAME_BYTES = 64 * 1024 * 1024;

	private FrameCodec() {
	}

	/** Reads one frame, or returns null on a clean end of stream between frames. */
	public static byte[] read(DataInputStream in) throws IOException {
		int b0 = in.read();
		if (b0 < 0) {
			return null;
		}
		int length = (b0 << 24) | (in.readUnsignedByte() << 16) | (in.readUnsignedByte() << 8) | in.readUnsignedByte();
		if (length < 0 || length > MAX_FRAME_BYTES) {
			throw new IOException("Invalid frame length " + length);
		}
		byte[] payload = new byte[length];
		try {
			in.readFully(payload);
		} catch (EOFException e) {
			throw new IOException("Connection closed inside a frame", e);
		}
		return payload;
	}

	public static void write(DataOutputStream out, byte[] payload) throws IOException {
		out.writeInt(payload.length);
		out.write(payload);
		out.flush();
	}
}

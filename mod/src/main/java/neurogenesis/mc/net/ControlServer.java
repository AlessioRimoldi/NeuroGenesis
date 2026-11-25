package neurogenesis.mc.net;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;

import neurogenesis.mc.Config;
import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.proto.v1.Request;
import neurogenesis.mc.sim.Lockstep;

/**
 * Accepts one Python client at a time and forwards its requests to {@link Lockstep}, which handles them on the
 * server thread. Further clients wait in the accept backlog until the current one disconnects.
 */
public final class ControlServer implements Runnable {
	private final Config config;
	private final Lockstep lockstep;
	private volatile boolean running = true;
	private volatile ServerSocket serverSocket;
	private volatile ControlConnection current;

	public ControlServer(Config config, Lockstep lockstep) {
		this.config = config;
		this.lockstep = lockstep;
	}

	public void start() {
		Thread thread = new Thread(this, "NeuroGenesis-Control");
		thread.setDaemon(true);
		thread.start();
	}

	@Override
	public void run() {
		try (ServerSocket socket = new ServerSocket()) {
			serverSocket = socket;
			socket.setReuseAddress(true);
			socket.bind(new InetSocketAddress(config.bindAddress(), config.port()));
			NeuroGenesisMod.LOGGER.info("Control server listening on {}:{}", config.bindAddress(), config.port());
			while (running) {
				Socket client;
				try {
					client = socket.accept();
				} catch (IOException e) {
					if (running) {
						NeuroGenesisMod.LOGGER.warn("Control accept failed: {}", e.toString());
					}
					continue;
				}
				serve(client);
			}
		} catch (IOException e) {
			NeuroGenesisMod.LOGGER.error("Control server failed on {}:{}", config.bindAddress(), config.port(), e);
		}
	}

	private void serve(Socket client) {
		ControlConnection connection;
		try {
			client.setTcpNoDelay(true);
			connection = new ControlConnection(client);
		} catch (IOException e) {
			NeuroGenesisMod.LOGGER.warn("Control client setup failed: {}", e.toString());
			return;
		}
		current = connection;
		NeuroGenesisMod.LOGGER.info("Control client connected: {}", connection.describe());
		lockstep.attach(connection);
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(client.getInputStream(), 1 << 16))) {
			byte[] frame;
			while (running && (frame = FrameCodec.read(in)) != null) {
				lockstep.submit(connection, Request.parseFrom(frame));
			}
		} catch (IOException e) {
			if (running && !connection.isClosed()) {
				NeuroGenesisMod.LOGGER.warn("Control client {} dropped: {}", connection.describe(), e.toString());
			}
		} finally {
			connection.close();
			lockstep.detach(connection);
			current = null;
			NeuroGenesisMod.LOGGER.info("Control client disconnected: {}", connection.describe());
		}
	}

	public void stop() {
		running = false;
		ControlConnection connection = current;
		if (connection != null) {
			connection.close();
		}
		ServerSocket socket = serverSocket;
		if (socket != null) {
			try {
				socket.close();
			} catch (IOException ignored) {
			}
		}
	}
}

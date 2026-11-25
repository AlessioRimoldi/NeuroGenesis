package neurogenesis.mc.sim;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.LockSupport;

import net.minecraft.server.MinecraftServer;
import net.minecraft.util.Util;

import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.mixin.MinecraftServerAccessor;
import neurogenesis.mc.net.ControlConnection;
import neurogenesis.mc.proto.v1.Closed;
import neurogenesis.mc.proto.v1.Error;
import neurogenesis.mc.proto.v1.Request;
import neurogenesis.mc.proto.v1.Response;
import neurogenesis.mc.proto.v1.StepResult;

/**
 * Tick barrier between the Python client and the server thread.
 *
 * <p>While a client is attached, the server thread stops at every tick boundary (start of
 * {@code MinecraftServer.waitUntilNextTick}) until a request authorises more ticks. Wall-clock pacing is removed, so
 * the game runs exactly as fast as Python steps it. Without a client the server ticks normally at 20 tps.
 *
 * <p>While waiting, the server keeps running its task queue (see the {@code haveTime} override in
 * {@code MinecraftServerMixin}), so console/RCON commands, chunk work and our own requests are processed.
 */
public final class Lockstep {
	private static final long IDLE_PARK_NANOS = 5_000_000L;

	private final MinecraftServer server;
	private final Simulation simulation;
	private final ConcurrentLinkedQueue<Inbound> inbox = new ConcurrentLinkedQueue<>();

	private volatile boolean active;
	private volatile boolean waiting;
	private volatile double pace; // ticks per second under lockstep; 0 = as fast as Python steps
	private long lastTickTargetNanos;

	// Server-thread state.
	private ControlConnection session;
	private int ticksRemaining;
	private Pending pending;
	private long requestStartNanos;

	public Lockstep(MinecraftServer server, Simulation simulation) {
		this.server = server;
		this.simulation = simulation;
	}

	// ---------------------------------------------------------------- control thread

	public void attach(ControlConnection connection) {
		active = true;
		enqueue(new Inbound(Inbound.Kind.ATTACH, connection, null));
	}

	public void submit(ControlConnection connection, Request request) {
		enqueue(new Inbound(Inbound.Kind.REQUEST, connection, request));
	}

	public void detach(ControlConnection connection) {
		enqueue(new Inbound(Inbound.Kind.DETACH, connection, null));
	}

	private void enqueue(Inbound message) {
		inbox.add(message);
		server.execute(() -> {
		}); // wakes the server thread
	}

	// ---------------------------------------------------------------- server thread

	public boolean isWaiting() {
		return waiting;
	}

	public boolean isActive() {
		return active;
	}

	public MinecraftServer server() {
		return server;
	}

	/** Limits lockstep to {@code ticksPerSecond} (0 = unlimited), e.g. 20 to watch agents in real time. */
	public void setPace(double ticksPerSecond) {
		pace = Math.max(0.0, ticksPerSecond);
	}

	public double pace() {
		return pace;
	}

	public boolean hasSession() {
		return active;
	}

	/** Park while idle at the barrier; wake-ups come from {@code server.execute} unparking the thread. */
	public void parkWhileWaiting() {
		LockSupport.parkNanos(this, IDLE_PARK_NANOS);
	}

	/** Called at the start of {@code waitUntilNextTick}, i.e. right after a tick finished. */
	public void onTickBoundary() {
		if (!active && session == null && inbox.isEmpty()) {
			return;
		}
		if (session != null) {
			simulation.afterTick();
		}
		if (ticksRemaining > 0) {
			ticksRemaining--;
			if (ticksRemaining > 0) {
				runNextTickImmediately();
				return;
			}
			completePending();
		}

		waiting = true;
		try {
			server.managedBlock(() -> {
				keepWatchdogQuiet();
				drainInbox();
				return ticksRemaining > 0 || session == null || !server.isRunning();
			});
		} finally {
			waiting = false;
		}
		runNextTickImmediately();
	}

	private void drainInbox() {
		Inbound message;
		while ((message = inbox.poll()) != null) {
			try {
				switch (message.kind()) {
					case ATTACH -> onAttach(message.connection());
					case DETACH -> onDetach(message.connection());
					case REQUEST -> {
						if (message.connection() == session) {
							handle(message.request());
						}
					}
				}
			} catch (RuntimeException e) {
				NeuroGenesisMod.LOGGER.error("Lockstep failed handling {}", message.kind(), e);
				if (message.request() != null && message.connection() == session) {
					pending = null;
					ticksRemaining = 0;
					sendError(message.request().getRequestId(), Error.Code.INTERNAL, e.toString());
				}
			}
		}
	}

	private void onAttach(ControlConnection connection) {
		if (session != null) {
			onDetach(session);
		}
		session = connection;
		active = true;
		ticksRemaining = 0;
		pending = null;
	}

	private void onDetach(ControlConnection connection) {
		if (connection != session) {
			return;
		}
		simulation.detach();
		session = null;
		active = !inbox.isEmpty() && inbox.stream().anyMatch(m -> m.kind() == Inbound.Kind.ATTACH);
		ticksRemaining = 0;
		pending = null;
	}

	private void handle(Request request) {
		requestStartNanos = Util.getNanos();
		long id = request.getRequestId();
		if (pending != null) {
			sendError(id, Error.Code.BAD_REQUEST, "previous request still in progress");
			return;
		}
		switch (request.getBodyCase()) {
			case HELLO -> {
				int theirs = request.getHello().getProtocolVersion();
				if (theirs != NeuroGenesisMod.PROTOCOL_VERSION) {
					sendError(id, Error.Code.PROTOCOL_MISMATCH,
						"client speaks protocol " + theirs + ", server speaks " + NeuroGenesisMod.PROTOCOL_VERSION);
					return;
				}
				send(Response.newBuilder().setRequestId(id).setSpec(simulation.spec()));
			}
			case RESET -> {
				try {
					simulation.reset(request.getReset());
				} catch (IllegalArgumentException e) {
					sendError(id, Error.Code.BAD_REQUEST, e.getMessage());
					return;
				}
				startTicks(id, request.getReset().getSettleTicks());
			}
			case STEP -> {
				if (!simulation.ready()) {
					sendError(id, Error.Code.NOT_READY, "call reset before step");
					return;
				}
				simulation.applyActions(request.getStep());
				startTicks(id, Math.max(1, request.getStep().getTicks()));
			}
			case CLOSE -> {
				send(Response.newBuilder().setRequestId(id).setClosed(Closed.getDefaultInstance()));
				onDetach(session);
			}
			default -> sendError(id, Error.Code.BAD_REQUEST, "empty request");
		}
	}

	private void startTicks(long requestId, int ticks) {
		pending = new Pending(requestId);
		ticksRemaining = ticks;
		if (ticks == 0) {
			completePending();
		}
	}

	private void completePending() {
		Pending done = pending;
		pending = null;
		ticksRemaining = 0;
		if (done == null || session == null) {
			return;
		}
		StepResult.Builder result = simulation.observe()
			.setTick(server.getTickCount())
			.setServerStepUs((Util.getNanos() - requestStartNanos) / 1_000L);
		send(Response.newBuilder().setRequestId(done.requestId()).setStepResult(result));
	}

	private void send(Response.Builder response) {
		if (session != null) {
			session.send(response.build());
		}
	}

	private void sendError(long requestId, Error.Code code, String message) {
		send(Response.newBuilder().setRequestId(requestId)
			.setError(Error.newBuilder().setCode(code).setMessage(message == null ? "" : message)));
	}

	/**
	 * Lets vanilla start the next tick without waiting for the 50 ms tick slot, or, with a pace set, no earlier than
	 * one pace interval after the previous tick (vanilla then waits, still serving tasks). No catch-up bursts.
	 */
	private void runNextTickImmediately() {
		long now = Util.getNanos();
		long next = now;
		double tps = pace;
		if (tps > 0) {
			next = Math.max(now, lastTickTargetNanos + (long) (1_000_000_000L / tps));
		}
		lastTickTargetNanos = next;
		MinecraftServerAccessor accessor = (MinecraftServerAccessor) server;
		accessor.ng$setNextTickTimeNanos(next);
		accessor.ng$setDelayedTasksMaxNextTickTimeNanos(next);
	}

	/** The watchdog compares wall time with the next-tick deadline; keep the deadline current while paused. */
	private void keepWatchdogQuiet() {
		((MinecraftServerAccessor) server).ng$setNextTickTimeNanos(Util.getNanos());
	}

	private record Pending(long requestId) {
	}

	private record Inbound(Kind kind, ControlConnection connection, Request request) {
		enum Kind { ATTACH, DETACH, REQUEST }
	}
}

package neurogenesis.mc.world;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;

import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.agent.AgentPlayer;
import neurogenesis.mc.sim.EpisodeManager;
import neurogenesis.mc.sim.Lockstep;

/**
 * Debug commands (op level 2):
 * <ul>
 *   <li>{@code /ng status} — session, episode, agents and pace</li>
 *   <li>{@code /ng pace} / {@code /ng pace <ticksPerSecond>} — limit lockstep speed (20 = real time, 0 = unlimited)</li>
 * </ul>
 */
public final class NgCommand {
	private NgCommand() {
	}

	public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
		dispatcher.register(Commands.literal("ng")
			.requires(Commands.hasPermission(Commands.LEVEL_GAMEMASTERS))
			.then(Commands.literal("status").executes(NgCommand::status))
			.then(Commands.literal("pace")
				.executes(NgCommand::queryPace)
				.then(Commands.argument("ticksPerSecond", DoubleArgumentType.doubleArg(0.0, 10_000.0))
					.executes(c -> setPace(c, DoubleArgumentType.getDouble(c, "ticksPerSecond"))))));
	}

	private static int status(CommandContext<CommandSourceStack> c) {
		CommandSourceStack source = c.getSource();
		Lockstep lockstep = NeuroGenesisMod.lockstepFor(source.getServer());
		EpisodeManager episodes = NeuroGenesisMod.episodesFor(source.getServer());
		if (lockstep == null || episodes == null) {
			source.sendFailure(Component.literal("NeuroGenesis control is disabled on this server"));
			return 0;
		}
		StringBuilder agents = new StringBuilder();
		for (AgentPlayer agent : episodes.agents().agents()) {
			if (!agents.isEmpty()) {
				agents.append(", ");
			}
			agents.append(agent.getGameProfile().name()).append(agent.isTerminated() ? " (dead)" : "");
		}
		String text = String.format("Session: %s | episode: %s | episode tick: %d | pace: %s | agents (%d): %s",
			lockstep.hasSession() ? "connected" : "none",
			episodes.isReady() ? "running" : "not started",
			episodes.episodeTick(),
			paceText(lockstep.pace()),
			episodes.agents().agents().size(),
			agents.isEmpty() ? "-" : agents);
		source.sendSuccess(() -> Component.literal(text), false);
		return 1;
	}

	private static int queryPace(CommandContext<CommandSourceStack> c) {
		Lockstep lockstep = NeuroGenesisMod.lockstepFor(c.getSource().getServer());
		if (lockstep == null) {
			c.getSource().sendFailure(Component.literal("NeuroGenesis control is disabled on this server"));
			return 0;
		}
		c.getSource().sendSuccess(() -> Component.literal("Lockstep pace: " + paceText(lockstep.pace())), false);
		return 1;
	}

	private static int setPace(CommandContext<CommandSourceStack> c, double tps) {
		Lockstep lockstep = NeuroGenesisMod.lockstepFor(c.getSource().getServer());
		if (lockstep == null) {
			c.getSource().sendFailure(Component.literal("NeuroGenesis control is disabled on this server"));
			return 0;
		}
		lockstep.setPace(tps);
		c.getSource().sendSuccess(() -> Component.literal("Lockstep pace set to " + paceText(tps)), true);
		return 1;
	}

	private static String paceText(double tps) {
		return tps <= 0 ? "unlimited" : String.format("%.4g ticks/s", tps);
	}
}

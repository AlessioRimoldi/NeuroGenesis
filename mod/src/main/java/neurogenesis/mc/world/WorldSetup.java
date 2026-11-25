package neurogenesis.mc.world;

import net.minecraft.server.MinecraftServer;

import neurogenesis.mc.NeuroGenesisMod;

/** One-time world configuration for RL episodes: a static, mob-free, quiet world. */
public final class WorldSetup {
	private static final String[] COMMANDS = {
		"gamerule advance_time false",
		"gamerule advance_weather false",
		"gamerule spawn_mobs false",
		"gamerule spawn_monsters false",
		"gamerule spawn_patrols false",
		"gamerule spawn_phantoms false",
		"gamerule spawn_wandering_traders false",
		"gamerule spawn_wardens false",
		"gamerule mob_griefing false",
		"gamerule fire_spread_radius_around_player 0",
		"gamerule show_death_messages false",
		"gamerule show_advancement_messages false",
		"gamerule log_admin_commands false",
		"gamerule immediate_respawn true",
		"time set noon",
		"weather clear",
		"difficulty normal",
	};

	private boolean done;

	public void ensure(MinecraftServer server) {
		if (done) {
			return;
		}
		var source = server.createCommandSourceStack().withSuppressedOutput();
		for (String command : COMMANDS) {
			server.getCommands().performPrefixedCommand(source, command);
		}
		NeuroGenesisMod.LOGGER.info("World configured for lockstep episodes");
		done = true;
	}
}

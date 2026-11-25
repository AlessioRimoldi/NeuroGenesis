package neurogenesis.mc;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.server.MinecraftServer;
import org.jspecify.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neurogenesis.mc.net.ControlServer;
import neurogenesis.mc.sim.EpisodeManager;
import neurogenesis.mc.sim.Lockstep;

/** Common entrypoint: runs on the dedicated server and (in M2) inside a client's integrated server. */
public class NeuroGenesisMod implements ModInitializer {
	public static final String MOD_ID = "neurogenesis";
	public static final int PROTOCOL_VERSION = 1;
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	private static volatile @Nullable Lockstep lockstep;
	private static volatile @Nullable ControlServer controlServer;

	@Override
	public void onInitialize() {
		Config config = Config.load();
		LOGGER.info("NeuroGenesis initialising (control {}, port {})", config.controlEnabled() ? "on" : "off", config.port());

		ServerLifecycleEvents.SERVER_STARTED.register(server -> {
			if (!config.controlEnabled()) {
				return;
			}
			Lockstep ls = new Lockstep(server, new EpisodeManager(server));
			lockstep = ls;
			ControlServer cs = new ControlServer(config, ls);
			controlServer = cs;
			cs.start();
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			ControlServer cs = controlServer;
			if (cs != null) {
				cs.stop();
			}
			controlServer = null;
			lockstep = null;
		});
	}

	/** The lockstep controller for this server, or null if control is disabled or not started yet. */
	public static @Nullable Lockstep lockstepFor(MinecraftServer server) {
		Lockstep ls = lockstep;
		return ls != null && ls.server() == server ? ls : null;
	}
}

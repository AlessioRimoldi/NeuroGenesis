package neurogenesis.mc.sim;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.SharedConstants;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.GameType;
import net.minecraft.world.phys.Vec3;
import org.jspecify.annotations.Nullable;

import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.agent.AgentManager;
import neurogenesis.mc.agent.AgentPlayer;
import neurogenesis.mc.arena.ArenaBuilder;
import neurogenesis.mc.arena.ArenaSpec;
import neurogenesis.mc.arena.FoodSpawner;
import neurogenesis.mc.crafting.RecipeIndex;
import neurogenesis.mc.obs.ObservationBuilder;
import neurogenesis.mc.proto.v1.AgentAction;
import neurogenesis.mc.proto.v1.AgentConfig;
import neurogenesis.mc.proto.v1.ObsConfig;
import neurogenesis.mc.proto.v1.Reset;
import neurogenesis.mc.proto.v1.SmellConfig;
import neurogenesis.mc.proto.v1.Spec;
import neurogenesis.mc.proto.v1.Step;
import neurogenesis.mc.proto.v1.StepResult;
import neurogenesis.mc.sense.SmellField;
import neurogenesis.mc.sense.SmellSense;
import neurogenesis.mc.world.WorldSetup;

/** Owns the episode: arena, agents, food, senses and observations. All calls happen on the server thread. */
public final class EpisodeManager implements Simulation {
	public static final int MAX_AGENTS = 64;

	private final MinecraftServer server;
	private final WorldSetup worldSetup = new WorldSetup();
	private final ArenaBuilder arenaBuilder = new ArenaBuilder();
	private final AgentManager agents;
	private final FoodSpawner food = new FoodSpawner();

	private @Nullable ArenaSpec spec;
	private @Nullable SmellSense smell;
	private SplittableRandom rng = new SplittableRandom(0);
	private @Nullable RecipeIndex recipeIndex;
	private ObsConfig obsConfig = ObsConfig.getDefaultInstance();
	private boolean ready;
	private int episodeTick;
	private int maxEpisodeTicks;

	public EpisodeManager(MinecraftServer server) {
		this.server = server;
		this.agents = new AgentManager(server);
	}

	public AgentManager agents() {
		return agents;
	}

	public boolean isReady() {
		return ready;
	}

	public int episodeTick() {
		return episodeTick;
	}

	public RecipeIndex recipes() {
		if (recipeIndex == null) {
			recipeIndex = RecipeIndex.build(server);
		}
		return recipeIndex;
	}

	/** Recipes can change with datapack reloads (/reload). */
	public void invalidateRecipes() {
		recipeIndex = null;
		for (AgentPlayer agent : agents.agents()) {
			agent.craftable().invalidate();
		}
	}

	@Override
	public Spec spec() {
		String modVersion = FabricLoader.getInstance().getModContainer(NeuroGenesisMod.MOD_ID)
			.map(c -> c.getMetadata().getVersion().getFriendlyString()).orElse("unknown");
		return Spec.newBuilder()
			.setProtocolVersion(NeuroGenesisMod.PROTOCOL_VERSION)
			.setMinecraftVersion(SharedConstants.getCurrentVersion().name())
			.setModVersion(modVersion)
			.setServerTick(server.getTickCount())
			.setMaxAgents(MAX_AGENTS)
			.addAllItemVocab(Vocab.items())
			.addAllBlockVocab(Vocab.blocks())
			.addAllEntityVocab(Vocab.entities())
			.addAllMenuVocab(Vocab.menus())
			.addAllRecipeVocab(recipes().entries().stream().map(RecipeIndex.Entry::proto).toList())
			.build();
	}

	@Override
	public void reset(Reset reset) {
		if (reset.getAgentsCount() > MAX_AGENTS) {
			throw new IllegalArgumentException("at most " + MAX_AGENTS + " agents are supported");
		}
		ServerLevel level = server.overworld();
		worldSetup.ensure(server);
		ArenaSpec newSpec = ArenaSpec.from(reset.getArena());
		rng = new SplittableRandom(reset.getSeed());
		level.getRandom().setSeed(reset.getSeed());

		arenaBuilder.build(level, newSpec);
		spec = newSpec;

		List<AgentConfig> configs = reset.getAgentsList();
		List<Vec3> spawns = new ArrayList<>();
		List<Float> yaws = new ArrayList<>();
		for (AgentConfig config : configs) {
			spawns.add(config.hasSpawnPos()
				? new Vec3(config.getSpawnPos().getX(), config.getSpawnPos().getY(), config.getSpawnPos().getZ())
				: freeSpot(spawns));
			yaws.add(config.hasSpawnYaw() ? config.getSpawnYaw() : (float) (rng.nextDouble() * 360.0 - 180.0));
		}
		agents.reset(level, configs, spawns, yaws);
		food.reset(level, newSpec, rng, spawns);
		smell = buildSmell(newSpec);

		obsConfig = reset.getObs();
		episodeTick = 0;
		maxEpisodeTicks = reset.getMaxEpisodeTicks();
		ready = true;
	}

	@Override
	public boolean ready() {
		return ready;
	}

	@Override
	public void applyActions(Step step) {
		Set<Integer> mentioned = new HashSet<>();
		for (AgentAction action : step.getActionsList()) {
			AgentPlayer agent = agents.get(action.getAgentId());
			mentioned.add(action.getAgentId());
			if (!agent.isTerminated()) {
				agent.input().set(action);
			}
		}
		for (AgentPlayer agent : agents.agents()) {
			if (!mentioned.contains(agent.agentId())) {
				agent.input().releaseAll();
			}
		}
	}

	@Override
	public void afterTick() {
		if (!ready) {
			return;
		}
		episodeTick++;
		ServerLevel level = server.overworld();
		for (AgentPlayer agent : agents.agents()) {
			if (agent.consumeDeath()) {
				if (agent.respawnOnDeath()) {
					agents.respawn(agent, freeSpot(List.of()), (float) (rng.nextDouble() * 360.0 - 180.0));
				} else {
					agent.markTerminated();
					agent.gameMode.changeGameModeForPlayer(GameType.SPECTATOR);
				}
			}
		}
		food.tick(level);
		if (smell != null) {
			smell.field().step(food.sources());
		}
	}

	@Override
	public StepResult.Builder observe() {
		boolean truncated = maxEpisodeTicks > 0 && episodeTick >= maxEpisodeTicks;
		StepResult.Builder result = StepResult.newBuilder().setEpisodeTick(episodeTick);
		RecipeIndex index = obsConfig.getCraftable() ? recipes() : null;
		for (AgentPlayer agent : agents.agents()) {
			result.addObs(ObservationBuilder.build(agent, smell, index, truncated));
		}
		return result;
	}

	@Override
	public void detach() {
		agents.removeAll();
		ready = false;
		episodeTick = 0;
	}

	private @Nullable SmellSense buildSmell(ArenaSpec s) {
		SmellConfig c = s.smell();
		if (!c.getEnabled()) {
			return null;
		}
		SmellField field = new SmellField(ArenaSpec.ORIGIN.getX(), ArenaSpec.ORIGIN.getZ(), s.sizeX(), s.sizeZ(),
			c.getCellSize() > 0 ? c.getCellSize() : 1.0,
			c.getDiffusion() > 0 ? c.getDiffusion() : 0.2,
			c.getDecay() > 0 ? c.getDecay() : 0.01,
			c.getEmission() > 0 ? c.getEmission() : 1.0);
		int prewarm = c.getPrewarmTicks() > 0 ? (int) c.getPrewarmTicks() : 400;
		List<Vec3> sources = food.sources();
		for (int i = 0; i < prewarm; i++) {
			field.step(sources);
		}
		return new SmellSense(field,
			c.getAntennaLateral() > 0 ? c.getAntennaLateral() : 0.3,
			c.getAntennaForward() > 0 ? c.getAntennaForward() : 0.3);
	}

	/** A random free floor cell centre, at least 1.5 blocks from {@code others}. */
	private Vec3 freeSpot(List<Vec3> others) {
		ArenaSpec s = spec;
		for (int attempt = 0; attempt < 500; attempt++) {
			double x = ArenaSpec.ORIGIN.getX() + rng.nextInt(s.sizeX()) + 0.5;
			double z = ArenaSpec.ORIGIN.getZ() + rng.nextInt(s.sizeZ()) + 0.5;
			Vec3 p = new Vec3(x, ArenaSpec.ORIGIN.getY(), z);
			if (others.stream().noneMatch(o -> o.distanceToSqr(p) < 2.25)) {
				return p;
			}
		}
		return new Vec3(ArenaSpec.ORIGIN.getX() + s.sizeX() / 2.0, ArenaSpec.ORIGIN.getY(), ArenaSpec.ORIGIN.getZ() + s.sizeZ() / 2.0);
	}
}

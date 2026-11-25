package neurogenesis.mc.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import com.mojang.authlib.GameProfile;
import net.minecraft.core.UUIDUtil;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.DisconnectionDetails;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.GameType;
import net.minecraft.world.phys.Vec3;

import neurogenesis.mc.proto.v1.AgentConfig;
import neurogenesis.mc.proto.v1.GameMode;

/** Spawns, resets and removes agents. Agents keep their entity across resets when the name is unchanged. */
public final class AgentManager {
	private static final Pattern NAME = Pattern.compile("[A-Za-z0-9_]{1,16}");

	private final MinecraftServer server;
	private final List<AgentPlayer> agents = new ArrayList<>();

	public AgentManager(MinecraftServer server) {
		this.server = server;
	}

	public List<AgentPlayer> agents() {
		return Collections.unmodifiableList(agents);
	}

	public AgentPlayer get(int id) {
		if (id < 0 || id >= agents.size()) {
			throw new IllegalArgumentException("unknown agent_id " + id);
		}
		return agents.get(id);
	}

	/** Makes the agent list match {@code configs} (by index), placing each agent at {@code spawns[i]}. */
	public void reset(ServerLevel level, List<AgentConfig> configs, List<Vec3> spawns, List<Float> yaws) {
		for (AgentConfig config : configs) {
			if (!NAME.matcher(config.getName()).matches()) {
				throw new IllegalArgumentException("invalid agent name '" + config.getName() + "' (1-16 chars of [A-Za-z0-9_])");
			}
		}
		if (configs.stream().map(AgentConfig::getName).distinct().count() != configs.size()) {
			throw new IllegalArgumentException("agent names must be unique");
		}
		// Remove agents whose slot changed name or disappeared.
		for (int i = agents.size() - 1; i >= 0; i--) {
			if (i >= configs.size() || !agents.get(i).getGameProfile().name().equals(configs.get(i).getName())) {
				remove(agents.remove(i));
			}
		}
		for (int i = 0; i < configs.size(); i++) {
			AgentConfig config = configs.get(i);
			AgentPlayer agent = i < agents.size() ? agents.get(i) : spawn(level, config.getName(), i, spawns.get(i), yaws.get(i));
			if (i >= agents.size()) {
				agents.add(agent);
			}
			resetAgent(agent, level, config, spawns.get(i), yaws.get(i));
		}
	}

	public void removeAll() {
		for (AgentPlayer agent : agents) {
			remove(agent);
		}
		agents.clear();
	}

	/** Puts a dead agent back at a spawn point with full health and food (respawn_on_death). */
	public void respawn(AgentPlayer agent, Vec3 pos, float yaw) {
		agent.removeAllEffects();
		agent.clearFire();
		agent.setAirSupply(agent.getMaxAirSupply());
		agent.setHealth(agent.getMaxHealth());
		agent.clearEpisodeStateKeepEvents();
		teleport(agent, agent.level(), pos, yaw);
	}

	private AgentPlayer spawn(ServerLevel level, String name, int id, Vec3 pos, float yaw) {
		GameProfile profile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(name), name);
		AgentPlayer agent = new AgentPlayer(server, level, profile, ClientInformation.createDefault(), id);
		agent.snapTo(pos.x, pos.y, pos.z, yaw, 0.0F);
		server.getPlayerList().placeNewPlayer(new AgentConnection(), agent,
			new CommonListenerCookie(profile, 0, agent.clientInformation(), false));
		((AgentListenerHooks) agent.connection).ng$markClientLoaded();
		return agent;
	}

	private void resetAgent(AgentPlayer agent, ServerLevel level, AgentConfig config, Vec3 pos, float yaw) {
		agent.stopUsingItem();
		agent.closeContainer();
		agent.stopRiding();
		agent.clearEpisodeState();
		agent.setRespawnOnDeath(config.getRespawnOnDeath());
		agent.gameMode.changeGameModeForPlayer(gameType(config.getGameMode()));
		agent.getAbilities().flying = false;
		agent.onUpdateAbilities();
		agent.removeAllEffects();
		agent.clearFire();
		agent.setTicksFrozen(0);
		agent.setAirSupply(agent.getMaxAirSupply());
		agent.setSprinting(false);
		agent.setShiftKeyDown(false);
		agent.setHealth(config.hasHealth() ? Math.max(0.5F, config.getHealth()) : agent.getMaxHealth());
		if (config.hasFood()) {
			agent.getFoodData().setFoodLevel(config.getFood());
		}
		agent.setExperienceLevels(0);
		agent.setExperiencePoints(0);
		agent.resetAttackStrengthTicker();

		Inventory inventory = agent.getInventory();
		inventory.clearContent();
		inventory.setSelectedSlot(0);
		for (var stack : config.getInventoryList()) {
			Item item = BuiltInRegistries.ITEM.getOptional(Identifier.parse(stack.getId()))
				.orElseThrow(() -> new IllegalArgumentException("unknown item " + stack.getId()));
			ItemStack itemStack = new ItemStack(item, Math.max(1, stack.getCount()));
			if (stack.getSlot() >= 0 && stack.getSlot() < inventory.getContainerSize()) {
				inventory.setItem(stack.getSlot(), itemStack);
			} else {
				inventory.add(itemStack);
			}
		}
		agent.inventoryMenu.broadcastChanges();
		teleport(agent, level, pos, yaw);
	}

	private static void teleport(AgentPlayer agent, ServerLevel level, Vec3 pos, float yaw) {
		agent.setDeltaMovement(Vec3.ZERO);
		agent.resetFallDistance();
		agent.teleportTo(level, pos.x, pos.y, pos.z, Set.of(), yaw, 0.0F, true);
		agent.setYHeadRot(yaw);
		agent.setYBodyRot(yaw);
	}

	private static void remove(AgentPlayer agent) {
		agent.connection.onDisconnect(new DisconnectionDetails(Component.literal("NeuroGenesis episode ended")));
	}

	private static GameType gameType(GameMode mode) {
		return switch (mode) {
			case GAME_MODE_ADVENTURE -> GameType.ADVENTURE;
			case GAME_MODE_CREATIVE -> GameType.CREATIVE;
			case GAME_MODE_SPECTATOR -> GameType.SPECTATOR;
			default -> GameType.SURVIVAL;
		};
	}
}

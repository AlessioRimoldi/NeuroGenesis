package neurogenesis.mc.agent;

import java.util.ArrayList;
import java.util.List;

import com.mojang.authlib.GameProfile;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.protocol.game.ServerboundPlayerCommandPacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.stats.Stat;
import net.minecraft.stats.Stats;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.Pose;
import net.minecraft.world.entity.player.Abilities;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;

import neurogenesis.mc.input.AgentInput;
import neurogenesis.mc.input.ClientEmulator;
import neurogenesis.mc.proto.v1.Event;

/**
 * A server-side player driven by the lockstep client instead of a network client.
 *
 * <p>Unlike a real player, whose movement is simulated by its client, an agent is fully simulated on the server
 * ({@link #isClientAuthoritative()} is false), using the same vanilla physics. The client-side input rules that a
 * real client would apply before sending its movement (LocalPlayer.aiStep: sprint rules, sneaking and item-use
 * slowdown, pushing out of blocks) are reproduced here. Structure adapted from Carpet's EntityPlayerMPFake (MIT).
 */
public final class AgentPlayer extends ServerPlayer {
	private final int agentId;
	private final AgentInput input = new AgentInput();
	private final ClientEmulator client = new ClientEmulator(this);
	private final List<Event> events = new ArrayList<>();

	private boolean respawnOnDeath;
	private boolean terminated;
	private boolean diedThisTick;
	private Vec2 moveVector = Vec2.ZERO;
	private boolean crouchingInput;
	private boolean prevJumpKey;

	public AgentPlayer(MinecraftServer server, ServerLevel level, GameProfile profile, ClientInformation info, int agentId) {
		super(server, level, profile, info);
		this.agentId = agentId;
	}

	public int agentId() {
		return agentId;
	}

	public AgentInput input() {
		return input;
	}

	public ClientEmulator client() {
		return client;
	}

	public boolean isTerminated() {
		return terminated;
	}

	public void setRespawnOnDeath(boolean respawnOnDeath) {
		this.respawnOnDeath = respawnOnDeath;
	}

	/** True once after the agent died; the episode manager respawns or retires it. */
	public boolean consumeDeath() {
		boolean died = diedThisTick;
		diedThisTick = false;
		return died;
	}

	public boolean respawnOnDeath() {
		return respawnOnDeath;
	}

	public void markTerminated() {
		terminated = true;
		input.releaseAll();
	}

	public void clearEpisodeState() {
		terminated = false;
		diedThisTick = false;
		events.clear();
		input.releaseAll();
		client.reset();
		moveVector = Vec2.ZERO;
		prevJumpKey = false;
		dead = false;
		deathTime = 0;
		foodData = new net.minecraft.world.food.FoodData();
	}

	/** Like {@link #clearEpisodeState()} but keeps pending events (e.g. the DEATH that caused the respawn). */
	public void clearEpisodeStateKeepEvents() {
		List<Event> pending = new ArrayList<>(events);
		clearEpisodeState();
		events.addAll(pending);
	}

	public List<Event> drainEvents() {
		List<Event> out = new ArrayList<>(events);
		events.clear();
		return out;
	}

	// ------------------------------------------------------------------ ticking

	@Override
	public void tick() {
		if (!terminated) {
			client.tick(input);
		}
		super.tick();   // server-side bookkeeping (what ServerLevel ticks for every player)
		doTick();       // movement and living-entity update (for real players: triggered by their move packets)
		level().getChunkSource().move(this);
	}

	/** LocalPlayer.aiStep: the client-side rules applied before the vanilla living-entity step. */
	@Override
	public void aiStep() {
		boolean keys = !terminated && !client.isScreenOpen() && isAlive() && !isSpectator();
		boolean jumpKey = keys && input.jump;
		boolean shiftKey = keys && input.sneak;
		boolean sprintKey = keys && input.sprint;
		Abilities abilities = getAbilities();

		crouchingInput = !abilities.flying
			&& !isSwimming()
			&& !isPassenger()
			&& canPlayerFitWithinBlocksAndEntitiesWhen(Pose.CROUCHING)
			&& (isShiftKeyDown() || !isSleeping() && !canPlayerFitWithinBlocksAndEntitiesWhen(Pose.STANDING));
		moveVector = client.moveVector(input, crouchingInput || isVisuallyCrawling());
		boolean hasForwardImpulse = moveVector.y > 1.0E-5F;

		if (!noPhysics && !isPassenger()) {
			double w = getBbWidth() * 0.35;
			moveTowardsClosestSpace(getX() - w, getZ() + w);
			moveTowardsClosestSpace(getX() - w, getZ() - w);
			moveTowardsClosestSpace(getX() + w, getZ() - w);
			moveTowardsClosestSpace(getX() + w, getZ() + w);
		}

		if (sprintKey && canStartSprinting(hasForwardImpulse)) {
			sendSprinting(true);
		}
		if (isSprinting()) {
			boolean stop = isSwimming()
				? !isSprintingPossible(true) || !isInWater() || !hasForwardImpulse && !onGround() && !shiftKey
				: !isSprintingPossible(abilities.flying) || !hasForwardImpulse || horizontalCollision && !minorHorizontalCollision;
			if (stop) {
				sendSprinting(false);
			}
		}

		if (jumpKey && !prevJumpKey && !onClimbable() && tryToStartFallFlying()) {
			connection.handlePlayerCommand(new ServerboundPlayerCommandPacket(this, ServerboundPlayerCommandPacket.Action.START_FALL_FLYING));
		}
		if (isInWater() && shiftKey && isAffectedByFluids()) {
			goDownInWater();
		}
		if (abilities.flying) {
			int vertical = (shiftKey ? -1 : 0) + (jumpKey ? 1 : 0);
			if (vertical != 0) {
				setDeltaMovement(getDeltaMovement().add(0.0, vertical * abilities.getFlyingSpeed() * 3.0F, 0.0));
			}
		}
		prevJumpKey = jumpKey;

		super.aiStep();
		if (onGround() && abilities.flying && !isSpectator()) {
			abilities.flying = false;
			onUpdateAbilities();
		}
	}

	/** LocalPlayer.applyInput: movement impulses from the (modified) key vector. */
	@Override
	protected void applyInput() {
		boolean keys = !terminated && !client.isScreenOpen() && isAlive() && !isSpectator();
		xxa = moveVector.x;
		zza = moveVector.y;
		jumping = keys && input.jump;
	}

	private boolean canStartSprinting(boolean hasForwardImpulse) {
		return !isSprinting()
			&& hasForwardImpulse
			&& isSprintingPossible(getAbilities().flying)
			&& !isSlowDueToUsingItem()
			&& (!isFallFlying() || isUnderWater())
			&& (!(crouchingInput || isVisuallyCrawling()) || isUnderWater());
	}

	private boolean isSprintingPossible(boolean allowedInShallowWater) {
		return !isMobilityRestricted()
			&& (isPassenger() ? getVehicle().canSprint() : hasEnoughFoodToDoExhaustiveManoeuvres())
			&& (allowedInShallowWater || !isInShallowWater());
	}

	private boolean isSlowDueToUsingItem() {
		return isUsingItem() && !getUseItem().getOrDefault(DataComponents.USE_EFFECTS,
			net.minecraft.world.item.component.UseEffects.DEFAULT).canSprint();
	}

	private void sendSprinting(boolean sprinting) {
		connection.handlePlayerCommand(new ServerboundPlayerCommandPacket(this, sprinting
			? ServerboundPlayerCommandPacket.Action.START_SPRINTING
			: ServerboundPlayerCommandPacket.Action.STOP_SPRINTING));
	}

	/** LocalPlayer.moveTowardsClosestSpace: nudges the player out of a block it is stuck in. */
	private void moveTowardsClosestSpace(double x, double z) {
		BlockPos pos = BlockPos.containing(x, getY(), z);
		if (!suffocatesAt(pos)) {
			return;
		}
		double xd = x - pos.getX();
		double zd = z - pos.getZ();
		Direction dir = null;
		double closest = Double.MAX_VALUE;
		for (Direction direction : new Direction[]{Direction.WEST, Direction.EAST, Direction.NORTH, Direction.SOUTH}) {
			double axisDistance = direction.getAxis().choose(xd, 0.0, zd);
			double distanceToEdge = direction.getAxisDirection() == Direction.AxisDirection.POSITIVE ? 1.0 - axisDistance : axisDistance;
			if (distanceToEdge < closest && !suffocatesAt(pos.relative(direction))) {
				closest = distanceToEdge;
				dir = direction;
			}
		}
		if (dir != null) {
			Vec3 old = getDeltaMovement();
			if (dir.getAxis() == Direction.Axis.X) {
				setDeltaMovement(0.1 * dir.getStepX(), old.y, old.z);
			} else {
				setDeltaMovement(old.x, old.y, 0.1 * dir.getStepZ());
			}
		}
	}

	private boolean suffocatesAt(BlockPos pos) {
		AABB box = getBoundingBox();
		AABB test = new AABB(pos.getX(), box.minY, pos.getZ(), pos.getX() + 1.0, box.maxY, pos.getZ() + 1.0).deflate(1.0E-7);
		return level().collidesWithSuffocatingBlock(this, test);
	}

	// ------------------------------------------------------------------ server-authoritative movement

	@Override
	public boolean isClientAuthoritative() {
		return false;
	}

	@Override
	public Vec3 getKnownMovement() {
		Entity vehicle = getVehicle();
		return vehicle != null && vehicle.getControllingPassenger() != this ? vehicle.getKnownMovement() : getDeltaMovement();
	}

	@Override
	public Vec3 getKnownSpeed() {
		Entity vehicle = getVehicle();
		return vehicle != null && vehicle.getControllingPassenger() != this ? vehicle.getKnownSpeed() : getDeltaMovement();
	}

	// ------------------------------------------------------------------ death and events

	@Override
	public void die(DamageSource source) {
		super.die(source);
		// Stay in the world: the episode manager respawns or retires the agent on its next tick.
		((AgentListenerHooks) connection).ng$markClientLoaded();
		dead = false;
		deathTime = 0;
		setHealth(getMaxHealth());
		diedThisTick = true;
		events.add(Event.newBuilder().setType(Event.Type.DEATH).build());
	}

	@Override
	public void awardStat(Stat<?> stat, int count) {
		super.awardStat(stat, count);
		Object value = stat.getValue();
		if (stat.getType() == Stats.ITEM_PICKED_UP && value instanceof Item item) {
			event(Event.Type.PICKED_UP, BuiltInRegistries.ITEM.getId(item), count);
		} else if (stat.getType() == Stats.ITEM_CRAFTED && value instanceof Item item) {
			event(Event.Type.CRAFTED, BuiltInRegistries.ITEM.getId(item), count);
		} else if (stat.getType() == Stats.BLOCK_MINED && value instanceof Block block) {
			event(Event.Type.BLOCK_BROKEN, BuiltInRegistries.BLOCK.getId(block), count);
		} else if (stat.getType() == Stats.ITEM_USED && value instanceof Item item
			&& item.components().has(DataComponents.CONSUMABLE)) {
			event(Event.Type.CONSUMED, BuiltInRegistries.ITEM.getId(item), count);
		} else if (stat.getType() == Stats.CUSTOM && Stats.DAMAGE_TAKEN.equals(value)) {
			event(Event.Type.DAMAGE, 0, count / 10.0F);
		} else if (stat.getType() == Stats.CUSTOM && Stats.DAMAGE_DEALT.equals(value)) {
			event(Event.Type.ATTACKED, 0, count / 10.0F);
		}
	}

	private void event(Event.Type type, int id, float amount) {
		events.add(Event.newBuilder().setType(type).setItemId(id).setAmount(amount).build());
	}

	public boolean isEyeInWater() {
		return isEyeInFluid(FluidTags.WATER);
	}
}

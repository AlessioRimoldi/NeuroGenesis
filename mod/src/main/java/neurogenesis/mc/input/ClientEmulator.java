package neurogenesis.mc.input;

import it.unimi.dsi.fastutil.ints.Int2ObjectMaps;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.network.HashedStack;
import net.minecraft.network.protocol.game.ServerboundAttackPacket;
import net.minecraft.network.protocol.game.ServerboundContainerClickPacket;
import net.minecraft.network.protocol.game.ServerboundContainerClosePacket;
import net.minecraft.network.protocol.game.ServerboundPlayerActionPacket;
import net.minecraft.network.protocol.game.ServerboundPlayerInputPacket;
import net.minecraft.network.protocol.game.ServerboundPunchPacket;
import net.minecraft.network.protocol.game.ServerboundSetCarriedItemPacket;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import net.minecraft.util.Mth;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Input;
import net.minecraft.world.inventory.AbstractContainerMenu;
import net.minecraft.world.inventory.ContainerInput;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;
import org.jspecify.annotations.Nullable;

import neurogenesis.mc.agent.AgentPlayer;
import neurogenesis.mc.proto.v1.ClickSlot;
import neurogenesis.mc.proto.v1.InventoryAction;

/**
 * Emulates the vanilla client for one agent: keybind handling (Minecraft.handleKeybinds), the block-breaking state
 * machine (MultiPlayerGameMode) and screen state. Everything the client would send as a packet is passed to the
 * agent's vanilla {@link ServerGamePacketListenerImpl} handler in-process, so the server applies exactly the same
 * validation it applies to human players.
 */
public final class ClientEmulator {
	private static final int RIGHT_CLICK_DELAY = 4;
	private static final int DESTROY_DELAY = 5;
	private static final int MISS_TIME = 10;

	private final AgentPlayer player;

	// Minecraft
	private boolean prevAttack;
	private boolean prevUse;
	private int rightClickDelay;
	private int missTime;
	private @Nullable HitResult hitResult;

	// MultiPlayerGameMode
	private boolean isDestroying;
	private BlockPos destroyBlockPos = new BlockPos(-1, -1, -1);
	private ItemStack destroyingItem = ItemStack.EMPTY;
	private @Nullable Direction destroyDirection;
	private float destroyProgress;
	private int destroyDelay;
	private int sequence;

	// Screen and input sync
	private boolean screenOpen;
	private int lastContainerId;
	private @Nullable Input lastSentInput;

	public ClientEmulator(AgentPlayer player) {
		this.player = player;
	}

	public void reset() {
		prevAttack = prevUse = false;
		rightClickDelay = missTime = 0;
		hitResult = null;
		isDestroying = false;
		destroyingItem = ItemStack.EMPTY;
		destroyDirection = null;
		destroyProgress = 0;
		destroyDelay = 0;
		screenOpen = false;
		lastContainerId = 0;
		lastSentInput = null;
	}

	public boolean isScreenOpen() {
		return screenOpen;
	}

	public @Nullable HitResult lastHitResult() {
		return hitResult;
	}

	/** Runs before the agent's server tick, like the client's tick runs before its packets reach the server. */
	public void tick(AgentInput in) {
		ServerGamePacketListenerImpl net = player.connection;
		if (rightClickDelay > 0) {
			rightClickDelay--;
		}
		if (missTime > 0) {
			missTime--;
		}

		syncScreenWithServer();
		if (in.hasOneShots()) {
			applyOneShots(in, net);
			in.clearOneShots();
		}

		// While a screen is open the client releases all key mappings.
		boolean keys = !screenOpen && player.isAlive() && !player.isSpectator();
		Input input = keys
			? new Input(in.forward, in.back, in.left, in.right, in.jump, in.sneak, in.sprint)
			: Input.EMPTY;
		if (!input.equals(lastSentInput)) {
			net.handlePlayerInput(new ServerboundPlayerInputPacket(input));
			lastSentInput = input;
		}

		boolean attackDown = keys && in.attack;
		boolean useDown = keys && in.use;
		hitResult = Crosshair.pick(player);
		if (player.isUsingItem()) {
			if (!useDown) {
				sendAction(ServerboundPlayerActionPacket.Action.RELEASE_USE_ITEM, BlockPos.ZERO, Direction.DOWN, 0);
			}
		} else if (keys) {
			if (attackDown && !prevAttack) {
				startAttack();
			}
			if (useDown && !prevUse) {
				startUseItem();
			}
		}
		if (useDown && rightClickDelay == 0 && !player.isUsingItem()) {
			startUseItem();
		}
		continueAttack(attackDown);
		prevAttack = attackDown;
		prevUse = useDown;
	}

	/** The client's movement vector (KeyboardInput.tick), already modified like LocalPlayer.modifyInput. */
	public Vec2 moveVector(AgentInput in, boolean movingSlowly) {
		if (screenOpen || !player.isAlive() || player.isSpectator()) {
			return Vec2.ZERO;
		}
		float forward = impulse(in.forward, in.back);
		float left = impulse(in.left, in.right);
		Vec2 input = new Vec2(left, forward).normalized();
		if (input.lengthSquared() == 0.0F) {
			return input;
		}
		Vec2 modified = input.scale(0.98F);
		if (player.isUsingItem() && !player.isPassenger()) {
			modified = modified.scale(player.getUseItem().getOrDefault(DataComponents.USE_EFFECTS,
				net.minecraft.world.item.component.UseEffects.DEFAULT).speedMultiplier());
		}
		if (movingSlowly) {
			modified = modified.scale((float) player.getAttributeValue(net.minecraft.world.entity.ai.attributes.Attributes.SNEAKING_SPEED));
		}
		return toSquare(modified);
	}

	// ------------------------------------------------------------------ one-shot actions

	private void applyOneShots(AgentInput in, ServerGamePacketListenerImpl net) {
		if (in.cameraDyaw != 0 || in.cameraDpitch != 0) {
			player.setYRot(player.getYRot() + in.cameraDyaw);
			player.setXRot(Mth.clamp(player.getXRot() + in.cameraDpitch, -90.0F, 90.0F));
			player.setYHeadRot(player.getYRot());
		}
		if (in.hotbar >= 1 && in.hotbar <= 9 && !screenOpen) {
			net.handleSetCarriedItem(new ServerboundSetCarriedItemPacket(in.hotbar - 1));
		}
		switch (in.inventory) {
			case INVENTORY_OPEN -> openInventory();
			case INVENTORY_CLOSE -> closeScreen(net);
			case INVENTORY_TOGGLE -> {
				if (screenOpen) {
					closeScreen(net);
				} else {
					openInventory();
				}
			}
			default -> {
			}
		}
		if (screenOpen) {
			for (ClickSlot click : in.clicks) {
				AbstractContainerMenu menu = player.containerMenu;
				net.handleContainerClick(new ServerboundContainerClickPacket(menu.containerId, menu.getStateId(),
					(short) click.getSlot(), (byte) click.getButton(), containerInput(click.getMode()),
					Int2ObjectMaps.emptyMap(), HashedStack.EMPTY));
			}
		} else if (player.isAlive() && !player.isSpectator()) {
			if (in.swapHands) {
				sendAction(ServerboundPlayerActionPacket.Action.SWAP_ITEM_WITH_OFFHAND, BlockPos.ZERO, Direction.DOWN, 0);
			}
			if (in.dropStack) {
				sendAction(ServerboundPlayerActionPacket.Action.DROP_ALL_ITEMS, BlockPos.ZERO, Direction.DOWN, 0);
			} else if (in.drop) {
				sendAction(ServerboundPlayerActionPacket.Action.DROP_ITEM, BlockPos.ZERO, Direction.DOWN, 0);
			}
		}
	}

	private void openInventory() {
		if (player.isAlive() && !player.isSpectator()) {
			screenOpen = true;
			stopDestroyBlock();
		}
	}

	private void closeScreen(ServerGamePacketListenerImpl net) {
		if (screenOpen || player.containerMenu != player.inventoryMenu) {
			net.handleContainerClose(new ServerboundContainerClosePacket(player.containerMenu.containerId));
		}
		screenOpen = false;
	}

	/** The server opens screens (crafting table, chest) and closes them (out of reach); follow it like a client. */
	private void syncScreenWithServer() {
		int containerId = player.containerMenu.containerId;
		if (containerId != lastContainerId) {
			if (player.containerMenu != player.inventoryMenu) {
				screenOpen = true;
				stopDestroyBlock();
			} else {
				screenOpen = false;
			}
			lastContainerId = containerId;
		}
	}

	private static ContainerInput containerInput(neurogenesis.mc.proto.v1.ClickMode mode) {
		return switch (mode) {
			case CLICK_QUICK_MOVE -> ContainerInput.QUICK_MOVE;
			case CLICK_SWAP -> ContainerInput.SWAP;
			case CLICK_CLONE -> ContainerInput.CLONE;
			case CLICK_THROW -> ContainerInput.THROW;
			case CLICK_QUICK_CRAFT -> ContainerInput.QUICK_CRAFT;
			case CLICK_PICKUP_ALL -> ContainerInput.PICKUP_ALL;
			default -> ContainerInput.PICKUP;
		};
	}

	// ------------------------------------------------------------------ attack (Minecraft.startAttack/continueAttack)

	private boolean hasMissTime() {
		return !player.isCreative();
	}

	private void startAttack() {
		if (missTime > 0) {
			return;
		}
		if (hitResult == null) {
			if (hasMissTime()) {
				missTime = MISS_TIME;
			}
			return;
		}
		ItemStack held = player.getItemInHand(InteractionHand.MAIN_HAND);
		if (!held.isItemEnabled(player.level().enabledFeatures()) || player.cannotAttackWithItem(held, 0)) {
			return;
		}
		if (held.has(DataComponents.PIERCING_WEAPON)) {
			return; // TODO: spears (piercing attacks) are not emulated yet
		}
		switch (hitResult.getType()) {
			case ENTITY -> player.connection.handleAttack(new ServerboundAttackPacket(((EntityHitResult) hitResult).getEntity().getId()));
			case BLOCK -> {
				BlockHitResult blockHit = (BlockHitResult) hitResult;
				if (!player.level().getBlockState(blockHit.getBlockPos()).isAir()) {
					startDestroyBlock(blockHit.getBlockPos(), blockHit.getDirection());
				} else {
					onMiss();
				}
			}
			case MISS -> onMiss();
		}
		player.connection.handlePunch(ServerboundPunchPacket.INSTANCE);
	}

	private void onMiss() {
		if (hasMissTime()) {
			missTime = MISS_TIME;
		}
		player.resetAttackStrengthTicker();
	}

	private void continueAttack(boolean down) {
		if (!down) {
			missTime = 0;
		}
		if (missTime > 0 || player.isUsingItem()) {
			return;
		}
		if (player.getItemInHand(InteractionHand.MAIN_HAND).has(DataComponents.PIERCING_WEAPON)) {
			return;
		}
		if (down && hitResult instanceof BlockHitResult blockHit && hitResult.getType() == HitResult.Type.BLOCK) {
			BlockPos pos = blockHit.getBlockPos();
			if (!player.level().getBlockState(pos).isAir() && continueDestroyBlock(pos, blockHit.getDirection())) {
				player.connection.handlePunch(ServerboundPunchPacket.INSTANCE);
			}
		} else {
			stopDestroyBlock();
		}
	}

	// ------------------------------------------------------------------ block breaking (MultiPlayerGameMode)

	private boolean startDestroyBlock(BlockPos pos, Direction direction) {
		ServerLevel level = player.level();
		if (player.blockActionRestricted(level, pos, player.gameMode.getGameModeForPlayer())
			|| !level.getWorldBorder().isWithinBounds(pos)) {
			return false;
		}
		if (player.getAbilities().instabuild) {
			sendAction(ServerboundPlayerActionPacket.Action.START_DESTROY_BLOCK, pos, direction, ++sequence);
			destroyDelay = DESTROY_DELAY;
		} else if (!isDestroying || !sameDestroyTarget(pos)) {
			if (isDestroying) {
				sendAction(ServerboundPlayerActionPacket.Action.ABORT_DESTROY_BLOCK, destroyBlockPos, direction, 0);
			}
			BlockState state = level.getBlockState(pos);
			boolean instant = !state.isAir() && state.getDestroyProgress(player, level, pos) >= 1.0F;
			if (!instant) {
				isDestroying = true;
				destroyBlockPos = pos;
				destroyDirection = direction;
				destroyingItem = player.getMainHandItem().copy();
				destroyProgress = 0.0F;
			}
			sendAction(ServerboundPlayerActionPacket.Action.START_DESTROY_BLOCK, pos, direction, ++sequence);
		}
		return true;
	}

	private boolean continueDestroyBlock(BlockPos pos, Direction direction) {
		if (destroyDelay > 0) {
			destroyDelay--;
			return true;
		}
		ServerLevel level = player.level();
		if (player.getAbilities().instabuild && level.getWorldBorder().isWithinBounds(pos)) {
			destroyDelay = DESTROY_DELAY;
			sendAction(ServerboundPlayerActionPacket.Action.START_DESTROY_BLOCK, pos, direction, ++sequence);
			return true;
		}
		if (!sameDestroyTarget(pos)) {
			return startDestroyBlock(pos, direction);
		}
		BlockState state = level.getBlockState(pos);
		if (state.isAir()) {
			isDestroying = false;
			return false;
		}
		destroyProgress += state.getDestroyProgress(player, level, pos);
		if (destroyProgress >= 1.0F) {
			isDestroying = false;
			sendAction(ServerboundPlayerActionPacket.Action.STOP_DESTROY_BLOCK, pos, direction, ++sequence);
			destroyDirection = null;
			destroyProgress = 0.0F;
			destroyDelay = DESTROY_DELAY;
		} else if (destroyDirection != direction) {
			destroyDirection = direction;
			sendAction(ServerboundPlayerActionPacket.Action.CHANGE_DESTROY_DIRECTION, pos, direction, 0);
		}
		return true;
	}

	private void stopDestroyBlock() {
		if (isDestroying) {
			sendAction(ServerboundPlayerActionPacket.Action.ABORT_DESTROY_BLOCK, destroyBlockPos, Direction.DOWN, 0);
			isDestroying = false;
			destroyProgress = 0.0F;
			player.resetAttackStrengthTicker();
		}
	}

	private boolean sameDestroyTarget(BlockPos pos) {
		return pos.equals(destroyBlockPos) && ItemStack.isSameItemSameComponents(player.getMainHandItem(), destroyingItem);
	}

	/** Fraction of the current block broken so far (0 when not mining). */
	public float destroyProgress() {
		return isDestroying ? destroyProgress : 0.0F;
	}

	// ------------------------------------------------------------------ use (Minecraft.startUseItem)

	/**
	 * Mirrors the client's hand loop. The interactions themselves run the same server code as the vanilla packet
	 * handlers (handleInteract / handleUseItemOn / handleUseItem), called directly so that the result is available
	 * to decide whether to try the next hand.
	 */
	private void startUseItem() {
		if (isDestroying) {
			return;
		}
		rightClickDelay = RIGHT_CLICK_DELAY;
		ServerLevel level = player.level();
		for (InteractionHand hand : InteractionHand.values()) {
			ItemStack held = player.getItemInHand(hand);
			if (!held.isItemEnabled(level.enabledFeatures())) {
				return;
			}
			if (hitResult != null) {
				switch (hitResult.getType()) {
					case ENTITY -> {
						EntityHitResult entityHit = (EntityHitResult) hitResult;
						Entity entity = entityHit.getEntity();
						if (!level.getWorldBorder().isWithinBounds(entity.blockPosition())) {
							return;
						}
						if (player.isWithinEntityInteractionRange(entity, 0.0)) {
							player.resetLastActionTime();
							Vec3 location = entityHit.getLocation().subtract(entity.getX(), entity.getY(), entity.getZ());
							ItemStack used = held.copy();
							if (player.interactOn(entity, hand, location) instanceof InteractionResult.Success success) {
								if (success.shouldSwing()) {
									player.swingAndResetAttackStrength(hand, used.getInteractAnimation(), true);
								}
								return;
							}
						}
					}
					case BLOCK -> {
						InteractionResult result = useItemOn(hand, held, (BlockHitResult) hitResult);
						if (result instanceof InteractionResult.Success || result instanceof InteractionResult.Fail) {
							return;
						}
					}
					default -> {
					}
				}
			}
			if (!held.isEmpty()) {
				player.resetLastActionTime();
				if (player.gameMode.useItem(player, level, held, hand) instanceof InteractionResult.Success success) {
					if (success.shouldSwing()) {
						player.swingAndResetAttackStrength(hand, held.getInteractAnimation(), true);
					}
					return;
				}
			}
		}
	}

	/** The checks of ServerGamePacketListenerImpl.handleUseItemOn, returning the interaction result. */
	private InteractionResult useItemOn(InteractionHand hand, ItemStack held, BlockHitResult hit) {
		ServerLevel level = player.level();
		BlockPos pos = hit.getBlockPos();
		if (!player.isWithinBlockInteractionRange(pos, 1.0)) {
			return InteractionResult.PASS;
		}
		if (pos.getY() > level.getMaxY() || pos.getY() < level.getMinY()) {
			return InteractionResult.FAIL;
		}
		if (player.level().getServer().isUnderSpawnProtection(level, pos, player) || !level.mayInteract(player, pos)) {
			return InteractionResult.FAIL;
		}
		player.resetLastActionTime();
		InteractionResult result = player.gameMode.useItemOn(player, level, held, hand, hit);
		if (result instanceof InteractionResult.Success success && success.shouldSwing()) {
			player.swingAndResetAttackStrength(hand, held.getInteractAnimation(), true);
		}
		return result;
	}

	// ------------------------------------------------------------------ helpers

	private void sendAction(ServerboundPlayerActionPacket.Action action, BlockPos pos, Direction direction, int seq) {
		player.connection.handlePlayerAction(new ServerboundPlayerActionPacket(action, pos, direction, seq));
	}

	private static float impulse(boolean positive, boolean negative) {
		return positive == negative ? 0.0F : positive ? 1.0F : -1.0F;
	}

	/** LocalPlayer.modifyInputSpeedForSquareMovement. */
	private static Vec2 toSquare(Vec2 input) {
		float length = input.length();
		if (length <= 0.0F) {
			return input;
		}
		Vec2 direction = input.scale(1.0F / length);
		float x = Math.abs(direction.x);
		float y = Math.abs(direction.y);
		float tan = y > x ? x / y : y / x;
		float distanceToUnitSquare = Mth.sqrt(1.0F + Mth.square(tan));
		return direction.scale(Math.min(length * distanceToUnitSquare, 1.0F));
	}
}

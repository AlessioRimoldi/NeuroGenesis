package neurogenesis.mc.mixin;

import java.util.Set;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import net.minecraft.world.entity.PositionMoveRotation;
import net.minecraft.world.entity.Relative;
import net.minecraft.world.phys.Vec3;
import org.jspecify.annotations.Nullable;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import neurogenesis.mc.agent.AgentListenerHooks;
import neurogenesis.mc.agent.AgentPlayer;

@Mixin(ServerGamePacketListenerImpl.class)
public abstract class ServerGamePacketListenerImplMixin implements AgentListenerHooks {
	@Shadow
	public ServerPlayer player;
	@Shadow
	private @Nullable Vec3 awaitingPositionFromClient;
	@Shadow
	private boolean waitingForRespawn;
	@Shadow
	private int clientLoadedTimeoutTimer;

	@Shadow
	public abstract void resetPosition();

	@Override
	public void ng$markClientLoaded() {
		waitingForRespawn = false;
		clientLoadedTimeoutTimer = 0;
	}

	/**
	 * A real client acknowledges every teleport; until then the server refuses block interactions. Agents have no
	 * client, so acknowledge immediately (as Carpet's fake players do).
	 */
	@Inject(method = "teleport(Lnet/minecraft/world/entity/PositionMoveRotation;Ljava/util/Set;)V", at = @At("TAIL"))
	private void ng$acceptAgentTeleport(PositionMoveRotation destination, Set<Relative> relatives, CallbackInfo ci) {
		if (player instanceof AgentPlayer) {
			awaitingPositionFromClient = null;
			resetPosition();
			// placeNewPlayer teleports before the player is added to the level; chunk tracking starts later.
			if (player.level().getPlayerByUUID(player.getUUID()) != null) {
				player.level().getChunkSource().move(player);
			}
		}
	}
}

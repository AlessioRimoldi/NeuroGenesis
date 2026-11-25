package neurogenesis.mc.mixin;

import com.llamalad7.mixinextras.injector.ModifyExpressionValue;
import com.llamalad7.mixinextras.sugar.Local;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

import neurogenesis.mc.agent.AgentPlayer;

/**
 * When a player hits another player, vanilla sends the knockback to the victim's client and restores the server-side
 * velocity. Agents are simulated on the server, so they must keep the velocity (adapted from Carpet, MIT).
 */
@Mixin(Player.class)
public abstract class PlayerKnockbackMixin {
	@ModifyExpressionValue(method = "causeExtraKnockback",
		at = @At(value = "FIELD", target = "Lnet/minecraft/world/entity/Entity;syncVelocity:Z", ordinal = 0))
	private boolean ng$keepAgentKnockback(boolean original, @Local(argsOnly = true) Entity target) {
		return original && !(target instanceof AgentPlayer);
	}
}

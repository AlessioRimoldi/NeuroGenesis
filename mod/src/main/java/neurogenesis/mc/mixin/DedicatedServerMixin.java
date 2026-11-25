package neurogenesis.mc.mixin;

import net.minecraft.server.dedicated.DedicatedServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.sim.Lockstep;

/** "pause-when-empty" would silently skip ticks while a Python client steps an empty world. */
@Mixin(DedicatedServer.class)
public abstract class DedicatedServerMixin {
	@Inject(method = "pauseWhenEmptySeconds", at = @At("HEAD"), cancellable = true)
	private void ng$neverPauseUnderLockstep(CallbackInfoReturnable<Integer> cir) {
		Lockstep lockstep = NeuroGenesisMod.lockstepFor((DedicatedServer) (Object) this);
		if (lockstep != null && lockstep.isActive()) {
			cir.setReturnValue(0);
		}
	}
}

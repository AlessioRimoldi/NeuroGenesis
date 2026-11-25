package neurogenesis.mc.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import neurogenesis.mc.NeuroGenesisMod;
import neurogenesis.mc.sim.Lockstep;

/** Hooks the lockstep barrier into the vanilla server loop. */
@Mixin(MinecraftServer.class)
public abstract class MinecraftServerMixin {
	@Inject(method = "waitUntilNextTick", at = @At("HEAD"))
	private void ng$lockstepBarrier(CallbackInfo ci) {
		Lockstep lockstep = NeuroGenesisMod.lockstepFor((MinecraftServer) (Object) this);
		if (lockstep != null) {
			lockstep.onTickBoundary();
		}
	}

	/** While paused at the barrier, let every queued task run (vanilla would defer "current tick" tasks). */
	@Inject(method = "haveTime", at = @At("HEAD"), cancellable = true)
	private void ng$haveTimeWhileWaiting(CallbackInfoReturnable<Boolean> cir) {
		Lockstep lockstep = NeuroGenesisMod.lockstepFor((MinecraftServer) (Object) this);
		if (lockstep != null && lockstep.isWaiting()) {
			cir.setReturnValue(true);
		}
	}

	/** Vanilla polls every 100 µs between tasks; while paused, park longer (new tasks unpark the thread). */
	@Inject(method = "waitForTasks", at = @At("HEAD"), cancellable = true)
	private void ng$parkWhileWaiting(CallbackInfo ci) {
		Lockstep lockstep = NeuroGenesisMod.lockstepFor((MinecraftServer) (Object) this);
		if (lockstep != null && lockstep.isWaiting()) {
			lockstep.parkWhileWaiting();
			ci.cancel();
		}
	}
}

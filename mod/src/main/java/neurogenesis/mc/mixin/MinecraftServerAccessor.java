package neurogenesis.mc.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(MinecraftServer.class)
public interface MinecraftServerAccessor {
	@Accessor("nextTickTimeNanos")
	void ng$setNextTickTimeNanos(long value);

	@Accessor("delayedTasksMaxNextTickTimeNanos")
	void ng$setDelayedTasksMaxNextTickTimeNanos(long value);
}

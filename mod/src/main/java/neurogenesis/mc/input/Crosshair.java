package neurogenesis.mc.input;

import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.EntitySelector;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.projectile.ProjectileUtil;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/** What the agent's crosshair points at, computed exactly like the vanilla client (LocalPlayer.pick). */
public final class Crosshair {
	private Crosshair() {
	}

	public static HitResult pick(Player player) {
		double blockRange = player.blockInteractionRange();
		double entityRange = player.entityInteractionRange();
		double maxDistance = Math.max(blockRange, entityRange);
		double maxDistanceSq = Mth.square(maxDistance);
		Vec3 from = player.getEyePosition(1.0F);
		HitResult blockHit = player.pick(maxDistance, 1.0F, false);
		double blockDistanceSq = blockHit.getLocation().distanceToSqr(from);
		if (blockHit.getType() != HitResult.Type.MISS) {
			maxDistanceSq = blockDistanceSq;
			maxDistance = Math.sqrt(maxDistanceSq);
		}
		Vec3 direction = player.getViewVector(1.0F);
		Vec3 to = from.add(direction.x * maxDistance, direction.y * maxDistance, direction.z * maxDistance);
		AABB box = player.getBoundingBox().expandTowards(direction.scale(maxDistance)).inflate(1.0, 1.0, 1.0);
		EntityHitResult entityHit = ProjectileUtil.getEntityHitResult(player, from, to, box, EntitySelector.CAN_BE_PICKED, maxDistanceSq);
		return entityHit != null && entityHit.getLocation().distanceToSqr(from) < blockDistanceSq
			? filter(entityHit, from, entityRange)
			: filter(blockHit, from, blockRange);
	}

	private static HitResult filter(HitResult hit, Vec3 from, double maxRange) {
		Vec3 location = hit.getLocation();
		if (!location.closerThan(from, maxRange)) {
			Direction direction = Direction.getApproximateNearest(location.x - from.x, location.y - from.y, location.z - from.z);
			return BlockHitResult.miss(location, direction, BlockPos.containing(location));
		}
		return hit;
	}
}

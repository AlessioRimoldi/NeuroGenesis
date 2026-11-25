package neurogenesis.mc.sense;

import net.minecraft.util.Mth;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.Vec3;

/** Two "antennas" left and right of the agent's head, sampling the smell field (bilateral, like a worm). */
public final class SmellSense {
	private final SmellField field;
	private final double lateral;
	private final double forward;

	public SmellSense(SmellField field, double lateral, double forward) {
		this.field = field;
		this.lateral = lateral;
		this.forward = forward;
	}

	public SmellField field() {
		return field;
	}

	/** Returns {left, right}. */
	public double[] sample(Player player) {
		Vec3 eye = player.getEyePosition();
		float yawRad = player.getYRot() * Mth.DEG_TO_RAD;
		// Minecraft: yaw 0 faces +Z (south); forward = (-sin(yaw), cos(yaw)); the player's left is (cos(yaw), sin(yaw)).
		double fx = -Mth.sin(yawRad), fz = Mth.cos(yawRad);
		double lx = Mth.cos(yawRad), lz = Mth.sin(yawRad);
		double cx = eye.x + fx * forward, cz = eye.z + fz * forward;
		double left = field.sample(cx + lx * lateral, cz + lz * lateral);
		double right = field.sample(cx - lx * lateral, cz - lz * lateral);
		return new double[]{left, right};
	}
}

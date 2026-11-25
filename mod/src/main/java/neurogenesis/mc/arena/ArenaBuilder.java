package neurogenesis.mc.arena;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;

/**
 * Builds a walled arena on a floor slab and clears everything else in the area (including what the previous
 * episode's arena left behind). Only blocks that differ from the target are written.
 */
public final class ArenaBuilder {
	/** Height above the floor that is cleared to air each reset. */
	public static final int CLEAR_HEIGHT = 24;

	private int prevMinX, prevMinZ, prevMaxX, prevMaxZ; // previous footprint incl. walls (inclusive)
	private boolean hasPrev;

	public void build(ServerLevel level, ArenaSpec spec) {
		int ox = ArenaSpec.ORIGIN.getX();
		int oz = ArenaSpec.ORIGIN.getZ();
		int oy = ArenaSpec.ORIGIN.getY();
		int minX = ox - 1, minZ = oz - 1, maxX = ox + spec.sizeX(), maxZ = oz + spec.sizeZ();
		int areaMinX = hasPrev ? Math.min(minX, prevMinX) : minX;
		int areaMinZ = hasPrev ? Math.min(minZ, prevMinZ) : minZ;
		int areaMaxX = hasPrev ? Math.max(maxX, prevMaxX) : maxX;
		int areaMaxZ = hasPrev ? Math.max(maxZ, prevMaxZ) : maxZ;

		for (int cx = areaMinX >> 4; cx <= areaMaxX >> 4; cx++) {
			for (int cz = areaMinZ >> 4; cz <= areaMaxZ >> 4; cz++) {
				level.setChunkForced(cx, cz, true);
			}
		}

		// Remove leftovers first so that falling/dropped items do not survive the rebuild.
		AABB area = new AABB(areaMinX - 2, oy - 4, areaMinZ - 2, areaMaxX + 3, oy + CLEAR_HEIGHT + 8, areaMaxZ + 3);
		for (Entity entity : level.getEntities((Entity) null, area, e -> !(e instanceof Player))) {
			entity.discard();
		}

		BlockState floor = spec.floor().defaultBlockState();
		BlockState wall = spec.wall().defaultBlockState();
		BlockState air = Blocks.AIR.defaultBlockState();
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		int flags = Block.UPDATE_CLIENTS | Block.UPDATE_KNOWN_SHAPE;
		for (int x = areaMinX; x <= areaMaxX; x++) {
			for (int z = areaMinZ; z <= areaMaxZ; z++) {
				boolean inFootprint = x >= minX && x <= maxX && z >= minZ && z <= maxZ;
				boolean onWall = inFootprint && (x == minX || x == maxX || z == minZ || z == maxZ);
				for (int y = oy - 1; y < oy + CLEAR_HEIGHT; y++) {
					BlockState target;
					if (y == oy - 1) {
						target = inFootprint ? floor : air;
					} else if (onWall && y < oy + spec.wallHeight()) {
						target = wall;
					} else {
						target = air;
					}
					pos.set(x, y, z);
					if (level.getBlockState(pos) != target) {
						level.setBlock(pos, target, flags);
					}
				}
			}
		}
		prevMinX = minX;
		prevMinZ = minZ;
		prevMaxX = maxX;
		prevMaxZ = maxZ;
		hasPrev = true;
	}
}

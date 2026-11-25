package neurogenesis.mc.arena;

import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.block.Block;

import neurogenesis.mc.proto.v1.ArenaConfig;
import neurogenesis.mc.proto.v1.FoodConfig;
import neurogenesis.mc.proto.v1.SmellConfig;

/** ArenaConfig with defaults applied and names resolved. The inner floor area starts at {@link #ORIGIN}. */
public record ArenaSpec(int sizeX, int sizeZ, int wallHeight, Block floor, Block wall, FoodConfig food,
						String foodId, SmellConfig smell) {
	public static final BlockPos ORIGIN = new BlockPos(0, 64, 0);
	public static final int MAX_SIZE = 256;

	public static ArenaSpec from(ArenaConfig c) {
		int sx = c.getSizeX() > 0 ? c.getSizeX() : 32;
		int sz = c.getSizeZ() > 0 ? c.getSizeZ() : 32;
		if (sx > MAX_SIZE || sz > MAX_SIZE) {
			throw new IllegalArgumentException("arena sides must be <= " + MAX_SIZE);
		}
		int wall = c.getWallHeight() > 0 ? c.getWallHeight() : 3;
		Block floor = block(c.getFloorBlock().isEmpty() ? "minecraft:grass_block" : c.getFloorBlock());
		Block wallBlock = block(c.getWallBlock().isEmpty() ? "minecraft:stone_bricks" : c.getWallBlock());
		String foodId = c.getFood().getId().isEmpty() ? "minecraft:apple" : c.getFood().getId();
		return new ArenaSpec(sx, sz, wall, floor, wallBlock, c.getFood(), foodId, c.getSmell());
	}

	public int floorY() {
		return ORIGIN.getY() - 1;
	}

	public boolean insideXZ(double x, double z) {
		return x >= ORIGIN.getX() && x < ORIGIN.getX() + sizeX && z >= ORIGIN.getZ() && z < ORIGIN.getZ() + sizeZ;
	}

	static Block block(String id) {
		Identifier key = Identifier.parse(id);
		return BuiltInRegistries.BLOCK.getOptional(key)
			.orElseThrow(() -> new IllegalArgumentException("unknown block " + id));
	}
}

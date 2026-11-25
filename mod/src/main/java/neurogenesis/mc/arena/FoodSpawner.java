package neurogenesis.mc.arena;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

import neurogenesis.mc.proto.v1.FoodConfig;

/** Places food (item entities or blocks) in the arena and replaces eaten food if configured. */
public final class FoodSpawner {
	private final List<ItemEntity> items = new ArrayList<>();
	private final List<BlockPos> blocks = new ArrayList<>();
	private final List<Integer> respawnTimers = new ArrayList<>();
	private ArenaSpec spec;
	private SplittableRandom rng;
	private List<Vec3> avoid = List.of();
	private Item item;
	private BlockState block;

	public void reset(ServerLevel level, ArenaSpec spec, SplittableRandom rng, List<Vec3> avoid) {
		this.spec = spec;
		this.rng = rng;
		this.avoid = avoid;
		items.clear();
		blocks.clear();
		respawnTimers.clear();
		FoodConfig food = spec.food();
		Identifier id = Identifier.parse(spec.foodId());
		if (food.getKind() == FoodConfig.Kind.KIND_BLOCK) {
			Block b = BuiltInRegistries.BLOCK.getOptional(id).orElseThrow(() -> new IllegalArgumentException("unknown food block " + id));
			block = b.defaultBlockState();
			item = null;
		} else {
			item = BuiltInRegistries.ITEM.getOptional(id).orElseThrow(() -> new IllegalArgumentException("unknown food item " + id));
			block = null;
		}
		for (int i = 0; i < food.getCount(); i++) {
			place(level);
		}
	}

	/** Call once per tick: notices eaten food and schedules/places replacements. */
	public void tick(ServerLevel level) {
		if (spec == null) {
			return;
		}
		items.removeIf(ItemEntity::isRemoved);
		if (block != null) {
			blocks.removeIf(pos -> level.getBlockState(pos).getBlock() != block.getBlock());
		}
		if (spec.food().getRespawn()) {
			int present = items.size() + blocks.size() + respawnTimers.size();
			for (int i = present; i < spec.food().getCount(); i++) {
				respawnTimers.add((int) spec.food().getRespawnDelay());
			}
			for (int i = respawnTimers.size() - 1; i >= 0; i--) {
				int t = respawnTimers.get(i) - 1;
				if (t <= 0) {
					respawnTimers.remove(i);
					place(level);
				} else {
					respawnTimers.set(i, t);
				}
			}
		}
	}

	/** Current smell sources (centre of each food item or block). */
	public List<Vec3> sources() {
		List<Vec3> out = new ArrayList<>(items.size() + blocks.size());
		for (ItemEntity e : items) {
			if (!e.isRemoved()) {
				out.add(e.position());
			}
		}
		for (BlockPos p : blocks) {
			out.add(Vec3.atBottomCenterOf(p));
		}
		return out;
	}

	private void place(ServerLevel level) {
		float minDist = spec.food().getMinDist();
		for (int attempt = 0; attempt < 200; attempt++) {
			int x = ArenaSpec.ORIGIN.getX() + rng.nextInt(spec.sizeX());
			int z = ArenaSpec.ORIGIN.getZ() + rng.nextInt(spec.sizeZ());
			Vec3 p = new Vec3(x + 0.5, ArenaSpec.ORIGIN.getY(), z + 0.5);
			if (minDist > 0 && avoid.stream().anyMatch(a -> a.distanceTo(p) < minDist)) {
				continue;
			}
			BlockPos pos = BlockPos.containing(p);
			if (!level.getBlockState(pos).isAir()) {
				continue;
			}
			if (block != null) {
				level.setBlockAndUpdate(pos, block);
				blocks.add(pos);
			} else {
				ItemEntity e = new ItemEntity(level, p.x, p.y + 0.1, p.z, new ItemStack(item));
				e.setDeltaMovement(Vec3.ZERO);
				e.setNoPickUpDelay();
				e.setUnlimitedLifetime();
				level.addFreshEntity(e);
				items.add(e);
			}
			return;
		}
	}
}

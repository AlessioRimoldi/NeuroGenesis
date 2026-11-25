package neurogenesis.mc.obs;

import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.inventory.AbstractContainerMenu;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import org.jspecify.annotations.Nullable;

import neurogenesis.mc.agent.AgentPlayer;
import neurogenesis.mc.input.Crosshair;
import neurogenesis.mc.proto.v1.AgentObs;
import neurogenesis.mc.proto.v1.ContainerView;
import neurogenesis.mc.proto.v1.Smell;
import neurogenesis.mc.proto.v1.Vec3d;
import neurogenesis.mc.proto.v1.Vec3i;
import neurogenesis.mc.sense.SmellSense;

/** Full server-side state of an agent; Python decides what each agent is allowed to perceive. */
public final class ObservationBuilder {
	public static final int INVENTORY_SLOTS = 41; // 0-35 main (0-8 hotbar), 36-39 armor, 40 offhand

	private ObservationBuilder() {
	}

	public static AgentObs build(AgentPlayer agent, @Nullable SmellSense smell, boolean truncated) {
		Vec3 pos = agent.position();
		Vec3 vel = agent.getDeltaMovement();
		AgentObs.Builder b = AgentObs.newBuilder()
			.setAgentId(agent.agentId())
			.setName(agent.getGameProfile().name())
			.setAlive(agent.isAlive() && !agent.isTerminated())
			.setTerminated(agent.isTerminated())
			.setTruncated(truncated)
			.setPos(Vec3d.newBuilder().setX(pos.x).setY(pos.y).setZ(pos.z))
			.setVel(Vec3d.newBuilder().setX(vel.x).setY(vel.y).setZ(vel.z))
			.setYaw(agent.getYRot())
			.setPitch(agent.getXRot())
			.setOnGround(agent.onGround())
			.setInWater(agent.isInWater())
			.setCrouching(agent.isCrouching())
			.setSprinting(agent.isSprinting())
			.setUsingItem(agent.isUsingItem())
			.setHealth(agent.getHealth())
			.setFood(agent.getFoodData().getFoodLevel())
			.setSaturation(agent.getFoodData().getSaturationLevel())
			.setAir(agent.getAirSupply())
			.setSelectedSlot(agent.getInventory().getSelectedSlot());

		Inventory inv = agent.getInventory();
		for (int slot = 0; slot < INVENTORY_SLOTS; slot++) {
			ItemStack stack = inv.getItem(slot);
			b.addInvItem(itemId(stack));
			b.addInvCount(stack.getCount());
		}

		b.setContainer(container(agent));
		b.setCrosshair(crosshair(agent));
		if (smell != null) {
			double[] lr = smell.sample(agent);
			b.setSmell(Smell.newBuilder().setLeft((float) lr[0]).setRight((float) lr[1]));
		}
		b.addAllEvents(agent.drainEvents());
		return b.build();
	}

	private static ContainerView container(AgentPlayer agent) {
		ContainerView.Builder v = ContainerView.newBuilder().setOpen(agent.client().isScreenOpen());
		if (!agent.client().isScreenOpen()) {
			return v.setMenuType(-1).build();
		}
		AbstractContainerMenu menu = agent.containerMenu;
		// InventoryMenu has no registered MenuType (getType() throws), so report it as -1.
		v.setMenuType(menu == agent.inventoryMenu ? -1 : BuiltInRegistries.MENU.getId(menu.getType()));
		for (var slot : menu.slots) {
			ItemStack stack = slot.getItem();
			v.addSlotItem(itemId(stack));
			v.addSlotCount(stack.getCount());
		}
		ItemStack carried = menu.getCarried();
		return v.setCarriedItem(itemId(carried)).setCarriedCount(carried.getCount()).build();
	}

	private static neurogenesis.mc.proto.v1.Crosshair crosshair(AgentPlayer agent) {
		HitResult hit = Crosshair.pick(agent);
		var c = neurogenesis.mc.proto.v1.Crosshair.newBuilder()
			.setDistance((float) hit.getLocation().distanceTo(agent.getEyePosition()));
		if (hit instanceof EntityHitResult e && hit.getType() == HitResult.Type.ENTITY) {
			c.setType(neurogenesis.mc.proto.v1.Crosshair.Type.ENTITY)
				.setEntityType(BuiltInRegistries.ENTITY_TYPE.getId(e.getEntity().getType()))
				.setEntityId(e.getEntity().getId());
		} else if (hit instanceof BlockHitResult bh && hit.getType() == HitResult.Type.BLOCK) {
			BlockPos p = bh.getBlockPos();
			c.setType(neurogenesis.mc.proto.v1.Crosshair.Type.BLOCK)
				.setBlockId(BuiltInRegistries.BLOCK.getId(agent.level().getBlockState(p).getBlock()))
				.setBlockPos(Vec3i.newBuilder().setX(p.getX()).setY(p.getY()).setZ(p.getZ()))
				.setFace(bh.getDirection().ordinal());
		}
		return c.build();
	}

	private static int itemId(ItemStack stack) {
		return stack.isEmpty() ? 0 : BuiltInRegistries.ITEM.getId(stack.getItem());
	}
}

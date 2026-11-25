package neurogenesis.mc.crafting;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.StackedItemContents;
import net.minecraft.world.inventory.AbstractContainerMenu;
import net.minecraft.world.inventory.CraftingMenu;
import net.minecraft.world.inventory.RecipeBookMenu;
import net.minecraft.world.inventory.RecipeBookType;
import net.minecraft.world.item.ItemStack;

import neurogenesis.mc.agent.AgentPlayer;

/**
 * Which unlocked recipes the agent could place right now — what the vanilla recipe book highlights. Results are
 * cached and only recomputed when the inventory, the open menu's grid or the recipe book changes.
 */
public final class CraftableTracker {
	private long lastFingerprint = Long.MIN_VALUE;
	private List<Integer> lastResult = List.of();

	public List<Integer> craftable(AgentPlayer agent, RecipeIndex index) {
		// The recipe book of the open screen; with no screen, the inventory's 2x2 grid.
		AbstractContainerMenu open = agent.client().isScreenOpen() ? agent.containerMenu : agent.inventoryMenu;
		RecipeBookMenu menu = open instanceof RecipeBookMenu rb ? rb : null;
		long fingerprint = fingerprint(agent, open);
		if (fingerprint == lastFingerprint) {
			return lastResult;
		}
		List<Integer> result = new ArrayList<>();
		if (menu != null) {
			StackedItemContents contents = new StackedItemContents();
			agent.getInventory().fillStackedContents(contents);
			menu.fillCraftSlotsStackedContents(contents);
			RecipeBookType type = menu.getRecipeBookType();
			boolean largeGrid = menu instanceof CraftingMenu;
			var book = agent.getRecipeBook();
			for (RecipeIndex.Entry entry : index.entries()) {
				if (RecipeIndex.fits(entry.station(), type, largeGrid)
					&& book.contains(entry.info().parent().id())
					&& entry.display().canCraft(contents)) {
					result.add(entry.proto().getDisplayId());
				}
			}
		}
		lastFingerprint = fingerprint;
		lastResult = List.copyOf(result);
		return lastResult;
	}

	public void invalidate() {
		lastFingerprint = Long.MIN_VALUE;
	}

	private static long fingerprint(AgentPlayer agent, AbstractContainerMenu menu) {
		long h = 1469598103934665603L;
		Inventory inv = agent.getInventory();
		for (int i = 0; i < inv.getContainerSize(); i++) {
			h = mix(h, stackHash(inv.getItem(i)));
		}
		h = mix(h, menu.containerId);
		h = mix(h, menu.slots.size());
		for (var slot : menu.slots) {
			if (slot.container != inv) {
				h = mix(h, stackHash(slot.getItem()));
			}
		}
		return mix(h, agent.recipeBookVersion());
	}

	private static long stackHash(ItemStack stack) {
		return stack.isEmpty() ? 0 : BuiltInRegistries.ITEM.getId(stack.getItem()) * 131L + stack.getCount();
	}

	private static long mix(long h, long v) {
		return (h ^ v) * 1099511628211L;
	}
}

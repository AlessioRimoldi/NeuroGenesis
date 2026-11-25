package neurogenesis.mc.crafting;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.inventory.RecipeBookType;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.crafting.RecipeBookCategories;
import net.minecraft.world.item.crafting.RecipeBookCategory;
import net.minecraft.world.item.crafting.RecipeManager;
import net.minecraft.world.item.crafting.display.FurnaceRecipeDisplay;
import net.minecraft.world.item.crafting.display.RecipeDisplay;
import net.minecraft.world.item.crafting.display.RecipeDisplayEntry;
import net.minecraft.world.item.crafting.display.RecipeDisplayId;
import net.minecraft.world.item.crafting.display.ShapedCraftingRecipeDisplay;
import net.minecraft.world.item.crafting.display.ShapelessCraftingRecipeDisplay;
import net.minecraft.world.item.crafting.display.SlotDisplayContext;
import org.jspecify.annotations.Nullable;

import neurogenesis.mc.proto.v1.RecipeInfo;

/**
 * All recipe-book entries ("recipe displays") of the running server, indexed by display id — the id the vanilla
 * recipe book uses when a player clicks a recipe. Rebuilt after datapack reloads.
 */
public final class RecipeIndex {
	public record Entry(RecipeDisplayEntry display, RecipeManager.ServerDisplayInfo info, RecipeInfo.Station station,
						RecipeInfo proto) {
	}

	private final List<Entry> entries;

	private RecipeIndex(List<Entry> entries) {
		this.entries = entries;
	}

	public List<Entry> entries() {
		return entries;
	}

	public @Nullable Entry get(int displayId) {
		return displayId >= 0 && displayId < entries.size() ? entries.get(displayId) : null;
	}

	public static RecipeIndex build(MinecraftServer server) {
		RecipeManager manager = server.getRecipeManager();
		var context = SlotDisplayContext.fromLevel(server.overworld());
		List<Entry> out = new ArrayList<>();
		for (int i = 0; ; i++) {
			RecipeManager.ServerDisplayInfo info = manager.getRecipeFromDisplay(new RecipeDisplayId(i));
			if (info == null) {
				break;
			}
			RecipeDisplayEntry display = info.display();
			RecipeDisplay d = display.display();
			int width = 0, height = 0;
			RecipeInfo.Station station;
			if (d instanceof ShapedCraftingRecipeDisplay shaped) {
				width = shaped.width();
				height = shaped.height();
				station = width <= 2 && height <= 2 ? RecipeInfo.Station.INVENTORY_2X2 : RecipeInfo.Station.CRAFTING_TABLE;
			} else if (d instanceof ShapelessCraftingRecipeDisplay shapeless) {
				station = shapeless.ingredients().size() <= 4 ? RecipeInfo.Station.INVENTORY_2X2 : RecipeInfo.Station.CRAFTING_TABLE;
			} else if (d instanceof FurnaceRecipeDisplay) {
				station = furnaceStation(display.category());
			} else {
				station = RecipeInfo.Station.OTHER;
			}
			List<ItemStack> results = display.resultItems(context);
			ItemStack result = results.isEmpty() ? ItemStack.EMPTY : results.getFirst();
			RecipeInfo proto = RecipeInfo.newBuilder()
				.setDisplayId(i)
				.setRecipeId(info.parent().id().identifier().toString())
				.setResultItem(result.isEmpty() ? 0 : BuiltInRegistries.ITEM.getId(result.getItem()))
				.setResultCount(result.getCount())
				.setStation(station)
				.setWidth(width)
				.setHeight(height)
				.build();
			out.add(new Entry(display, info, station, proto));
		}
		return new RecipeIndex(List.copyOf(out));
	}

	/** Whether a recipe of this station can be placed in a menu with the given recipe book type and grid size. */
	public static boolean fits(RecipeInfo.Station station, @Nullable RecipeBookType type, boolean largeGrid) {
		if (type == null) {
			return false;
		}
		return switch (station) {
			case INVENTORY_2X2 -> type == RecipeBookType.CRAFTING;
			case CRAFTING_TABLE -> type == RecipeBookType.CRAFTING && largeGrid;
			case FURNACE -> type == RecipeBookType.FURNACE;
			case BLAST_FURNACE -> type == RecipeBookType.BLAST_FURNACE;
			case SMOKER -> type == RecipeBookType.SMOKER;
			default -> false;
		};
	}

	private static RecipeInfo.Station furnaceStation(RecipeBookCategory category) {
		if (category == RecipeBookCategories.BLAST_FURNACE_BLOCKS || category == RecipeBookCategories.BLAST_FURNACE_MISC) {
			return RecipeInfo.Station.BLAST_FURNACE;
		}
		if (category == RecipeBookCategories.SMOKER_FOOD) {
			return RecipeInfo.Station.SMOKER;
		}
		if (category == RecipeBookCategories.FURNACE_FOOD || category == RecipeBookCategories.FURNACE_BLOCKS
			|| category == RecipeBookCategories.FURNACE_MISC) {
			return RecipeInfo.Station.FURNACE;
		}
		return RecipeInfo.Station.OTHER; // campfire
	}
}

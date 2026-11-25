package neurogenesis.mc.crafting;

import net.minecraft.advancements.AdvancementHolder;
import net.minecraft.advancements.AdvancementProgress;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.PlayerAdvancements;
import net.minecraft.server.level.ServerPlayer;

import neurogenesis.mc.proto.v1.RecipeUnlock;

/** Sets up an agent's recipe book at the start of an episode. */
public final class RecipePolicy {
	private RecipePolicy() {
	}

	public static void apply(ServerPlayer player, RecipeUnlock policy) {
		MinecraftServer server = player.level().getServer();
		var recipes = server.getRecipeManager().getRecipes();
		if (policy == RecipeUnlock.RECIPES_VANILLA) {
			// Forget everything, including the recipe advancements that would otherwise stay completed.
			PlayerAdvancements advancements = player.getAdvancements();
			for (AdvancementHolder holder : server.getAdvancements().getAllAdvancements()) {
				AdvancementProgress progress = advancements.getOrStartProgress(holder);
				for (String criterion : progress.getCompletedCriteria()) {
					advancements.revoke(holder, criterion);
				}
			}
			player.resetRecipes(recipes);
		} else {
			player.awardRecipes(recipes);
		}
	}
}

package neurogenesis.mc.sim;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;

/** Registry vocabularies sent once in Spec; observation ids index into these lists. */
public final class Vocab {
	private Vocab() {
	}

	public static List<String> items() {
		return names(BuiltInRegistries.ITEM);
	}

	public static List<String> blocks() {
		return names(BuiltInRegistries.BLOCK);
	}

	public static List<String> entities() {
		return names(BuiltInRegistries.ENTITY_TYPE);
	}

	public static List<String> menus() {
		return names(BuiltInRegistries.MENU);
	}

	private static <T> List<String> names(Registry<T> registry) {
		List<String> out = new ArrayList<>(registry.size());
		for (int id = 0; id < registry.size(); id++) {
			T value = registry.byId(id);
			Identifier key = value == null ? null : registry.getKey(value);
			out.add(key == null ? "" : key.toString());
		}
		return out;
	}
}

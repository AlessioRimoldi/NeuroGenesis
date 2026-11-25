"""Recipe lookup and container slot layouts, for building crafting actions.

Crafting the human way is: open a crafting screen, click a recipe in the recipe book (the server moves the
ingredients into the grid), then take the result. :func:`craft` builds that as a single step.
"""

from __future__ import annotations

from dataclasses import dataclass

from neurogenesis.mc.actions import action
from neurogenesis.mc.v1 import action_pb2, control_pb2

Station = control_pb2.RecipeInfo.Station


# Slot layouts of the vanilla menus (menu slot indices used by click_slot).
@dataclass(frozen=True)
class MenuLayout:
    result: int
    grid: range
    main: range      # inventory slots 9-35, in order
    hotbar: range    # inventory slots 0-8, in order


INVENTORY_MENU = MenuLayout(result=0, grid=range(1, 5), main=range(9, 36), hotbar=range(36, 45))
CRAFTING_TABLE_MENU = MenuLayout(result=0, grid=range(1, 10), main=range(10, 37), hotbar=range(37, 46))
FURNACE_MENU = MenuLayout(result=2, grid=range(0, 1), main=range(3, 30), hotbar=range(30, 39))
FURNACE_FUEL_SLOT = 1
INVENTORY_ARMOR = range(5, 9)    # in the inventory menu: head, chest, legs, feet
INVENTORY_OFFHAND = 45           # in the inventory menu


def inventory_to_menu_slot(index: int, menu: MenuLayout = INVENTORY_MENU) -> int:
    """Maps a player inventory index (0-8 hotbar, 9-35 main, as in AgentObs.inv_item) to a menu slot."""
    if 0 <= index < 9:
        return menu.hotbar[index]
    if 9 <= index < 36:
        return menu.main[index - 9]
    if menu is INVENTORY_MENU and index == 40:
        return INVENTORY_OFFHAND
    if menu is INVENTORY_MENU and 36 <= index < 40:
        return INVENTORY_ARMOR[3 - (index - 36)]  # inventory 36 = feet ... 39 = head
    raise ValueError(f"inventory slot {index} is not shown in this menu")


class RecipeBook:
    """Recipe display ids from the server's Spec, searchable by recipe id or result item."""

    def __init__(self, spec: control_pb2.Spec):
        self.recipes = list(spec.recipe_vocab)
        self._items = list(spec.item_vocab)
        self._by_recipe = {r.recipe_id: r for r in self.recipes}

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, display_id: int) -> control_pb2.RecipeInfo:
        return self.recipes[display_id]

    def find(self, name: str, station: int | None = None) -> control_pb2.RecipeInfo:
        """Finds a recipe by recipe id ("minecraft:oak_planks") or result item; prefers the recipe id."""
        if not name.startswith("minecraft:") and ":" not in name:
            name = f"minecraft:{name}"
        if name in self._by_recipe and (station is None or self._by_recipe[name].station == station):
            return self._by_recipe[name]
        matches = [
            r for r in self.recipes
            if self._items[r.result_item] == name and (station is None or r.station == station)
        ]
        if not matches:
            raise KeyError(f"no recipe for {name}")
        return matches[0]

    def result_name(self, recipe: control_pb2.RecipeInfo) -> str:
        return self._items[recipe.result_item]


def craft(
    agent_id: int,
    recipe: control_pb2.RecipeInfo | int,
    *,
    all: bool = False,
    open_inventory: bool = True,
) -> action_pb2.AgentAction:
    """One step that crafts ``recipe``: (open the inventory,) place it with the recipe book, take the result.

    With ``all=True`` the recipe is placed as many times as the inventory allows and the whole result is taken.
    For a crafting table or furnace, open its screen first (look at it and ``use``) and pass
    ``open_inventory=False``. Furnace results take time: place now, take the result (slot 2) later.
    """
    rid = recipe if isinstance(recipe, int) else recipe.display_id
    station = None if isinstance(recipe, int) else recipe.station
    take = [] if station in (Station.FURNACE, Station.BLAST_FURNACE, Station.SMOKER) else [(0, 0, "quick_move")]
    return action(
        agent_id,
        inventory="open" if open_inventory else None,
        recipes=[(rid, all)],
        clicks=take,
    )

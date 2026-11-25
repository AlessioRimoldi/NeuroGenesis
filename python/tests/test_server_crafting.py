"""Inventory and crafting through the recipe book (pytest -m server)."""

import pytest

from neurogenesis.mc.actions import action, agent, arena
from neurogenesis.mc.crafting import FURNACE_FUEL_SLOT, FURNACE_MENU, RecipeBook, Station, craft, inventory_to_menu_slot

pytestmark = pytest.mark.server

CENTER = (16.5, 64.0, 16.5)


def _event_total(obs, kind, item_id=None):
    from neurogenesis.mc.v1 import observation_pb2

    t = observation_pb2.Event.Type.Value(kind)
    return sum(e.amount for e in obs.events if e.type == t and (item_id is None or e.item_id == item_id))


def _count(spec, obs, name):
    item = list(spec.item_vocab).index(name)
    return sum(c for i, c in zip(obs.inv_item, obs.inv_count) if i == item)


def _slot(spec, obs, name):
    return list(obs.inv_item).index(list(spec.item_vocab).index(name))


def _place_and_open(client, menu_name):
    """Places the block held in hotbar slot 1 on the floor in front of the agent, then opens it."""
    client.step([action(0, camera=(0, 60))])                 # look at the floor ~1 block ahead
    r = client.step([action(0, ["use"])])                    # place
    blocks = list(client.spec.block_vocab)
    assert blocks[r.obs[0].crosshair.block_id] == menu_name.replace("minecraft:crafting", "minecraft:crafting_table")
    client.step([action(0)])                                 # release the key
    r = client.step([action(0, ["use"])])                    # open
    menus = list(client.spec.menu_vocab)
    assert r.obs[0].container.open and menus[r.obs[0].container.menu_type] == menu_name
    return r


@pytest.fixture
def book(client):
    return RecipeBook(client.spec)


def test_recipe_vocab(book):
    assert len(book) > 1000
    assert book.find("minecraft:oak_planks").station == Station.INVENTORY_2X2
    assert book.find("minecraft:wooden_pickaxe").station == Station.CRAFTING_TABLE
    assert book.find("minecraft:iron_ingot_from_smelting_raw_iron").station == Station.FURNACE


def test_recipe_book_in_inventory_grid(client, book):
    spec = client.spec
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER, inventory=[("minecraft:oak_log", 3, 0)])])
    r = client.step([craft(0, book.find("minecraft:oak_planks"), all=True)])
    planks = list(spec.item_vocab).index("minecraft:oak_planks")
    assert _event_total(r.obs[0], "CRAFTED", planks) == 12
    assert _count(spec, r.obs[0], "minecraft:oak_planks") == 12 and _count(spec, r.obs[0], "minecraft:oak_log") == 0
    client.step([craft(0, book.find("minecraft:crafting_table"))])
    r = client.step([craft(0, book.find("minecraft:stick"))])
    assert _count(spec, r.obs[0], "minecraft:crafting_table") == 1
    assert _count(spec, r.obs[0], "minecraft:stick") == 4
    assert _count(spec, r.obs[0], "minecraft:oak_planks") == 6
    r = client.step([action(0, inventory="close")])
    assert not r.obs[0].container.open


def test_crafting_table_and_tool_durability(client, book):
    spec = client.spec
    inv = [("minecraft:crafting_table", 1, 0), ("minecraft:oak_planks", 3, 1), ("minecraft:stick", 2, 2)]
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER, yaw=0, inventory=inv)], craftable=True)
    r = _place_and_open(client, "minecraft:crafting")
    pickaxe = book.find("minecraft:wooden_pickaxe")
    assert pickaxe.display_id in r.obs[0].craftable
    r = client.step([craft(0, pickaxe, open_inventory=False)])
    assert _count(spec, r.obs[0], "minecraft:wooden_pickaxe") == 1
    slot = _slot(spec, r.obs[0], "minecraft:wooden_pickaxe")
    assert r.obs[0].inv_durability[slot] == 1.0

    # Close the table, hold the pickaxe, mine the grass under the agent: the tool wears.
    client.step([action(0, inventory="close", camera=(0, 30))])
    if slot >= 9:
        client.step([action(0, inventory="open", clicks=[(inventory_to_menu_slot(slot), 0, "swap")])])
        client.step([action(0, inventory="close")])
        slot = 0
    r = client.step([action(0, hotbar=slot + 1)])
    for _ in range(60):
        r = client.step([action(0, ["attack"])])
        if _event_total(r.obs[0], "BLOCK_BROKEN"):
            break
    assert 0.0 < r.obs[0].inv_durability[slot] < 1.0


def test_craftable_mask(client, book):
    client.reset(
        seed=1, arena=arena(), agents=[agent("a", pos=CENTER, inventory=[("minecraft:oak_log", 1, 0)])], craftable=True
    )
    r = client.step([action(0)])
    mask = set(r.obs[0].craftable)
    assert book.find("minecraft:oak_planks").display_id in mask
    assert book.find("minecraft:crafting_table").display_id not in mask      # needs 4 planks
    r = client.step([craft(0, book.find("minecraft:oak_planks"))])
    mask = set(r.obs[0].craftable)
    assert book.find("minecraft:crafting_table").display_id in mask
    assert book.find("minecraft:stick").display_id in mask
    assert book.find("minecraft:wooden_pickaxe").display_id not in mask      # 3x3 only
    assert list(client.step([action(0)]).obs[0].craftable) == list(r.obs[0].craftable)  # cached, unchanged


def test_vanilla_recipe_progression(client, book):
    spec = client.spec
    torch = book.find("minecraft:torch")
    inv = [("minecraft:coal", 1, 0), ("minecraft:stick", 1, 1)]
    # A 1x1 arena whose only "food" is a stone pickaxe lying where the agent stands: picked up on the first tick.
    cfg = dict(seed=1, arena=arena((1, 1), food="minecraft:stone_pickaxe", food_count=1), settle_ticks=0, craftable=True)

    for episode in range(2):  # the second episode must start locked again
        r = client.reset(agents=[agent("a", pos=(0.5, 64, 0.5), recipes="vanilla", inventory=inv)], **cfg)
        assert torch.display_id not in r.obs[0].craftable   # has coal and a stick, but torches need a stone pickaxe
        r = client.step([craft(0, torch)])
        assert _event_total(r.obs[0], "PICKED_UP") == 1     # picked up the pickaxe on that tick ...
        assert _event_total(r.obs[0], "RECIPE_UNLOCKED") > 0, f"episode {episode}"  # ... which unlocked the torch
        assert _count(spec, r.obs[0], "minecraft:torch") == 0  # the placement happened before the unlock
        r = client.step([craft(0, torch)])
        assert _count(spec, r.obs[0], "minecraft:torch") == 4

    # Default policy: everything unlocked from the start, no unlock events during the episode.
    client.reset(agents=[agent("a", pos=(0.5, 64, 0.5), inventory=inv)], **cfg)
    r = client.step([craft(0, torch)])
    assert _count(spec, r.obs[0], "minecraft:torch") == 4
    assert _event_total(r.obs[0], "RECIPE_UNLOCKED") == 0


def test_furnace_smelting(client, book):
    spec = client.spec
    inv = [("minecraft:furnace", 1, 0), ("minecraft:raw_iron", 1, 1), ("minecraft:coal", 1, 2)]
    client.reset(seed=1, arena=arena(), agents=[agent("a", pos=CENTER, yaw=0, inventory=inv)])
    r = _place_and_open(client, "minecraft:furnace")
    recipe = book.find("minecraft:iron_ingot_from_smelting_raw_iron")
    coal = inventory_to_menu_slot(_slot(spec, r.obs[0], "minecraft:coal"), FURNACE_MENU)
    r = client.step([craft(0, recipe, open_inventory=False)])
    assert _count(spec, r.obs[0], "minecraft:raw_iron") == 0        # moved into the furnace input
    client.step([action(0, clicks=[(coal, 0), (FURNACE_FUEL_SLOT, 0)])])
    client.step(ticks=205)                                          # smelting takes 200 ticks
    r = client.step([action(0, clicks=[(FURNACE_MENU.result, 0, "quick_move")])])
    assert _count(spec, r.obs[0], "minecraft:iron_ingot") == 1

"""Helpers to build agent actions and episode configs without touching protobuf directly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from neurogenesis.mc.v1 import action_pb2, arena_pb2, common_pb2

#: Held keys, in a fixed order (handy for MultiBinary action spaces).
KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use")

INVENTORY = {
    None: action_pb2.INVENTORY_NONE,
    "open": action_pb2.INVENTORY_OPEN,
    "close": action_pb2.INVENTORY_CLOSE,
    "toggle": action_pb2.INVENTORY_TOGGLE,
}

CLICK_MODES = {
    "pickup": action_pb2.CLICK_PICKUP,
    "quick_move": action_pb2.CLICK_QUICK_MOVE,
    "swap": action_pb2.CLICK_SWAP,
    "clone": action_pb2.CLICK_CLONE,
    "throw": action_pb2.CLICK_THROW,
    "quick_craft": action_pb2.CLICK_QUICK_CRAFT,
    "pickup_all": action_pb2.CLICK_PICKUP_ALL,
}


def action(
    agent_id: int,
    keys: Iterable[str] | Mapping[str, bool] = (),
    *,
    camera: tuple[float, float] = (0.0, 0.0),
    hotbar: int = 0,
    drop: bool = False,
    drop_stack: bool = False,
    swap_hands: bool = False,
    inventory: str | None = None,
    clicks: Iterable[tuple[int, int] | tuple[int, int, str]] = (),
    recipes: Iterable[int | tuple[int, bool]] = (),
) -> action_pb2.AgentAction:
    """Builds an AgentAction.

    ``keys`` are held for every tick of the step (names from :data:`KEYS`); ``camera`` is (dyaw, dpitch) in
    degrees; ``clicks`` are (slot, button[, mode]) applied in the open container screen; ``recipes`` are
    recipe display ids (or (id, max) pairs) placed with the recipe book before the clicks.
    """
    held = {k for k, v in keys.items() if v} if isinstance(keys, Mapping) else set(keys)
    unknown = held - set(KEYS)
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)}; valid keys: {KEYS}")
    a = action_pb2.AgentAction(
        agent_id=agent_id,
        camera_dyaw=camera[0],
        camera_dpitch=camera[1],
        hotbar=hotbar,
        drop=drop,
        drop_stack=drop_stack,
        swap_hands=swap_hands,
        inventory=INVENTORY[inventory],
        **{k: True for k in held},
    )
    for click in clicks:
        slot, button, *mode = click
        a.clicks.add(slot=slot, button=button, mode=CLICK_MODES[mode[0] if mode else "pickup"])
    for recipe in recipes:
        rid, use_max = (recipe, False) if isinstance(recipe, int) else recipe
        a.recipes.add(recipe=rid, max=use_max)
    return a


def agent(
    name: str,
    *,
    pos: tuple[float, float, float] | None = None,
    yaw: float | None = None,
    health: float | None = None,
    food: int | None = None,
    inventory: Iterable[tuple[str, int] | tuple[str, int, int]] = (),
    respawn_on_death: bool = False,
    game_mode: str = "survival",
    recipes: str = "all",
) -> arena_pb2.AgentConfig:
    """Builds an AgentConfig. ``inventory`` entries are (item_id, count[, slot]).

    ``recipes`` is the recipe-book policy: ``"all"`` (every recipe unlocked each episode) or ``"vanilla"``
    (none at start; unlocked by vanilla advancements as ingredients are obtained).
    """
    cfg = arena_pb2.AgentConfig(
        name=name,
        respawn_on_death=respawn_on_death,
        game_mode=arena_pb2.GameMode.Value(f"GAME_MODE_{game_mode.upper()}"),
        recipes=arena_pb2.RecipeUnlock.Value(f"RECIPES_{recipes.upper()}"),
    )
    if pos is not None:
        cfg.spawn_pos.CopyFrom(common_pb2.Vec3d(x=pos[0], y=pos[1], z=pos[2]))
    if yaw is not None:
        cfg.spawn_yaw = yaw
    if health is not None:
        cfg.health = health
    if food is not None:
        cfg.food = food
    for item in inventory:
        item_id, count, *slot = item
        cfg.inventory.add(id=item_id, count=count, slot=slot[0] if slot else -1)
    return cfg


def arena(
    size: tuple[int, int] = (32, 32),
    *,
    wall_height: int = 3,
    floor: str = "",
    wall: str = "",
    food: str = "minecraft:apple",
    food_count: int = 0,
    food_blocks: bool = False,
    food_min_dist: float = 0.0,
    food_respawn: bool = False,
    food_respawn_delay: int = 0,
    smell: bool = False,
    **smell_params: float,
) -> arena_pb2.ArenaConfig:
    """Builds an ArenaConfig; ``smell_params`` are SmellConfig fields (diffusion, decay, ...)."""
    cfg = arena_pb2.ArenaConfig(
        size_x=size[0], size_z=size[1], wall_height=wall_height, floor_block=floor, wall_block=wall
    )
    cfg.food.CopyFrom(
        arena_pb2.FoodConfig(
            kind=arena_pb2.FoodConfig.KIND_BLOCK if food_blocks else arena_pb2.FoodConfig.KIND_ITEM,
            id=food,
            count=food_count,
            min_dist=food_min_dist,
            respawn=food_respawn,
            respawn_delay=food_respawn_delay,
        )
    )
    cfg.smell.CopyFrom(arena_pb2.SmellConfig(enabled=smell, **smell_params))
    return cfg

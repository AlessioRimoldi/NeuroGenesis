from neurogenesis.mc.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GameMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GAME_MODE_SURVIVAL: _ClassVar[GameMode]
    GAME_MODE_ADVENTURE: _ClassVar[GameMode]
    GAME_MODE_CREATIVE: _ClassVar[GameMode]
    GAME_MODE_SPECTATOR: _ClassVar[GameMode]
GAME_MODE_SURVIVAL: GameMode
GAME_MODE_ADVENTURE: GameMode
GAME_MODE_CREATIVE: GameMode
GAME_MODE_SPECTATOR: GameMode

class ArenaConfig(_message.Message):
    __slots__ = ("size_x", "size_z", "wall_height", "floor_block", "wall_block", "food", "smell")
    SIZE_X_FIELD_NUMBER: _ClassVar[int]
    SIZE_Z_FIELD_NUMBER: _ClassVar[int]
    WALL_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FLOOR_BLOCK_FIELD_NUMBER: _ClassVar[int]
    WALL_BLOCK_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    SMELL_FIELD_NUMBER: _ClassVar[int]
    size_x: int
    size_z: int
    wall_height: int
    floor_block: str
    wall_block: str
    food: FoodConfig
    smell: SmellConfig
    def __init__(self, size_x: _Optional[int] = ..., size_z: _Optional[int] = ..., wall_height: _Optional[int] = ..., floor_block: _Optional[str] = ..., wall_block: _Optional[str] = ..., food: _Optional[_Union[FoodConfig, _Mapping]] = ..., smell: _Optional[_Union[SmellConfig, _Mapping]] = ...) -> None: ...

class FoodConfig(_message.Message):
    __slots__ = ("kind", "id", "count", "min_dist", "respawn", "respawn_delay")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_ITEM: _ClassVar[FoodConfig.Kind]
        KIND_BLOCK: _ClassVar[FoodConfig.Kind]
    KIND_ITEM: FoodConfig.Kind
    KIND_BLOCK: FoodConfig.Kind
    KIND_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    MIN_DIST_FIELD_NUMBER: _ClassVar[int]
    RESPAWN_FIELD_NUMBER: _ClassVar[int]
    RESPAWN_DELAY_FIELD_NUMBER: _ClassVar[int]
    kind: FoodConfig.Kind
    id: str
    count: int
    min_dist: float
    respawn: bool
    respawn_delay: int
    def __init__(self, kind: _Optional[_Union[FoodConfig.Kind, str]] = ..., id: _Optional[str] = ..., count: _Optional[int] = ..., min_dist: _Optional[float] = ..., respawn: _Optional[bool] = ..., respawn_delay: _Optional[int] = ...) -> None: ...

class SmellConfig(_message.Message):
    __slots__ = ("enabled", "cell_size", "diffusion", "decay", "emission", "prewarm_ticks", "antenna_lateral", "antenna_forward")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    CELL_SIZE_FIELD_NUMBER: _ClassVar[int]
    DIFFUSION_FIELD_NUMBER: _ClassVar[int]
    DECAY_FIELD_NUMBER: _ClassVar[int]
    EMISSION_FIELD_NUMBER: _ClassVar[int]
    PREWARM_TICKS_FIELD_NUMBER: _ClassVar[int]
    ANTENNA_LATERAL_FIELD_NUMBER: _ClassVar[int]
    ANTENNA_FORWARD_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    cell_size: float
    diffusion: float
    decay: float
    emission: float
    prewarm_ticks: int
    antenna_lateral: float
    antenna_forward: float
    def __init__(self, enabled: _Optional[bool] = ..., cell_size: _Optional[float] = ..., diffusion: _Optional[float] = ..., decay: _Optional[float] = ..., emission: _Optional[float] = ..., prewarm_ticks: _Optional[int] = ..., antenna_lateral: _Optional[float] = ..., antenna_forward: _Optional[float] = ...) -> None: ...

class AgentConfig(_message.Message):
    __slots__ = ("name", "spawn_pos", "spawn_yaw", "game_mode", "health", "food", "inventory", "respawn_on_death")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SPAWN_POS_FIELD_NUMBER: _ClassVar[int]
    SPAWN_YAW_FIELD_NUMBER: _ClassVar[int]
    GAME_MODE_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    INVENTORY_FIELD_NUMBER: _ClassVar[int]
    RESPAWN_ON_DEATH_FIELD_NUMBER: _ClassVar[int]
    name: str
    spawn_pos: _common_pb2.Vec3d
    spawn_yaw: float
    game_mode: GameMode
    health: float
    food: int
    inventory: _containers.RepeatedCompositeFieldContainer[_common_pb2.ItemStack]
    respawn_on_death: bool
    def __init__(self, name: _Optional[str] = ..., spawn_pos: _Optional[_Union[_common_pb2.Vec3d, _Mapping]] = ..., spawn_yaw: _Optional[float] = ..., game_mode: _Optional[_Union[GameMode, str]] = ..., health: _Optional[float] = ..., food: _Optional[int] = ..., inventory: _Optional[_Iterable[_Union[_common_pb2.ItemStack, _Mapping]]] = ..., respawn_on_death: _Optional[bool] = ...) -> None: ...

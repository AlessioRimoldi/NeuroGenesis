from neurogenesis.mc.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentObs(_message.Message):
    __slots__ = ("agent_id", "name", "alive", "terminated", "truncated", "pos", "vel", "yaw", "pitch", "on_ground", "in_water", "crouching", "sprinting", "using_item", "health", "food", "saturation", "air", "inv_item", "inv_count", "selected_slot", "container", "crosshair", "smell", "events", "craftable", "inv_durability")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIVE_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    POS_FIELD_NUMBER: _ClassVar[int]
    VEL_FIELD_NUMBER: _ClassVar[int]
    YAW_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    ON_GROUND_FIELD_NUMBER: _ClassVar[int]
    IN_WATER_FIELD_NUMBER: _ClassVar[int]
    CROUCHING_FIELD_NUMBER: _ClassVar[int]
    SPRINTING_FIELD_NUMBER: _ClassVar[int]
    USING_ITEM_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    SATURATION_FIELD_NUMBER: _ClassVar[int]
    AIR_FIELD_NUMBER: _ClassVar[int]
    INV_ITEM_FIELD_NUMBER: _ClassVar[int]
    INV_COUNT_FIELD_NUMBER: _ClassVar[int]
    SELECTED_SLOT_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    CROSSHAIR_FIELD_NUMBER: _ClassVar[int]
    SMELL_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    CRAFTABLE_FIELD_NUMBER: _ClassVar[int]
    INV_DURABILITY_FIELD_NUMBER: _ClassVar[int]
    agent_id: int
    name: str
    alive: bool
    terminated: bool
    truncated: bool
    pos: _common_pb2.Vec3d
    vel: _common_pb2.Vec3d
    yaw: float
    pitch: float
    on_ground: bool
    in_water: bool
    crouching: bool
    sprinting: bool
    using_item: bool
    health: float
    food: int
    saturation: float
    air: int
    inv_item: _containers.RepeatedScalarFieldContainer[int]
    inv_count: _containers.RepeatedScalarFieldContainer[int]
    selected_slot: int
    container: ContainerView
    crosshair: Crosshair
    smell: Smell
    events: _containers.RepeatedCompositeFieldContainer[Event]
    craftable: _containers.RepeatedScalarFieldContainer[int]
    inv_durability: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_id: _Optional[int] = ..., name: _Optional[str] = ..., alive: _Optional[bool] = ..., terminated: _Optional[bool] = ..., truncated: _Optional[bool] = ..., pos: _Optional[_Union[_common_pb2.Vec3d, _Mapping]] = ..., vel: _Optional[_Union[_common_pb2.Vec3d, _Mapping]] = ..., yaw: _Optional[float] = ..., pitch: _Optional[float] = ..., on_ground: _Optional[bool] = ..., in_water: _Optional[bool] = ..., crouching: _Optional[bool] = ..., sprinting: _Optional[bool] = ..., using_item: _Optional[bool] = ..., health: _Optional[float] = ..., food: _Optional[int] = ..., saturation: _Optional[float] = ..., air: _Optional[int] = ..., inv_item: _Optional[_Iterable[int]] = ..., inv_count: _Optional[_Iterable[int]] = ..., selected_slot: _Optional[int] = ..., container: _Optional[_Union[ContainerView, _Mapping]] = ..., crosshair: _Optional[_Union[Crosshair, _Mapping]] = ..., smell: _Optional[_Union[Smell, _Mapping]] = ..., events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., craftable: _Optional[_Iterable[int]] = ..., inv_durability: _Optional[_Iterable[float]] = ...) -> None: ...

class ContainerView(_message.Message):
    __slots__ = ("open", "menu_type", "slot_item", "slot_count", "carried_item", "carried_count")
    OPEN_FIELD_NUMBER: _ClassVar[int]
    MENU_TYPE_FIELD_NUMBER: _ClassVar[int]
    SLOT_ITEM_FIELD_NUMBER: _ClassVar[int]
    SLOT_COUNT_FIELD_NUMBER: _ClassVar[int]
    CARRIED_ITEM_FIELD_NUMBER: _ClassVar[int]
    CARRIED_COUNT_FIELD_NUMBER: _ClassVar[int]
    open: bool
    menu_type: int
    slot_item: _containers.RepeatedScalarFieldContainer[int]
    slot_count: _containers.RepeatedScalarFieldContainer[int]
    carried_item: int
    carried_count: int
    def __init__(self, open: _Optional[bool] = ..., menu_type: _Optional[int] = ..., slot_item: _Optional[_Iterable[int]] = ..., slot_count: _Optional[_Iterable[int]] = ..., carried_item: _Optional[int] = ..., carried_count: _Optional[int] = ...) -> None: ...

class Crosshair(_message.Message):
    __slots__ = ("type", "block_id", "block_pos", "face", "entity_type", "entity_id", "distance")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        MISS: _ClassVar[Crosshair.Type]
        BLOCK: _ClassVar[Crosshair.Type]
        ENTITY: _ClassVar[Crosshair.Type]
    MISS: Crosshair.Type
    BLOCK: Crosshair.Type
    ENTITY: Crosshair.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_POS_FIELD_NUMBER: _ClassVar[int]
    FACE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    type: Crosshair.Type
    block_id: int
    block_pos: _common_pb2.Vec3i
    face: int
    entity_type: int
    entity_id: int
    distance: float
    def __init__(self, type: _Optional[_Union[Crosshair.Type, str]] = ..., block_id: _Optional[int] = ..., block_pos: _Optional[_Union[_common_pb2.Vec3i, _Mapping]] = ..., face: _Optional[int] = ..., entity_type: _Optional[int] = ..., entity_id: _Optional[int] = ..., distance: _Optional[float] = ...) -> None: ...

class Smell(_message.Message):
    __slots__ = ("left", "right")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: float
    right: float
    def __init__(self, left: _Optional[float] = ..., right: _Optional[float] = ...) -> None: ...

class Event(_message.Message):
    __slots__ = ("type", "item_id", "amount")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[Event.Type]
        DAMAGE: _ClassVar[Event.Type]
        DEATH: _ClassVar[Event.Type]
        CONSUMED: _ClassVar[Event.Type]
        PICKED_UP: _ClassVar[Event.Type]
        BLOCK_BROKEN: _ClassVar[Event.Type]
        CRAFTED: _ClassVar[Event.Type]
        ATTACKED: _ClassVar[Event.Type]
        RECIPE_UNLOCKED: _ClassVar[Event.Type]
    UNKNOWN: Event.Type
    DAMAGE: Event.Type
    DEATH: Event.Type
    CONSUMED: Event.Type
    PICKED_UP: Event.Type
    BLOCK_BROKEN: Event.Type
    CRAFTED: Event.Type
    ATTACKED: Event.Type
    RECIPE_UNLOCKED: Event.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    type: Event.Type
    item_id: int
    amount: float
    def __init__(self, type: _Optional[_Union[Event.Type, str]] = ..., item_id: _Optional[int] = ..., amount: _Optional[float] = ...) -> None: ...

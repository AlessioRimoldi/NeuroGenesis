from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InventoryAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INVENTORY_NONE: _ClassVar[InventoryAction]
    INVENTORY_OPEN: _ClassVar[InventoryAction]
    INVENTORY_CLOSE: _ClassVar[InventoryAction]
    INVENTORY_TOGGLE: _ClassVar[InventoryAction]

class ClickMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLICK_PICKUP: _ClassVar[ClickMode]
    CLICK_QUICK_MOVE: _ClassVar[ClickMode]
    CLICK_SWAP: _ClassVar[ClickMode]
    CLICK_CLONE: _ClassVar[ClickMode]
    CLICK_THROW: _ClassVar[ClickMode]
    CLICK_QUICK_CRAFT: _ClassVar[ClickMode]
    CLICK_PICKUP_ALL: _ClassVar[ClickMode]
INVENTORY_NONE: InventoryAction
INVENTORY_OPEN: InventoryAction
INVENTORY_CLOSE: InventoryAction
INVENTORY_TOGGLE: InventoryAction
CLICK_PICKUP: ClickMode
CLICK_QUICK_MOVE: ClickMode
CLICK_SWAP: ClickMode
CLICK_CLONE: ClickMode
CLICK_THROW: ClickMode
CLICK_QUICK_CRAFT: ClickMode
CLICK_PICKUP_ALL: ClickMode

class AgentAction(_message.Message):
    __slots__ = ("agent_id", "forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "camera_dyaw", "camera_dpitch", "hotbar", "drop", "drop_stack", "swap_hands", "inventory", "clicks")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    FORWARD_FIELD_NUMBER: _ClassVar[int]
    BACK_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    JUMP_FIELD_NUMBER: _ClassVar[int]
    SNEAK_FIELD_NUMBER: _ClassVar[int]
    SPRINT_FIELD_NUMBER: _ClassVar[int]
    ATTACK_FIELD_NUMBER: _ClassVar[int]
    USE_FIELD_NUMBER: _ClassVar[int]
    CAMERA_DYAW_FIELD_NUMBER: _ClassVar[int]
    CAMERA_DPITCH_FIELD_NUMBER: _ClassVar[int]
    HOTBAR_FIELD_NUMBER: _ClassVar[int]
    DROP_FIELD_NUMBER: _ClassVar[int]
    DROP_STACK_FIELD_NUMBER: _ClassVar[int]
    SWAP_HANDS_FIELD_NUMBER: _ClassVar[int]
    INVENTORY_FIELD_NUMBER: _ClassVar[int]
    CLICKS_FIELD_NUMBER: _ClassVar[int]
    agent_id: int
    forward: bool
    back: bool
    left: bool
    right: bool
    jump: bool
    sneak: bool
    sprint: bool
    attack: bool
    use: bool
    camera_dyaw: float
    camera_dpitch: float
    hotbar: int
    drop: bool
    drop_stack: bool
    swap_hands: bool
    inventory: InventoryAction
    clicks: _containers.RepeatedCompositeFieldContainer[ClickSlot]
    def __init__(self, agent_id: _Optional[int] = ..., forward: _Optional[bool] = ..., back: _Optional[bool] = ..., left: _Optional[bool] = ..., right: _Optional[bool] = ..., jump: _Optional[bool] = ..., sneak: _Optional[bool] = ..., sprint: _Optional[bool] = ..., attack: _Optional[bool] = ..., use: _Optional[bool] = ..., camera_dyaw: _Optional[float] = ..., camera_dpitch: _Optional[float] = ..., hotbar: _Optional[int] = ..., drop: _Optional[bool] = ..., drop_stack: _Optional[bool] = ..., swap_hands: _Optional[bool] = ..., inventory: _Optional[_Union[InventoryAction, str]] = ..., clicks: _Optional[_Iterable[_Union[ClickSlot, _Mapping]]] = ...) -> None: ...

class ClickSlot(_message.Message):
    __slots__ = ("slot", "button", "mode")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    BUTTON_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    slot: int
    button: int
    mode: ClickMode
    def __init__(self, slot: _Optional[int] = ..., button: _Optional[int] = ..., mode: _Optional[_Union[ClickMode, str]] = ...) -> None: ...

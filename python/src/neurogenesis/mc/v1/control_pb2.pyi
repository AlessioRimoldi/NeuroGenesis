from neurogenesis.mc.v1 import action_pb2 as _action_pb2
from neurogenesis.mc.v1 import arena_pb2 as _arena_pb2
from neurogenesis.mc.v1 import observation_pb2 as _observation_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Request(_message.Message):
    __slots__ = ("request_id", "hello", "reset", "step", "close")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    HELLO_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    hello: Hello
    reset: Reset
    step: Step
    close: Close
    def __init__(self, request_id: _Optional[int] = ..., hello: _Optional[_Union[Hello, _Mapping]] = ..., reset: _Optional[_Union[Reset, _Mapping]] = ..., step: _Optional[_Union[Step, _Mapping]] = ..., close: _Optional[_Union[Close, _Mapping]] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("request_id", "spec", "step_result", "error", "closed")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STEP_RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CLOSED_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    spec: Spec
    step_result: StepResult
    error: Error
    closed: Closed
    def __init__(self, request_id: _Optional[int] = ..., spec: _Optional[_Union[Spec, _Mapping]] = ..., step_result: _Optional[_Union[StepResult, _Mapping]] = ..., error: _Optional[_Union[Error, _Mapping]] = ..., closed: _Optional[_Union[Closed, _Mapping]] = ...) -> None: ...

class Hello(_message.Message):
    __slots__ = ("protocol_version", "client_name")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    CLIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    client_name: str
    def __init__(self, protocol_version: _Optional[int] = ..., client_name: _Optional[str] = ...) -> None: ...

class Spec(_message.Message):
    __slots__ = ("protocol_version", "minecraft_version", "mod_version", "server_tick", "max_agents", "item_vocab", "block_vocab", "entity_vocab", "menu_vocab")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MINECRAFT_VERSION_FIELD_NUMBER: _ClassVar[int]
    MOD_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVER_TICK_FIELD_NUMBER: _ClassVar[int]
    MAX_AGENTS_FIELD_NUMBER: _ClassVar[int]
    ITEM_VOCAB_FIELD_NUMBER: _ClassVar[int]
    BLOCK_VOCAB_FIELD_NUMBER: _ClassVar[int]
    ENTITY_VOCAB_FIELD_NUMBER: _ClassVar[int]
    MENU_VOCAB_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    minecraft_version: str
    mod_version: str
    server_tick: int
    max_agents: int
    item_vocab: _containers.RepeatedScalarFieldContainer[str]
    block_vocab: _containers.RepeatedScalarFieldContainer[str]
    entity_vocab: _containers.RepeatedScalarFieldContainer[str]
    menu_vocab: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, protocol_version: _Optional[int] = ..., minecraft_version: _Optional[str] = ..., mod_version: _Optional[str] = ..., server_tick: _Optional[int] = ..., max_agents: _Optional[int] = ..., item_vocab: _Optional[_Iterable[str]] = ..., block_vocab: _Optional[_Iterable[str]] = ..., entity_vocab: _Optional[_Iterable[str]] = ..., menu_vocab: _Optional[_Iterable[str]] = ...) -> None: ...

class Reset(_message.Message):
    __slots__ = ("seed", "arena", "agents", "settle_ticks", "max_episode_ticks")
    SEED_FIELD_NUMBER: _ClassVar[int]
    ARENA_FIELD_NUMBER: _ClassVar[int]
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    SETTLE_TICKS_FIELD_NUMBER: _ClassVar[int]
    MAX_EPISODE_TICKS_FIELD_NUMBER: _ClassVar[int]
    seed: int
    arena: _arena_pb2.ArenaConfig
    agents: _containers.RepeatedCompositeFieldContainer[_arena_pb2.AgentConfig]
    settle_ticks: int
    max_episode_ticks: int
    def __init__(self, seed: _Optional[int] = ..., arena: _Optional[_Union[_arena_pb2.ArenaConfig, _Mapping]] = ..., agents: _Optional[_Iterable[_Union[_arena_pb2.AgentConfig, _Mapping]]] = ..., settle_ticks: _Optional[int] = ..., max_episode_ticks: _Optional[int] = ...) -> None: ...

class Step(_message.Message):
    __slots__ = ("ticks", "actions")
    TICKS_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    ticks: int
    actions: _containers.RepeatedCompositeFieldContainer[_action_pb2.AgentAction]
    def __init__(self, ticks: _Optional[int] = ..., actions: _Optional[_Iterable[_Union[_action_pb2.AgentAction, _Mapping]]] = ...) -> None: ...

class StepResult(_message.Message):
    __slots__ = ("tick", "episode_tick", "obs", "server_step_us")
    TICK_FIELD_NUMBER: _ClassVar[int]
    EPISODE_TICK_FIELD_NUMBER: _ClassVar[int]
    OBS_FIELD_NUMBER: _ClassVar[int]
    SERVER_STEP_US_FIELD_NUMBER: _ClassVar[int]
    tick: int
    episode_tick: int
    obs: _containers.RepeatedCompositeFieldContainer[_observation_pb2.AgentObs]
    server_step_us: int
    def __init__(self, tick: _Optional[int] = ..., episode_tick: _Optional[int] = ..., obs: _Optional[_Iterable[_Union[_observation_pb2.AgentObs, _Mapping]]] = ..., server_step_us: _Optional[int] = ...) -> None: ...

class Close(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Closed(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Error(_message.Message):
    __slots__ = ("code", "message")
    class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[Error.Code]
        BAD_REQUEST: _ClassVar[Error.Code]
        NOT_READY: _ClassVar[Error.Code]
        PROTOCOL_MISMATCH: _ClassVar[Error.Code]
        INTERNAL: _ClassVar[Error.Code]
    UNKNOWN: Error.Code
    BAD_REQUEST: Error.Code
    NOT_READY: Error.Code
    PROTOCOL_MISMATCH: Error.Code
    INTERNAL: Error.Code
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: Error.Code
    message: str
    def __init__(self, code: _Optional[_Union[Error.Code, str]] = ..., message: _Optional[str] = ...) -> None: ...

from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DispositionGrid(_message.Message):
    __slots__ = ("gear_code", "species_code", "disposition_cells")
    GEAR_CODE_FIELD_NUMBER: _ClassVar[int]
    SPECIES_CODE_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_CELLS_FIELD_NUMBER: _ClassVar[int]
    gear_code: str
    species_code: str
    disposition_cells: _containers.RepeatedCompositeFieldContainer[DispositionCell]
    def __init__(self, gear_code: _Optional[str] = ..., species_code: _Optional[str] = ..., disposition_cells: _Optional[_Iterable[_Union[DispositionCell, _Mapping]]] = ...) -> None: ...

class DispositionCell(_message.Message):
    __slots__ = ("longitude", "latitude", "gross_catch", "live_discards", "dead_discards")
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    GROSS_CATCH_FIELD_NUMBER: _ClassVar[int]
    LIVE_DISCARDS_FIELD_NUMBER: _ClassVar[int]
    DEAD_DISCARDS_FIELD_NUMBER: _ClassVar[int]
    longitude: float
    latitude: float
    gross_catch: float
    live_discards: float
    dead_discards: float
    def __init__(self, longitude: _Optional[float] = ..., latitude: _Optional[float] = ..., gross_catch: _Optional[float] = ..., live_discards: _Optional[float] = ..., dead_discards: _Optional[float] = ...) -> None: ...

class CatchDispositionSummary(_message.Message):
    __slots__ = ("measurement_unit", "disposition_grids")
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_GRIDS_FIELD_NUMBER: _ClassVar[int]
    measurement_unit: str
    disposition_grids: _containers.RepeatedCompositeFieldContainer[DispositionGrid]
    def __init__(self, measurement_unit: _Optional[str] = ..., disposition_grids: _Optional[_Iterable[_Union[DispositionGrid, _Mapping]]] = ...) -> None: ...

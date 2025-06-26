from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BiomassGrid(_message.Message):
    __slots__ = ("species_code", "biomass_cells")
    SPECIES_CODE_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_CELLS_FIELD_NUMBER: _ClassVar[int]
    species_code: str
    biomass_cells: _containers.RepeatedCompositeFieldContainer[BiomassCell]
    def __init__(self, species_code: _Optional[str] = ..., biomass_cells: _Optional[_Iterable[_Union[BiomassCell, _Mapping]]] = ...) -> None: ...

class BiomassCell(_message.Message):
    __slots__ = ("longitude", "latitude", "biomass")
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_FIELD_NUMBER: _ClassVar[int]
    longitude: float
    latitude: float
    biomass: float
    def __init__(self, longitude: _Optional[float] = ..., latitude: _Optional[float] = ..., biomass: _Optional[float] = ...) -> None: ...

class BiomassSummary(_message.Message):
    __slots__ = ("measurement_unit", "biomass_grids")
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_GRIDS_FIELD_NUMBER: _ClassVar[int]
    measurement_unit: str
    biomass_grids: _containers.RepeatedCompositeFieldContainer[BiomassGrid]
    def __init__(self, measurement_unit: _Optional[str] = ..., biomass_grids: _Optional[_Iterable[_Union[BiomassGrid, _Mapping]]] = ...) -> None: ...

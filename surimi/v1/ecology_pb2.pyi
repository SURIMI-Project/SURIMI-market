from surimi.v1 import biomass_pb2 as _biomass_pb2
from surimi.v1 import disposition_pb2 as _disposition_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetBiomassRequest(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class GetBiomassResponse(_message.Message):
    __slots__ = ("measurement_unit", "biomass_grids")
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_GRIDS_FIELD_NUMBER: _ClassVar[int]
    measurement_unit: str
    biomass_grids: _containers.RepeatedCompositeFieldContainer[_biomass_pb2.BiomassGrid]
    def __init__(self, measurement_unit: _Optional[str] = ..., biomass_grids: _Optional[_Iterable[_Union[_biomass_pb2.BiomassGrid, _Mapping]]] = ...) -> None: ...

class UpdateCatchDispositionSummaryRequest(_message.Message):
    __slots__ = ("simulation_id", "measurement_unit", "disposition_grids")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_GRIDS_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    measurement_unit: str
    disposition_grids: _containers.RepeatedCompositeFieldContainer[_disposition_pb2.DispositionGrid]
    def __init__(self, simulation_id: _Optional[str] = ..., measurement_unit: _Optional[str] = ..., disposition_grids: _Optional[_Iterable[_Union[_disposition_pb2.DispositionGrid, _Mapping]]] = ...) -> None: ...

class UpdateCatchDispositionSummaryResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

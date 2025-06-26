from surimi.v1 import biomass_pb2 as _biomass_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetBiomassRequest(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class GetBiomassResponse(_message.Message):
    __slots__ = ("simulation_id", "biomass_summary")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    biomass_summary: _biomass_pb2.BiomassSummary
    def __init__(self, simulation_id: _Optional[str] = ..., biomass_summary: _Optional[_Union[_biomass_pb2.BiomassSummary, _Mapping]] = ...) -> None: ...

class UpdateBiomassRequest(_message.Message):
    __slots__ = ("simulation_id", "biomass_summary")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    biomass_summary: _biomass_pb2.BiomassSummary
    def __init__(self, simulation_id: _Optional[str] = ..., biomass_summary: _Optional[_Union[_biomass_pb2.BiomassSummary, _Mapping]] = ...) -> None: ...

class UpdateBiomassResponse(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from surimi.v1 import biomass_pb2 as _biomass_pb2
from surimi.v1 import species_price_pb2 as _species_price_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InitRequest(_message.Message):
    __slots__ = ("simulation_id", "scenario_id", "start_date_time", "step_size")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_ID_FIELD_NUMBER: _ClassVar[int]
    START_DATE_TIME_FIELD_NUMBER: _ClassVar[int]
    STEP_SIZE_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    scenario_id: str
    start_date_time: _timestamp_pb2.Timestamp
    step_size: str
    def __init__(self, simulation_id: _Optional[str] = ..., scenario_id: _Optional[str] = ..., start_date_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., step_size: _Optional[str] = ...) -> None: ...

class InitResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdatePricesRequest(_message.Message):
    __slots__ = ("simulation_id", "prices")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    PRICES_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    prices: _containers.RepeatedCompositeFieldContainer[_species_price_pb2.SpeciesPrice]
    def __init__(self, simulation_id: _Optional[str] = ..., prices: _Optional[_Iterable[_Union[_species_price_pb2.SpeciesPrice, _Mapping]]] = ...) -> None: ...

class UpdatePricesResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SimulateStepRequest(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class SimulateStepResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdateBiomassRequest(_message.Message):
    __slots__ = ("simulation_id", "measurement_unit", "biomass_grids")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    BIOMASS_GRIDS_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    measurement_unit: str
    biomass_grids: _containers.RepeatedCompositeFieldContainer[_biomass_pb2.BiomassGrid]
    def __init__(self, simulation_id: _Optional[str] = ..., measurement_unit: _Optional[str] = ..., biomass_grids: _Optional[_Iterable[_Union[_biomass_pb2.BiomassGrid, _Mapping]]] = ...) -> None: ...

class UpdateBiomassResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

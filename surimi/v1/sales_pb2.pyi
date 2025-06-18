from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SalesSummary(_message.Message):
    __slots__ = ("market_code", "measurement_unit", "currency", "sales")
    MARKET_CODE_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    CURRENCY_FIELD_NUMBER: _ClassVar[int]
    SALES_FIELD_NUMBER: _ClassVar[int]
    market_code: str
    measurement_unit: str
    currency: str
    sales: _containers.RepeatedCompositeFieldContainer[Sale]
    def __init__(self, market_code: _Optional[str] = ..., measurement_unit: _Optional[str] = ..., currency: _Optional[str] = ..., sales: _Optional[_Iterable[_Union[Sale, _Mapping]]] = ...) -> None: ...

class Sale(_message.Message):
    __slots__ = ("species_code", "quantity", "value")
    SPECIES_CODE_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    species_code: str
    quantity: float
    value: float
    def __init__(self, species_code: _Optional[str] = ..., quantity: _Optional[float] = ..., value: _Optional[float] = ...) -> None: ...

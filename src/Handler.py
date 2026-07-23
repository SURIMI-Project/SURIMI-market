import abc
import datetime
from surimi.v1 import (
    finalise_simulation_pb2,
    initialise_simulation_pb2,
    get_species_prices_pb2,
    simulate_step_pb2,
    simulation_pb2,
    update_sales_pb2
)
import MarketError


class Handler(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        simulation_id: str,
        scenario_id: str,
        start_time: datetime.datetime,
        step_size: int
    ):
        ...

    @abc.abstractmethod
    def init_simulation(
        self,
        simulation: simulation_pb2.Simulation,
        root_config_dir: str
    ) -> tuple[
        initialise_simulation_pb2.InitialiseSimulationResponse,
        MarketError.MarketError
    ]:
        ...

    @abc.abstractmethod
    def update_sales(
        self,
        request: update_sales_pb2.UpdateSalesRequest) -> tuple[
            update_sales_pb2.UpdateSalesResponse,
            MarketError.MarketError
    ]:
        ...

    @abc.abstractmethod
    def simulate_step(
            self,
            controller_simulation_time: datetime.datetime
    ) -> tuple[
        simulate_step_pb2.SimulateStepResponse,
        MarketError.MarketError
    ]:
        ...

    @abc.abstractmethod
    def get_species_prices(
        self
    ) -> get_species_prices_pb2.GetSpeciesPricesResponse:
        ...

    @abc.abstractmethod
    def finalise_simulation(
        self,
        folder_path: str
    ) -> finalise_simulation_pb2.FinaliseSimulationResponse:
        ...

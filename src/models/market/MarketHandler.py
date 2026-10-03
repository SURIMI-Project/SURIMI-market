"""
Handler plugin entrypoint (spec section 1), mirroring ElsaHandler's shape:
config loading, grpc contract validation, wiring into MarketSystem.
"""
import datetime
import logging
from pathlib import Path

import grpc
import pandas as pd
from surimi.v1 import initialise_simulation_pb2, simulate_step_pb2, finalise_simulation_pb2, update_sales_pb2

from MarketError import MarketError
import Handler

from . import consts
from .MarketSystem import MarketSystem


class MarketHandler(Handler.Handler):
    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        self.logger = logging.getLogger(f"{consts.SIMULATION_HANDLER_LOGGER_NAME}.{simulation_id}")
        self.market_system = MarketSystem(currency=consts.SUPPORTED_CURRENCY, start_time=start_time)
        self.size_ratios = None  # optional (spec section 13); not yet consumed by this plan

    def init_simulation(self, simulation, root_config_dir: str):
        species_path = Path(root_config_dir, self.scenario_id, consts.SPECIES_CSV)
        markets_path = Path(root_config_dir, self.scenario_id, consts.PORTS_CSV)
        prices_path = Path(root_config_dir, self.scenario_id, consts.PRICES_CSV)
        size_ratios_path = Path(root_config_dir, self.scenario_id, consts.SIZE_RATIOS_CSV)

        if not (species_path.exists() and markets_path.exists() and prices_path.exists()):
            err_msg = f"Configuration files were not found for scenario '{self.scenario_id}':\n"
            if not species_path.exists():
                err_msg += "- species.csv\n"
            if not markets_path.exists():
                err_msg += "- markets.csv\n"
            if not prices_path.exists():
                err_msg += "- prices.csv\n"
            err_msg += f"The configured directory path is {root_config_dir}."
            return None, MarketError(grpc.StatusCode.NOT_FOUND, err_msg)

        markets_df = pd.read_csv(markets_path)
        species_df = pd.read_csv(species_path)
        prices_df = pd.read_csv(prices_path, parse_dates=["date"])
        if "size" not in prices_df.columns:
            prices_df["size"] = None

        loaded_species = set(species_df["species_code"].dropna().unique())
        known_market_codes = set(markets_df["market_code"].dropna().unique())

        contract_ports = {m.market_code for m in simulation.items.markets}
        contract_species = {s.species_code for s in simulation.items.species}
        category_contract = {pc.category_code for pc in simulation.items.price_categories}

        missing_ports = contract_ports - known_market_codes
        if missing_ports:
            self.logger.critical(f"Missing the following ports: {missing_ports}")
            return None, MarketError(
                grpc.StatusCode.NOT_FOUND, f"The market is missing the following ports: {missing_ports}"
            )

        missing_species = contract_species - loaded_species
        if missing_species:
            self.logger.critical(f"Missing the following species: {missing_species}")
            return None, MarketError(
                grpc.StatusCode.NOT_FOUND, f"The market is missing the following species: {missing_species}"
            )

        # Trim to exactly the contracted local ports, but always keep every
        # 2-letter combined-market code -- those aren't part of the per-port
        # contract and section 4's topology needs them regardless.
        two_letter_codes = {c for c in known_market_codes if len(c) == 2}
        local_codes_5 = {c for c in known_market_codes if len(c) == 5}
        valid_local_codes = local_codes_5 & contract_ports
        markets_df = markets_df[markets_df["market_code"].isin(valid_local_codes | two_letter_codes)]

        self.market_system.species_codes = list(contract_species)
        self.market_system.category_codes_contract = list(category_contract)

        err = self.market_system.setup_topology(markets_df)
        if err is not None:
            return None, err

        if size_ratios_path.exists():
            self.size_ratios = pd.read_csv(size_ratios_path)

        latest_date = pd.Timestamp(prices_df["date"].max())
        self.market_system.fit_all(prices_df, latest_date)

        self.logger.info(f"Tick {self.market_system.tick.get_tick()} initialized.")
        return initialise_simulation_pb2.InitialiseSimulationResponse(simulation_id=self.simulation_id), None

    def update_sales(self, request):
        err = self.market_system.update_sales(
            request.sales_summary.market_sales, request.start_date_time.seconds
        )
        return update_sales_pb2.UpdateSalesResponse(simulation_id=self.simulation_id), err

    def simulate_step(self, controller_simulation_time: datetime.datetime):
        if controller_simulation_time != self.market_system.tick.get_datetime():
            return None, MarketError(
                grpc.StatusCode.NOT_FOUND,
                f"The MarketService's time, {self.market_system.tick.get_timestamp()} does not "
                f"match the input time of {controller_simulation_time.isoformat()}Z",
            )

        self.market_system.step()

        tick = self.market_system.tick.get_tick()
        self.logger.info(f"Tick {tick} completed.")
        return simulate_step_pb2.SimulateStepResponse(simulation_id=self.simulation_id), None

    def get_species_prices(self):
        return self.market_system.get_species_prices_message()

    def finalise_simulation(self, folder_path):
        self.logger.info("Finalising simulation...")
        self.market_system.export_results(self.simulation_id, str(Path(folder_path, self.scenario_id)))
        return finalise_simulation_pb2.FinaliseSimulationResponse(simulation_id=self.simulation_id)

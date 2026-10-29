import definitions
from MarketError import MarketError
import Handler
from . import util

import datetime
import json
import logging
import os
from pathlib import Path

import catboost
from google.protobuf.json_format import ParseDict, MessageToDict
import pandas as pd

from surimi.v1 import (
    initialise_simulation_pb2,
    finalise_simulation_pb2,
    get_species_prices_pb2,
    simulate_step_pb2,
    update_sales_pb2
)

dataset_path = Path("")


class CatboostHandler(Handler.Handler):
    """
    The CatboostHandler uses the CatBoost Regressor machine learning algorithm.

    Our algorithm works as follows:
    1. Initialisation: Train the model using the prices dataset.
    2. Update: Accumulate landings for (market_code, category_code, species_code)
               cartesian product
    3. Simulate: Perform predictions for the current month; one prediction per
                 cartesian product element.
    4. Save history: Store the current month's predicted prices in a historic
                     dataframe.
    5. Finalise: Save to disk.
    """

    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        self.logger = logging.getLogger(
            f"Catboost (simulation id: {simulation_id})")

        self.model = None

        self.historic_prices = pd.DataFrame()
        self.pending_catches = pd.DataFrame()

    def init_simulation(self, simulation, root_config_dir):
        """
        Train the model using the CatBoost Regressor algorithm.
        """
        dataset_path = definitions.ROOT_DIR / f"config/{self.scenario_id}"
        X_train, Y_train, X_test, Y_test = util.get_dataframe_XY(
            dataset_path/"prices.csv", 1.0)

        cat_features = ["category_code", "market_code", "species_code"]

        self.model = catboost.CatBoostRegressor(
            iterations=100,
            learning_rate=0.05,
            depth=8,
            verbose=100,

            # To enable GPU, uncomment below.
            # task_type="GPU"
        )

        self.model.fit(X_train.iloc[:, 1:], Y_train, cat_features=cat_features)

        return initialise_simulation_pb2.InitialiseSimulationResponse(simulation_id=self.simulation_id), None

    def update_sales(self, request):
        # Collect individual sales.
        rows = []

        date = request.start_date_time.ToDatetime()

        dataset_path = definitions.ROOT_DIR / f"config/{self.scenario_id}"
        all_sales = MessageToDict(request)["salesSummary"]["marketSales"]

        # Each element, aka. market sale, is belongs to one port. It can have
        # many sales.
        for market_sale in all_sales:
            new_row_df = pd.DataFrame()
            market_code = market_sale["marketCode"]

            # Each market sale can be thought of as a fleet selling its landings
            # to the market.
            for fleet_sale in market_sale["sales"]:
                # These make the features
                species_code = fleet_sale["species"]["speciesCode"]
                category_code = fleet_sale["categoryCode"]
                landings = fleet_sale["quantity"]
                rows.append({
                    "date": date,
                    "market_code": market_code,
                    "species_code": species_code,
                    "category_code": category_code,
                    "landings_kg": landings
                })

        new_df = pd.DataFrame(rows)
        new_df["date"] = pd.to_datetime(new_df["date"], utc=True)
        self.pending_catches = pd.concat(
            [self.pending_catches, new_df], ignore_index=True
        )

        # TODO add landings
        return update_sales_pb2.UpdateSalesResponse(simulation_id=self.simulation_id), None

    def simulate_step(self, controller_simulation_time):
        requested_date = str(controller_simulation_time)
        requested_date = datetime.datetime.fromisoformat(requested_date)

        max_date_from_df = str(self.pending_catches["date"].max())
        max_date_from_df = datetime.datetime.fromisoformat(max_date_from_df)
        current_month_df = self.pending_catches[self.pending_catches["date"]
                                                == requested_date]

        pred_df = (
            current_month_df
            .groupby(
                [
                    "date",
                    "market_code",
                    "species_code",
                    "category_code",
                ],
                as_index=False
            )
            .agg({
                "landings_kg": "sum"
            })
        )

        pred_df["year"] = pred_df["date"].dt.year
        pred_df["month"] = pred_df["date"].dt.month

        # Date is in the 1st column. We exclude that
        pred_df["price"] = self.model.predict(pred_df.iloc[:, 1:])

        self.historic_prices = pd.concat(
            [self.historic_prices, pred_df],
            ignore_index=True
        )

        self.pending_catches = self.pending_catches[
            self.pending_catches["date"] != requested_date
        ]

        return simulate_step_pb2.SimulateStepResponse(simulation_id=self.simulation_id), None

    def get_species_prices(self):
        msg = {
            "simulation_id": self.simulation_id,
            "species_price_summary": {
                "species_prices": []
            }
        }

        # Get most recent year
        recent_sales_df = self.historic_prices[
            self.historic_prices["date"]
            == self.historic_prices["date"].max()
        ]

        # Filter fields that are required by the interface
        recent_sales_df = recent_sales_df[[
            "species_code",
            "category_code",
            "price"
        ]]

        # Convert to dictionary (we need for ParseDict())
        recent_sales_df["currency"] = "EUR"
        recent_sales = recent_sales_df.to_dict(orient="records")

        # Edge case: "species_code" is in "species"
        for i, sale in enumerate(recent_sales):
            species_code = sale.get("species_code")
            sale["species"] = {
                "species_code": species_code
            }
            del sale["species_code"]

        # Compose the final message
        msg["species_price_summary"]["species_prices"] = recent_sales

        # To inspect the result, uncomment the two lines below
        # with open("predictions.json", "w") as f:
        #     json.dump(msg, f, indent=True)

        return ParseDict(msg, get_species_prices_pb2.GetSpeciesPricesResponse())

    def finalise_simulation(self, folder_path):
        dataset_path = definitions.ROOT_DIR / \
            f"simulations/{self.scenario_id}/{self.simulation_id}/local_prices.csv"
        os.makedirs(dataset_path.parent.resolve(), exist_ok=True)

        # Strip away date
        self.historic_prices["date"] = self.historic_prices["date"].dt.strftime(
            "%Y-%m-%d")

        # Rename columns to match expected feature name conventions
        self.historic_prices.rename(columns={
            "market_code": "market",
            "species_code": "species",
            "landings_kg": "catch"
        }, inplace=True)

        # Filter features (e.g. we don't need year and month)
        self.historic_prices = self.historic_prices[[
            "date", "market", "species", "category_code", "price", "catch"]]
        self.historic_prices.to_csv(dataset_path, index=False)

        return finalise_simulation_pb2.FinaliseSimulationResponse(simulation_id=self.simulation_id)

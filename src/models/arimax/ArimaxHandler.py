import sys

import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from google.protobuf.json_format import MessageToDict, ParseDict

import definitions
import Handler

from surimi.v1 import (
    initialise_simulation_pb2,
    finalise_simulation_pb2,
    get_species_prices_pb2,
    simulate_step_pb2,
    update_sales_pb2,
)

from . import util


class ArimaxHandler(Handler.Handler):
    """
    SARIMAX handler.

    Lifecycle:
        1. Load historical prices and fit a model per series.
        2. Accumulate incoming landings.
        3. Forecast prices for the current simulation month.
        4. Store predicted prices and update model state.
        5. Export the simulation history.
    """

    # Initial configuration based on the notebook's tested orders.
    ORDER = (2, 1, 1)
    SEASONAL_ORDER = (2, 1, 1, 12)

    def __init__(
        self,
        simulation_id,
        scenario_id,
        start_time,
        step_size,
    ):
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        self.logger = logging.getLogger(
            f"SARIMAX (simulation id: {simulation_id})"
        )

        # One fitted model per (market, species, category).
        self.models = {}

        self.pending_catches = pd.DataFrame(
            columns=[
                "date",
                "market_code",
                "species_code",
                "category_code",
                "landings_kg",
            ]
        )

        self.historic_prices = pd.DataFrame(
            columns=[
                "date",
                "market_code",
                "species_code",
                "category_code",
                "landings_kg",
                "price",
            ]
        )

    @staticmethod
    def _month_start(value):
        """Normalize a date to the first day of its month."""
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)

        return timestamp.to_period("M").to_timestamp()

    def _fit_model(self, series_df):
        """
        Fit SARIMAX to monthly prices, using monthly landings as exog.

        Missing months are filled with zero volume and interpolated prices.
        Historical observations with a zero price are treated as missing
        prices, since they may represent missing-data placeholders.
        """
        df = series_df.copy()

        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].map(self._month_start)

        monthly = (
            df.groupby("date")
            .agg(
                price=("price", "mean"),
                volume=("landings_kg", "sum"),
            )
            .sort_index()
        )

        # For testing, restrict to the first 100 samples
        # monthly = monthly.head(24)

        # Ensure a continuous monthly index.
        monthly = monthly.asfreq("MS")

        monthly["volume"] = monthly["volume"].fillna(0)

        # Do not train on zero prices that may indicate missing data.
        monthly["price"] = monthly["price"].replace(0, np.nan)
        monthly["price"] = monthly["price"].interpolate(
            method="linear",
            limit_area="inside",
        )

        monthly = monthly.dropna(subset=["price"])

        if len(monthly) < 24:
            raise ValueError(
                "At least 24 monthly price observations are required "
                "for this initial seasonal SARIMAX configuration."
            )

        model = SARIMAX(
            endog=monthly["price"],
            exog=monthly[["volume"]],
            order=self.ORDER,
            seasonal_order=self.SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        return model.fit(disp=False)

    def init_simulation(self, simulation, root_config_dir):
        """
        Load historical prices and train a model for each series.
        """
        dataset_path = (
            definitions.ROOT_DIR
            / f"config/{self.scenario_id}"
            / "prices.csv"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Historical prices dataset not found: {dataset_path}"
            )

        df = util.preprocess_dataset(str(dataset_path))
        # df = df.rename(columns={"port_code": "market_code"})
        df = df[df["species_code"] == "ANE"]

        df["date"] = pd.to_datetime(df["date"])

        group_columns = [
            # "market_code",
            "species_code"
            # "category_code",
        ]

        self.models.clear()

        for key, group in df.groupby(group_columns):
            # Limit num of models for testing
            # if len(self.models) > 10:
            #     break

            try:
                self.models[key] = self._fit_model(group)
                self.logger.info("Trained SARIMAX model for %s", key)
            except (ValueError, np.linalg.LinAlgError) as exc:
                # self.logger.warning(
                #     "Skipping series %s: %s", key, exc
                # )
                pass

        if not self.models:
            raise ValueError(
                "No SARIMAX models could be trained. "
                "Check the dataset and model configuration."
            )

        return (
            initialise_simulation_pb2.InitialiseSimulationResponse(
                simulation_id=self.simulation_id
            ),
            None,
        )

    def update_sales(self, request):
        """
        Accumulate landings for the relevant simulation month.
        """
        request_dict = MessageToDict(request)

        sales_summary = request_dict.get("salesSummary", {})
        market_sales = sales_summary.get("marketSales", [])

        date = self._month_start(
            request.start_date_time.ToDatetime()
        )

        rows = []

        for market_sale in market_sales:
            market_code = market_sale["marketCode"]

            for fleet_sale in market_sale.get("sales", []):
                rows.append({
                    "date": date,
                    "market_code": market_code,
                    "species_code": (
                        fleet_sale["species"]["speciesCode"]
                    ),
                    "category_code": fleet_sale["categoryCode"],
                    "landings_kg": float(fleet_sale["quantity"]),
                })

        if rows:
            new_df = pd.DataFrame(rows)

            self.pending_catches = pd.concat(
                [self.pending_catches, new_df],
                ignore_index=True,
            )

        return (
            update_sales_pb2.UpdateSalesResponse(
                simulation_id=self.simulation_id
            ),
            None,
        )

    def simulate_step(self, controller_simulation_time):
        """
        Aggregate the current month's landings, forecast prices, and
        advance the state of each applicable SARIMAX model.
        """
        requested_date = self._month_start(
            controller_simulation_time
        )

        if self.pending_catches.empty:
            self.logger.warning(
                "No pending landings for %s", requested_date
            )
            return (
                simulate_step_pb2.SimulateStepResponse(
                    simulation_id=self.simulation_id
                ),
                None,
            )

        current_month = self.pending_catches[
            self.pending_catches["date"].map(self._month_start)
            == requested_date
        ]

        if current_month.empty:
            self.logger.warning(
                "No landings found for %s", requested_date
            )
            return (
                simulate_step_pb2.SimulateStepResponse(
                    simulation_id=self.simulation_id
                ),
                None,
            )

        aggregated = (
            current_month.groupby(
                [
                    "date",
                    "market_code",
                    "species_code",
                    "category_code",
                ],
                as_index=False,
            )
            .agg({"landings_kg": "sum"})
        )

        predictions = []

        for row in aggregated.to_dict(orient="records"):
            key = (
                # row["market_code"],
                row["species_code"],
                # row["category_code"],
            )

            if key not in self.models:
                # self.logger.warning(
                #     "No trained SARIMAX model for %s; skipping", key
                # )
                continue
            else:
                print(f"Trained model found for {key}")

            fitted_model = self.models[key]
            volume = float(row["landings_kg"])

            try:
                # Forecast one month using the observed landings volume.
                forecast = fitted_model.get_forecast(
                    steps=1,
                    exog=pd.DataFrame({"volume": [volume]}),
                )

                predicted_price = float(
                    np.asarray(forecast.predicted_mean)[0]
                )

                if not np.isfinite(predicted_price):
                    raise ValueError(
                        "SARIMAX returned a non-finite prediction"
                    )

                predictions.append({
                    "date": requested_date,
                    "market_code": row["market_code"],
                    "species_code": row["species_code"],
                    "category_code": row["category_code"],
                    "landings_kg": volume,
                    "price": predicted_price,
                })

                # Advance the model using the predicted price and the
                # observed exogenous volume, so it can forecast the
                # following month without an actual price observation.
                self.models[key] = fitted_model.append(
                    endog=np.array([predicted_price]),
                    exog=np.array([[volume]]),
                    refit=False,
                )

            except (ValueError, np.linalg.LinAlgError) as exc:
                self.logger.exception(
                    "Forecast failed for %s: %s", key, exc
                )
            
        if predictions:
            self.historic_prices = pd.concat(
                [
                    self.historic_prices,
                    pd.DataFrame(predictions),
                ],
                ignore_index=True,
            )

        # Remove processed landings for this month only.
        self.pending_catches = self.pending_catches[
            self.pending_catches["date"].map(self._month_start)
            != requested_date
        ].reset_index(drop=True)

        return (
            simulate_step_pb2.SimulateStepResponse(
                simulation_id=self.simulation_id
            ),
            None,
        )

    def get_species_prices(self):
        """
        Return the prices predicted in the most recent simulated month.
        """
        msg = {
            "simulation_id": self.simulation_id,
            "species_price_summary": {
                "species_prices": [],
            },
        }

        if self.historic_prices.empty:
            return ParseDict(
                msg,
                get_species_prices_pb2.GetSpeciesPricesResponse(),
            )

        latest_date = self.historic_prices["date"].max()

        latest = self.historic_prices[
            self.historic_prices["date"] == latest_date
        ][
            ["species_code", "category_code", "price"]
        ].copy()

        latest["currency"] = "EUR"

        sales = latest.to_dict(orient="records")

        for sale in sales:
            species_code = sale.pop("species_code")
            sale["species"] = {"species_code": species_code}

        msg["species_price_summary"]["species_prices"] = sales

        return ParseDict(
            msg,
            get_species_prices_pb2.GetSpeciesPricesResponse(),
        )

    def finalise_simulation(self, folder_path):
        """
        Export simulated prices using the expected CSV schema.
        """
        dataset_path = definitions.ROOT_DIR / \
            f"simulations/{self.scenario_id}/{self.simulation_id}/local_prices.csv"
        os.makedirs(dataset_path.parent.resolve(), exist_ok=True)

        output_df = self.historic_prices.copy()

        if not output_df.empty:
            output_df["date"] = pd.to_datetime(
                output_df["date"]
            ).dt.strftime("%Y-%m-%d")

            output_df = output_df.rename(columns={
                "market_code": "market",
                "species_code": "species",
                "landings_kg": "catch",
            })

            output_df = output_df[
                [
                    "date",
                    "market",
                    "species",
                    "category_code",
                    "price",
                    "catch",
                ]
            ]

        output_df.to_csv(dataset_path, index=False)

        return (
            finalise_simulation_pb2.FinaliseSimulationResponse(
                simulation_id=self.simulation_id
            )
        )
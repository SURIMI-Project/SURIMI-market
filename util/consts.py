"""
This module contains constants used throughout the project. They can be commonly
used literals that don't change.
"""

GRPC_HOST = "[::]"
GRPC_PORT = "5001"

SPECIES_CSV = "species.csv"
PORTS_CSV = "ports.csv"
PRICES_CSV = "prices.csv"

# The MarketModel should accept any arbitrary currencies and handle it
# accordingly. At the moment, it only supports EUR.
SUPPORTED_CURRENCY = "EUR"

# TODO add supported measurement unit

# Logging
APP_LOGGER_NAME = "app"
VAULT_LOGGER_NAME = "VaultService"
MARKET_SERVICE_LOGGER_NAME = "MarketService"
MARKET_SYSTEM_LOGGER_NAME = "MarketSystem"
S3_LOGGER_NAME = "S3_Storage"
GLOBAL_MARKET_LOGGER_NAME = "GlobalMarket"
LOCAL_MARKET_LOGGER_NAME = "LocalMarket"
SIMULATION_HANDLER_LOGGER_NAME = "simulation"
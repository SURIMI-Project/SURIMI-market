# Standard library
from itertools import product
import logging

# Util modules
from util import consts

# Src modules
from .LocalMarket import LocalMarket

class GlobalMarket:
    """
    Represents the global market context, computing global species prices
    as an average of valid local prices (preliminary solution).
    """
    def __init__(self):
        self.logger = logging.getLogger(consts.GLOBAL_MARKET_LOGGER_NAME)
        self.global_price_history = []  # [(tick, {"avg_prices": {...}, "total_catches": {...}})]

    def get_most_recent_price(self, species_code: str, gear_code: str) -> float | None:
        """
        Get the most recent price for a species.

        Arguments
        ---------
            species_code : The species' code.
            gear_code : The gear code that the species price is associated with.
        """
        assert len(self.global_price_history) > 0, "There must be a history before the most recent price can be retrieved."
        return self.global_price_history[-1][1].get((species_code, gear_code))
    
    def compute_prices(self,
                       local_markets: dict[str, LocalMarket], 
                       species_codes: list[str],
                       gear_codes: list[str]) -> tuple[dict, dict]:
        """
        Set the current global species prices using the mean of species price.

        A species' global price is the mean of the corresponding species' price
        across all local markets. I.e., if species X costs
        
        - 3 EUR in Port A
        - 5 EUR in Port B
        
        then X's global price is 4.

        The species' global prices are also based on the fleet vessel's gear
        code. The method finds all the different combinations of species and
        gear codes using the Cartesian product of `species_codes` and 
        `gear_codes`.

        Arguments
        ---------
            local markets : A dictionary of all local markets that the global
                            prices are based on, according to the contract.
            species_list : The list of all species, according to the contract.
            gear_codes : The list of all gear codes, according to the contract.
        
        Returns
        -------
            global_prices: Dictionary of the mean price for each species.
            global_catches: Dictionary of accumulated catches for the species
                            globally.
        """

        # Return values
        global_catches = {}
        global_prices = {}

        # Species prices depend on the gear code, so we a use Cartesian product
        # to combine them.
        lookup_keys = list(product(species_codes, gear_codes))
        missing_global_count = 0
        valid_global_count = 0

        for key in lookup_keys:
            # key[n] = (species_code, gear_code)
            valid_prices = []
            total_catch = 0.0

            # Each market accumulates catches between ticks.
            for m in local_markets.values():
                price = m.local_prices.get(key)
                if price is not None and price > 0:
                    valid_prices.append(price)
                
                if len(m.price_history) > 0:
                    total_catch += m.price_history[-1][1]["catches"].get(key, 0.0)

            if len(valid_prices) > 0:
                # TODO improve this, as it shouldnt just do a mean of the local market prices.
                # TODO take inflation as a parameter.
                global_prices[key] = sum(valid_prices) / len(valid_prices)
                valid_global_count += 1

            else:
                global_prices[key] = None
                missing_global_count += 1

            if total_catch > 0:
                global_catches[key] = total_catch
            
        self.logger.debug(
            f"Global prices (valid={valid_global_count}, missing={missing_global_count}"
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            for s, p in global_prices.items():
                if p is not None:
                    self.logger.debug(f"  {s}: {p:.2f} EUR/kg")
                else:
                    self.logger.debug(f"  {s}: None (no valid local prices)")
    
        return global_prices, global_catches

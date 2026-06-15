class GlobalMarket:
    """
    Represents the global market context, computing global species prices
    as an average of valid local prices (preliminary solution).
    """
    def __init__(self, species_dict):
        self.species_dict = species_dict
        self.global_prices: dict[tuple[str, str], float | None] = {}
        self.global_price_history = []  # [(tick, {"avg_prices": {...}, "total_catches": {...}})]

    def get_global_price(self, species_code: str, gear_code: str) -> float | None:
        return self.global_prices.get((species_code, gear_code))
    
    # TODO the global market computations should happen here.


class Species:
    """
    Represents a species traded in the market.
    Maintains epsilon for price elasticity.
    Price path memory is handled per LocalMarket.
    """
    def __init__(self, species_name: str, species_code: str, epsilon=0.5):
        self.name = species_name
        self.code = species_code
        self.epsilon = epsilon

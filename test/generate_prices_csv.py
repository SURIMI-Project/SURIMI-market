"""
This file generates
"""

if __name__ == "__main__":
    from itertools import product
    import pandas as pd
    from pathlib import Path
    import datetime

    data = {
        "date": [
            str(datetime.date(y, m, 1))
                for y in range(2010, 2013)
                    for m in range(1, 13)
            ],
        "market_code": [
            "PORT_A",
            "PORT_B",
            "PORT_C",
            "PORT_D"
        ],
        "species_code": [
            "Species_A",
            "Species_B",
            "Species_C",
            "Species_D"
        ],
        "category_code": [
            "PriceCategory_A",
            "PriceCategory_B",
            "PriceCategory_C",
            "PriceCategory_D"
        ],
        "landings_kg": [
            "1"
        ],
        "price": [
            "1"
        ],
        "currency": [
            "EUR"
        ],
        "measurement_unit": [
            "kg"
        ]
    }


    combinations = product(*data.values())
    df = pd.DataFrame(combinations, columns=data.keys())
    
    out_path = Path(__file__).parent / "market/fixtures/my_scenario_name/prices.csv"
    df.to_csv(out_path, index=False)
else:
    raise ImportError("This module cannot be imported")
from pathlib import Path
import pandas as pd
from datetime import datetime


def preprocess_dataset(path: str) -> pd.DataFrame:
    """
    We perform the following:
    1. Convert `date` column to datetime.
    2. For duplicate `(date, port_code, species_code, category_code)` entries,
       use the mean price and summed landings_kg.
    """
    df = pd.read_csv(Path(path))

    # Only keep Fresh - Whole
    df = df[df["category_code"] == "Fresh - Whole"]

    # Convert the date column to datetime
    df["date"] = df["date"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d")
    )

    # There may be duplicate rows of @param input_params. Use the first occurence
    # for each duplicate.
    features = {
        "price": "mean",
        "landings_kg": "sum"
    }

    input_params = ["date", "market_code", "species_code", "category_code"]

    for col in df.columns.difference(["date", "market_code", *features.keys()]):
        features[col] = "first"

    df = (
        df.groupby(input_params, as_index=False)
        .agg(features)
    )

    # Fill missing dates
    # Get all possible combinations of the input params
    full_index = pd.MultiIndex.from_product(
        [df[param].unique() for param in input_params],
        names=input_params
    )

    # Insert nan in prices and weights for missing rows
    df = (
        df.set_index(input_params)
        .reindex(full_index)
        .reset_index()
    )

    df = df.fillna(0)

    return df


def get_species_dataframe(df: pd.DataFrame,
                          species: str,
                          port: str) -> pd.DataFrame:
    species_df = df.sort_values("date", ascending=True)
    species_df = species_df[(species_df["species_code"] == species) & (
        species_df["port_code"] == port)]

    print(f"Species: {species}")
    print(f"Port: {port}")
    return species_df


def split_dataset(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split the dataset by years
    years = list(df["date"].dt.year.unique())
    split_threshold = round(ratio * len(years))

    train_df = df[df["date"].dt.year.isin(years[:split_threshold])]
    test_df = df[df["date"].dt.year.isin(years[split_threshold:])]

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    total_samples = len(train_df) + len(test_df)
    train_ratio = len(train_df) / total_samples
    test_ratio = len(test_df) / total_samples

    print(
        f"Ratio: [{round(train_ratio*100)}/{round(test_ratio*100)}] x [train/test]")
    return train_df, test_df


def get_dataframe(filepath: str,
                  ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = preprocess_dataset(filepath)
    # df = get_species_dataframe(df, species, port)
    train, test = split_dataset(df, ratio)
    return train, test


def get_dataframe_XY(filepath: str,
                     ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = get_dataframe(filepath, ratio)

    train_df["year"] = train_df["date"].dt.year
    train_df["month"] = train_df["date"].dt.month

    test_df["year"] = test_df["date"].dt.year
    test_df["month"] = test_df["date"].dt.month

    features = [
        "date",
        "species_code",
        "market_code",
        "category_code",
        "landings_kg",
        "year",
        "month"
    ]

    X_train = train_df[features]

    Y_train = train_df["price"]


    X_test = test_df[features]

    Y_test = test_df["price"]

    return X_train, Y_train, X_test, Y_test
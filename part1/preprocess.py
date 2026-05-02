from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from part1.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    LABEL,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    SPLITS_PATH,
)
from part1.data import load_data


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["RestingBP"] > 0].copy()
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    return df


def load_splits(splits_path: Path) -> pd.Series:
    return pd.read_csv(splits_path).set_index("row_index")["split"]


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for numeric / categorical / binary features."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # Nominal categoricals: one-hot, drop first level to avoid collinearity.
    nominal_features = [f for f in CATEGORICAL_FEATURES if f not in ORDINAL_FEATURES]
    nominal = OneHotEncoder(drop="first", handle_unknown="ignore")
    # Ordinal categoricals: encode by clinical risk order then standardize.
    ordinal_features = list(ORDINAL_FEATURES.keys())
    ordinal = Pipeline(
        [
            (
                "encode",
                OrdinalEncoder(
                    categories=list(ORDINAL_FEATURES.values()),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("nom", nominal, nominal_features),
            ("ord", ordinal, ordinal_features),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def _strip_prefix(name: str) -> str:
    for prefix in ("num__", "nom__", "ord__", "bin__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame


def prepare(
    data_path: Path = DATA_PATH,
    splits_path: Path = SPLITS_PATH,
) -> PreparedData:
    df = load_data(data_path)
    splits = load_splits(splits_path)

    df = df.assign(split=df.index.map(splits))
    df = clean(df)

    train_df = (
        df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    )
    test_df = (
        df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    )

    X_train_raw = train_df.drop(columns=[LABEL])
    X_test_raw = test_df.drop(columns=[LABEL])
    y_train = train_df[LABEL].to_numpy()
    y_test = test_df[LABEL].to_numpy()

    preprocessor = build_preprocessor()
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw))
    X_test = np.asarray(preprocessor.transform(X_test_raw))
    feature_names = [_strip_prefix(n) for n in preprocessor.get_feature_names_out()]

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
        raw_train=train_df,
        raw_test=test_df,
    )


def main() -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 40)

    data = prepare()
    print(f"Train: {data.X_train.shape}, Test: {data.X_test.shape}")
    print(f"Train positive rate: {data.y_train.mean():.3f}")
    print(f"Test  positive rate: {data.y_test.mean():.3f}")
    print(f"\nFeatures after preprocessing ({len(data.feature_names)}):")
    for name in data.feature_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()

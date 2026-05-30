"""
Automated preprocessing pipeline for the Titanic dataset.

Tahapan:
1. Load raw dataset (data_titanic_raw/train.csv).
2. Tangani missing values (Age, Embarked, Cabin).
3. Hapus duplikat.
4. Feature engineering (Title, FamilySize, IsAlone).
5. Drop kolom non-informatif (PassengerId, Name, Ticket).
6. Encoding kategorikal (Sex, Embarked, Title).
7. Tangani outlier Fare dengan IQR capping.
8. Standarisasi fitur numerik.
9. Train/test split (stratified).
10. Simpan hasil ke folder preprocessing/dataset_preprocessing/.

Dirancang agar dapat dipanggil dari CLI maupun di-import sebagai modul.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Capt": "Rare", "Sir": "Rare",
}

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]


def load_data(input_path: str | os.PathLike) -> pd.DataFrame:
    """Load raw Titanic CSV."""
    df = pd.read_csv(input_path)
    print(f"[load_data] Shape: {df.shape}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[remove_duplicates] removed {before - len(df)} duplicates")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].map(TITLE_MAP).fillna("Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["PassengerId", "Name", "Ticket"] if c in df.columns]
    return df.drop(columns=cols)


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def cap_outliers_iqr(df: pd.DataFrame, column: str = "Fare") -> pd.DataFrame:
    df = df.copy()
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower, upper=upper)
    print(f"[cap_outliers_iqr] {column}: [{lower:.2f}, {upper:.2f}]")
    return df


def scale_numeric(df: pd.DataFrame, features=None) -> pd.DataFrame:
    df = df.copy()
    features = features or NUMERIC_FEATURES
    features = [f for f in features if f in df.columns]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def split_data(df: pd.DataFrame, target: str = "Survived",
               test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocess(input_path: str, output_dir: str) -> pd.DataFrame:
    """Run end-to-end preprocessing and persist outputs."""
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)
    df = drop_unused(df)
    df = encode_categorical(df)
    df = cap_outliers_iqr(df, "Fare")
    df = scale_numeric(df)

    clean_path = Path(output_dir) / "titanic_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"[preprocess] saved cleaned dataset -> {clean_path}")

    X_train, X_test, y_train, y_test = split_data(df)
    X_train.to_csv(Path(output_dir) / "X_train.csv", index=False)
    X_test.to_csv(Path(output_dir) / "X_test.csv", index=False)
    y_train.to_csv(Path(output_dir) / "y_train.csv", index=False)
    y_test.to_csv(Path(output_dir) / "y_test.csv", index=False)
    print(f"[preprocess] train: {X_train.shape}  test: {X_test.shape}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate Titanic preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        default="data_titanic_raw/train.csv",
        help="Path to raw train.csv (default: data_titanic_raw/train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="preprocessing/dataset_preprocessing",
        help="Directory to write processed outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(args.input, args.output_dir)


if __name__ == "__main__":
    main()

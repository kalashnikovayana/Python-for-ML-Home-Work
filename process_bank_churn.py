import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Optional


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Tuple:
    df = raw_df.drop(columns=["Surname"], errors="ignore").copy()

    input_cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    target_col = "Exited"

    X = df[input_cols]
    y = df[target_col]

    X_train, X_val, train_targets, val_targets = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_train.select_dtypes('object').columns.tolist()

    scaler = None
    if scaler_numeric:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    def apply_encoding(X_df):
        encoded = encoder.transform(X_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_df.index)
        return pd.concat([X_df.drop(columns=categorical_cols), encoded_df], axis=1)

    X_train = apply_encoding(X_train)
    X_val = apply_encoding(X_val)

    return X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder


def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str],
                        scaler: Optional[StandardScaler],
                        encoder: Optional[OneHotEncoder]) -> pd.DataFrame:
    df = new_df.copy()
    df.drop(columns=["Surname"], errors="ignore", inplace=True)

    X = df[input_cols].copy()
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    categorical_cols = X.select_dtypes('object').columns.tolist()

    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    if encoder is not None and categorical_cols:
        encoded = encoder.transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

    return X


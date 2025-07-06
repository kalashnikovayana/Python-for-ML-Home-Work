import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Optional


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Tuple:
    """
    Preprocess the raw dataframe for Decision Tree or Logistic Regression models.

    Args:
        raw_df (pd.DataFrame): Raw input dataframe.
        scaler_numeric (bool): Whether to scale numeric features. Default is True.

    Returns:
        Tuple: X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder
    """
    # 1. Видаляємо колонку Surname (вона нам не потрібна)
    df = raw_df.drop(columns=["Surname"], errors="ignore").copy()

    # 2. Колонки
    input_cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    target_col = "Exited"

    X = df[input_cols]
    y = df[target_col]

    # 3. Train/Validation split
    X_train, X_val, train_targets, val_targets = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Обробка числових і категоріальних колонок
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_train.select_dtypes('object').columns.tolist()

    # 5. Масштабування числових
    scaler = None
    if scaler_numeric:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    # 6. One-hot encoding категоріальних
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    def apply_encoding(X_df):
        encoded = encoder.transform(X_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_df.index)
        X_encoded = pd.concat([X_df.drop(columns=categorical_cols), encoded_df], axis=1)
        return X_encoded

    X_train = apply_encoding(X_train)
    X_val = apply_encoding(X_val)

    return X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder


    def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], 
                        scaler: Optional[StandardScaler], 
                        encoder: Optional[OneHotEncoder]) -> pd.DataFrame:
        """
        Preprocess new data (e.g. test set) using fitted scaler and encoder.
    
        Args:
            new_df (pd.DataFrame): Raw test data.
            input_cols (List[str]): List of feature columns to keep.
            scaler (StandardScaler or None): Fitted scaler from train data (or None if not used).
            encoder (OneHotEncoder or None): Fitted encoder from train data.
    
        Returns:
            pd.DataFrame: Processed input features for prediction.
        """
        df = new_df.copy()
        
        # Видаляємо колонку Surname, якщо є
        df.drop(columns=["Surname"], errors="ignore", inplace=True)
    
        # Вибираємо лише необхідні колонки
        X = df[input_cols].copy()
    
        # Визначаємо типи колонок
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        categorical_cols = X.select_dtypes('object').columns.tolist()
    
        # Масштабування числових ознак (якщо scaler не None)
        if scaler is not None:
            X[numeric_cols] = scaler.transform(X[numeric_cols])
    
        # One-hot encoding
        if encoder is not None and categorical_cols:
            encoded = encoder.transform(X[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
            X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    
        return X


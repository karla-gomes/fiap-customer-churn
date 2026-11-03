"""Pré-processamento em Pipeline do Scikit-Learn."""
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas e categóricas do DataFrame."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str], scale_numeric: bool = True) -> ColumnTransformer:
    """Codifica categóricas (One-Hot) e (opcionalmente) escala numéricas (StandardScaler)."""
    
    num_transformer = Pipeline([("scaler", StandardScaler())]) if scale_numeric else "passthrough"
    
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [("num", num_transformer, num_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_model_pipeline(model: BaseEstimator, X_ref: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Monta o pipeline completo (pré-processamento + classificador) extraindo as colunas de X_ref."""
    num_cols, cat_cols = split_columns(X_ref)
    
    prep = build_preprocessor(num_cols, cat_cols, scale_numeric)
    
    return Pipeline([
        ("preprocessor", prep),
        ("classifier", model)
    ])
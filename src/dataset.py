"""Leitura, limpeza estrutural e divisão treino/teste."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COLUMNS_TO_DROP, 
    DATA_RAW_PATH, 
    FEATURES_TO_DROP,
    RANDOM_STATE, 
    TARGET_COL, 
    TEST_SIZE
)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Lê o dataset bruto (xlsx do IBM Telco)."""
    return pd.read_excel(path or DATA_RAW_PATH)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte `Total Charges` para número. Brancos viram 0.0 (clientes novos, Tenure Months = 0).
    Texto inválido (que não seja branco) levanta ValueError em vez de ser mascarado."""
    out = df.copy()
    
    raw = out["Total Charges"].astype(str).str.strip().replace("", np.nan)
    converted = pd.to_numeric(raw, errors="coerce")
    
    invalid = converted.isna() & raw.notna()
    if invalid.any():
        raise ValueError(f"Total Charges com {int(invalid.sum())} valores não numéricos (além de brancos).")
    
    out["Total Charges"] = converted.fillna(0.0)
    return out


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica a limpeza e devolve (X, y), sem identificadores, leakage nem colunas descartadas."""
    df = fix_total_charges(df)
    
    y = df[TARGET_COL].astype(int)
    
    # Junta as colunas estruturais a remover e as features da ablação
    to_drop = [c for c in [*COLUMNS_TO_DROP, *FEATURES_TO_DROP, TARGET_COL] if c in df.columns]
    
    return df.drop(columns=to_drop), y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Divisão 80/20 estratificada pelo target, com seed fixa (idêntica à do notebook)."""
    return train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
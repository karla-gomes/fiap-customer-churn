import pandas as pd
import pytest
from src.predict import ChurnPredictor

def test_preprocess_payload(monkeypatch):
    # Ignora o carregamento do modelo real durante o teste unitário
    monkeypatch.setattr(ChurnPredictor, "__init__", lambda self: None)
    
    predictor = ChurnPredictor()
    raw_payload = {
        "Total Charges": " ",
        "CustomerID": "12345",
        "Monthly Charges": 50.0
    }
    
    df_clean = predictor.preprocess_payload(raw_payload)
    
    # Valida que as colunas definidas em COLUMNS_TO_DROP foram removidas com sucesso
    assert "Total Charges" not in df_clean.columns
    assert "CustomerID" not in df_clean.columns
    
    # Valida que colunas de features válidas continuam no DataFrame
    assert "Monthly Charges" in df_clean.columns
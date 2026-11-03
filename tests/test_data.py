import pandas as pd
import pytest
from src.dataset import fix_total_charges, build_features_and_target
from src.config import TARGET_COL

def test_fix_total_charges_success():
    """Valida se espaços em branco são convertidos para 0.0 e numéricos mantidos."""
    df_mock = pd.DataFrame({"Total Charges": ["100.5", " ", "", "250.0"]})
    df_result = fix_total_charges(df_mock)
    
    assert df_result["Total Charges"].iloc[1] == 0.0
    assert df_result["Total Charges"].iloc[2] == 0.0
    assert df_result["Total Charges"].iloc[0] == 100.5
    assert df_result["Total Charges"].dtype == float

def test_fix_total_charges_invalid():
    """Garante que a função levanta ValueError se houver texto inválido não-branco."""
    df_mock = pd.DataFrame({"Total Charges": ["100.5", "erro_digitação"]})
    with pytest.raises(ValueError, match="não numéricos"):
        fix_total_charges(df_mock)

def test_build_features_and_target(monkeypatch):
    """Valida se a função separa X e y corretamente e remove colunas indesejadas."""
    # Mockando config para o teste não depender do config.py real
    monkeypatch.setattr("src.dataset.COLUMNS_TO_DROP", ["CustomerID", "Country"])
    monkeypatch.setattr("src.dataset.FEATURES_TO_DROP", [])
    monkeypatch.setattr("src.dataset.TARGET_COL", "Churn Value")
    
    df_mock = pd.DataFrame({
        "CustomerID": ["123", "456"],
        "Country": ["US", "BR"],
        "Total Charges": ["10.0", "20.0"],
        "Monthly Charges": [5.0, 10.0],
        "Churn Value": [1, 0]
    })
    
    X, y = build_features_and_target(df_mock)
    
    # O target deve ser separado
    assert y.name == "Churn Value"
    assert list(y.values) == [1, 0]
    
    # Colunas de drop e target não podem estar em X
    assert "CustomerID" not in X.columns
    assert "Country" not in X.columns
    assert "Churn Value" not in X.columns
    
    # Features válidas devem permanecer
    assert "Total Charges" in X.columns
    assert "Monthly Charges" in X.columns
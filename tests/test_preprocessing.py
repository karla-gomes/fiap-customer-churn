import pandas as pd
from sklearn.compose import ColumnTransformer
from src.preprocessing import split_columns, build_preprocessor

def test_split_columns():
    """Valida se as colunas são separadas corretamente em numéricas e categóricas."""
    df_mock = pd.DataFrame({
        "Idade": [25, 30],                  # Numérica (int)
        "Saldo": [100.50, 200.75],          # Numérica (float)
        "Categoria": ["A", "B"],            # Categórica (object)
        "Status": pd.Categorical(["S", "N"]) # Categórica (category)
    })
    
    num_cols, cat_cols = split_columns(df_mock)
    
    # Verifica se a separação identificou os tipos corretos
    assert set(num_cols) == {"Idade", "Saldo"}
    assert set(cat_cols) == {"Categoria", "Status"}

def test_build_preprocessor():
    """Garante que a função retorna um ColumnTransformer configurado."""
    num_cols = ["Idade", "Saldo"]
    cat_cols = ["Categoria", "Status"]
    
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    
    # Valida o tipo do objeto retornado
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Valida se os nomes das transformações (num e cat) foram definidos
    transformers = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformers
    assert "cat" in transformers
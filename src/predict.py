import pandas as pd
import joblib
from src.config import get_model_path, COLUMNS_TO_DROP, FEATURES_TO_DROP
from src.dataset import fix_total_charges

class ChurnPredictor:
    def __init__(self):
        self.model_path = get_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def preprocess_payload(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        
        if "Total Charges" in df.columns:
            df = fix_total_charges(df)
            
        cols_to_drop = [c for c in [*COLUMNS_TO_DROP, *FEATURES_TO_DROP] if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        return df

    def predict(self, data: dict) -> dict:
        df_processed = self.preprocess_payload(data)
        propensity = self.model.predict_proba(df_processed)[0][1]
        
        return {
            "churn_propensity": round(float(propensity), 4),
            "high_risk": bool(propensity > 0.5)
        }
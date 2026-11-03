"""Avalia o modelo campeão no conjunto de teste (split fixo, seed 42) e por subgrupo."""
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import get_model_path
from src.dataset import build_features_and_target, load_raw_data, split_train_test

THRESHOLD = 0.5
GROUP_COLS = ["Gender", "Senior Citizen", "Partner", "Dependents", "Contract"]


def main() -> None:
    X, y = build_features_and_target(load_raw_data())
    _, X_test, _, y_test = split_train_test(X, y)
    model = joblib.load(get_model_path())

    proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    pred = (proba >= THRESHOLD).astype(int)

    print("=== Geral (teste) ===")
    print(f"ROC-AUC   {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC    {average_precision_score(y_test, proba):.4f}")
    print(f"F1        {f1_score(y_test, pred):.4f}")
    print(f"Accuracy  {accuracy_score(y_test, pred):.4f}")
    print(f"Recall    {recall_score(y_test, pred):.4f}")
    print(f"Precision {precision_score(y_test, pred):.4f}")

    print("\n=== Por subgrupo (teste) ===")
    rows = []
    for col in GROUP_COLS:
        for value, idx in X_test.groupby(col).groups.items():
            yt, pt, pr = y_test.loc[idx], proba.loc[idx], pred.loc[idx]
            rows.append({
                "Grupo": f"{col}={value}",
                "N": len(idx),
                "Churn real": round(yt.mean(), 3),
                "Sinalizados": round(pr.mean(), 3),
                "Recall": round(recall_score(yt, pr), 3),
                "Precision": round(precision_score(yt, pr, zero_division=0), 3),
                "ROC-AUC": round(roc_auc_score(yt, pt), 3),
            })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
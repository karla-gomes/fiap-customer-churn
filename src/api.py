"""API de inferência de churn (FastAPI)."""
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import ChurnPredictor
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse


YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """Features do cliente, com os mesmos nomes de coluna usados no treino."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Gender": "Female",
                "Senior Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure Months": 2,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 70.7,
                "Total Charges": 151.65,
                "CLTV": 3239,
            }
        },
    )

    gender: Literal["Female", "Male"] = Field(alias="Gender")
    senior_citizen: YesNo = Field(alias="Senior Citizen")
    partner: YesNo = Field(alias="Partner")
    dependents: YesNo = Field(alias="Dependents")
    tenure_months: int = Field(alias="Tenure Months", ge=0)
    phone_service: YesNo = Field(alias="Phone Service")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(alias="Multiple Lines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(alias="Internet Service")
    online_security: YesNoNoInternet = Field(alias="Online Security")
    online_backup: YesNoNoInternet = Field(alias="Online Backup")
    device_protection: YesNoNoInternet = Field(alias="Device Protection")
    tech_support: YesNoNoInternet = Field(alias="Tech Support")
    streaming_tv: YesNoNoInternet = Field(alias="Streaming TV")
    streaming_movies: YesNoNoInternet = Field(alias="Streaming Movies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless_billing: YesNo = Field(alias="Paperless Billing")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(alias="Payment Method")
    monthly_charges: float = Field(alias="Monthly Charges", ge=0)
    total_charges: float = Field(alias="Total Charges", ge=0)
    cltv: int = Field(alias="CLTV", ge=0)


class PredictionResponse(BaseModel):
    churn_propensity: float = Field(ge=0, le=1)
    high_risk: bool


app = FastAPI(
    title="Telco Churn API",
    description="Propensão de churn por cliente (RandomForest campeão).",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _load_predictor() -> ChurnPredictor:
    return ChurnPredictor()


def get_predictor() -> ChurnPredictor:
    """Dependência: carrega o modelo uma única vez; 503 se o artefato não existir."""
    try:
        return _load_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    return PredictionResponse(**predictor.predict(customer.model_dump(by_alias=True)))
import httpx

from src.api import CustomerFeatures

payload = CustomerFeatures.model_config["json_schema_extra"]["example"]
response = httpx.post("http://127.0.0.1:8000/predict", json=payload)
print(response.status_code, response.json())
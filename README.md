# Tech Challenge Fase 1: Previsão de Churn

Pipeline preditivo de churn para uma operadora de telecomunicações: EDA, comparação de modelos clássicos (Regressão Logística, Random Forest e MLP, com Scikit-Learn) e modelo campeão servido via API REST (FastAPI). Projeto da Fase 1 da pós tech em Engenharia de Machine Learning da FIAP.

## Resultados

Conjunto de teste (1.409 clientes, 26,5% de churn):

| Modelo | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| Baseline: Regressão Logística | 0,8465 | 0,6412 | 0,6149 | 0,7727 |
| **Campeão: Random Forest** | **0,8522** | **0,6634** | **0,6377** | **0,7861** |

Detalhes, limitações e vieses estão em [`docs/model_card.md`](docs/model_card.md).

## Requisitos
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (gerenciador de dependências)

## Instalação

```bash
git clone <URL-DO-REPOSITORIO>
cd fiap-customer-churn
uv sync
```

O `uv sync` cria o `.venv` e instala as dependências do `pyproject.toml` (versões travadas no `uv.lock`).

### Dados (somente para retreinar ou avaliar)
O modelo já está em `models/campeao_otimizado.pkl`, e rodar a API **não** exige o dataset. Para avaliar o modelo ou rodar o notebook, baixe os dados em `data/raw/`:

```bash
uv run python -m src.data.download_data
```

## Executando a API

```bash
uv run uvicorn src.api:app --port 8000
```

- Documentação interativa (Swagger): http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Exemplo de predição, com a API rodando em outro terminal:

```bash
uv run python -m scripts.try_predict
```

Ou pelo Swagger (`POST /predict` → *Try it out*), que já vem com um payload de exemplo. Resposta:

```json
{"churn_propensity": 0.78, "high_risk": true}
```

| Campo | Descrição |
|---|---|
| `churn_propensity` | Score de risco de 0 a 1 (usar para ranquear; não é probabilidade calibrada) |
| `high_risk` | `true` se `churn_propensity` ≥ 0,5 |

Os campos de entrada usam os nomes das colunas do dataset (`"Tenure Months"`, `"Contract"` etc.) e estão descritos no schema em `/docs`.

## Testes e qualidade

```bash
uv run pytest -v
uv run ruff check src tests scripts
```

Reproduzir as métricas do modelo versionado:

```bash
uv run python -m scripts.evaluate_model
```

## Estrutura do projeto

```
├── data/                  <- raw/ (versionado com DVC), interim/, processed/, external/
├── docs/                  <- Model Card e documentação
├── models/                <- Modelo campeão (campeao_otimizado.pkl)
├── notebooks/             <- 01_eda_baseline.ipynb (EDA, baseline, CV, Optuna, campeão)
├── scripts/               <- evaluate_model.py, try_predict.py
├── src/
│   ├── api.py             <- API FastAPI (/health, /predict)
│   ├── config.py          <- Caminhos, seeds e colunas descartadas
│   ├── dataset.py         <- Leitura, limpeza e split treino/teste
│   ├── preprocessing.py   <- ColumnTransformer e Pipeline
│   ├── predict.py         <- ChurnPredictor (carrega o modelo e prediz)
│   └── data/download_data.py
├── tests/                 <- Testes pytest (dados, pré-processamento, predição, API)
├── pyproject.toml / uv.lock
└── README.md
```

## Reprodutibilidade
- Seed `42` em split, validação cruzada, modelos e otimização (`src/config.py`).
- Split 80/20 estratificado; seleção de modelo por validação cruzada de 5 folds somente no treino.
- Variáveis de ambiente opcionais: `CHURN_MODEL_PATH` e `CHURN_METADATA_PATH` sobrescrevem o caminho do modelo.

## Decisões técnicas
- **Vazamento removido:** `Churn Label`, `Churn Reason` e `Churn Score`.
- **Métricas:** ROC-AUC e PR-AUC (ranqueamento), F1 e recall (classe de churn, desbalanceada em 26,5%).
- **Modelos comparados:** Regressão Logística, Random Forest e MLP (`MLPClassifier`), com Optuna para os hiperparâmetros.
- **Campeão:** Random Forest, escolhido pelo melhor F1 e recall. O ganho de ROC-AUC sobre o baseline é pequeno (ver Model Card).
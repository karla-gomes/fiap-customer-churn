# Tech Challenge Fase 1: Previsão de Churn

Pipeline preditivo de churn para uma operadora de telecomunicações: EDA, comparação de modelos clássicos (Regressão Logística, Random Forest e MLP, com Scikit-Learn) e modelo campeão servido via API REST (FastAPI). Projeto da Fase 1 da pós tech em Engenharia de Machine Learning da FIAP.

## ML Canvas

| Bloco | Descrição |
|---|---|
| **Problema e contexto** | Uma operadora de telecomunicações perde clientes em ritmo acelerado. A diretoria precisa identificar com antecedência quem tem risco de cancelar. |
| **Proposta de valor** | Reduzir a taxa de churn trimestral, direcionando as ações de retenção para os clientes de maior risco, em vez de abordar toda a base. |
| **Stakeholders** | Diretoria (patrocínio), equipe de dados/TI (modelo e API), CRM (campanhas de retenção) e atendimento (contato com o cliente). |
| **Tarefa de ML** | Classificação binária: o cliente cancela no trimestre (`Churn Value` = 1) ou permanece (0). A saída é uma propensão de 0 a 1. |
| **Decisão suportada** | Priorizar quais clientes recebem ação de retenção. A propensão ranqueia; a decisão final e o limiar são do negócio. |
| **Fonte de dados** | IBM Telco Customer Churn (Kaggle `yeanzc/telco-customer-churn-ibm-dataset`): 7.043 clientes, 26,5% de churn, Califórnia (EUA). |



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
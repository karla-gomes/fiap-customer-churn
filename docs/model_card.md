# Model Card — Churn Telco (RandomForest)

| | |
|---|---|
| **Versão** | 0.1.0 (03/11/2026) |
| **Autoria** | Karla Gomes de Souza, FIAP Pós Tech MLET, Tech Challenge Fase 1 |
| **Modelo** | `sklearn.pipeline.Pipeline`: `ColumnTransformer` (StandardScaler + OneHotEncoder) + `RandomForestClassifier` |
| **Artefato** | `models/campeao_otimizado.pkl` |
| **Hiperparâmetros** | `n_estimators=500`, `max_depth=9`, `min_samples_leaf=4`, `min_samples_split=20`, `class_weight="balanced"`, `random_state=42` |
| **Saída** | `churn_propensity` (0 a 1) e `high_risk` (propensão ≥ 0,5) |

## 1. Uso pretendido
- **Uso principal:** priorizar clientes de telecom com maior risco de cancelamento para ações de retenção (CRM e atendimento).
- **Usuários:** equipes de dados, CRM e atendimento.
- **Fora de escopo:** decisões automáticas com efeito adverso sobre o cliente (negar serviço, alterar preço ou crédito), uso em outras operadoras, países ou períodos sem revalidação, e uso da propensão como probabilidade calibrada.

## 2. Dados
- **Fonte:** IBM Telco Customer Churn (Kaggle `yeanzc/telco-customer-churn-ibm-dataset`). São 7.043 clientes, uma linha por cliente, com 26,5% de churn.
- **Cobertura:** clientes da Califórnia (EUA), de um único trimestre. É um dataset de exemplo da IBM.
- **Divisão:** 80/20 estratificada, `random_state=42` (5.634 treino e 1.409 teste, com 374 churns no teste). A seleção de modelos usou validação cruzada estratificada de 5 folds só no treino.
- **Removidas:** identificadores e geografia (`CustomerID`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Country`, `State`, `Count`) e vazamento (`Churn Label`, `Churn Reason`, `Churn Score`).
- **Features usadas (20):** perfil (`Gender`, `Senior Citizen`, `Partner`, `Dependents`), serviços contratados, `Contract`, `Paperless Billing`, `Payment Method`, `Tenure Months`, `Monthly Charges`, `Total Charges` e `CLTV`.

## 3. Performance (conjunto de teste, limiar 0,5)

| Métrica | Baseline (Reg. Logística) | **Campeão (RandomForest)** |
|---|---|---|
| ROC-AUC | 0,8465 | **0,8522** |
| PR-AUC | 0,6412 | **0,6634** |
| F1 | 0,6149 | **0,6377** |
| Recall | 0,7727 | **0,7861** |
| Precision | 0,5106 | **0,5365** |
| Acurácia | 0,7431 | **0,7630** |

Matriz de confusão do campeão: 294 verdadeiros positivos, 80 falsos negativos, 254 falsos positivos e 781 verdadeiros negativos.

**Validação cruzada (5 folds, treino, hiperparâmetros iniciais):**

| Modelo | ROC-AUC | F1 |
|---|---|---|
| Random Forest | 0,8588 ± 0,0080 | 0,6533 ± 0,0151 |
| MLP | 0,8584 ± 0,0112 | 0,6048 ± 0,0255 |
| Regressão Logística | 0,8580 ± 0,0123 | 0,6396 ± 0,0196 |

A otimização com Optuna (50 trials, ROC-AUC e F1 em CV) escolheu o RandomForest.

**Leitura honesta:** o ganho sobre a Regressão Logística é pequeno (+0,006 de ROC-AUC no teste) e está dentro do desvio-padrão da CV. O RandomForest foi escolhido pelo melhor F1 e recall, não por uma diferença clara de ranqueamento. O modelo também apresenta sobreajuste: ROC-AUC 0,911 no treino contra 0,852 no teste.

Reprodução: `uv run python -m scripts.evaluate_model`.

## 4. Limitações
- **Não calibrado:** com `class_weight="balanced"`, a propensão média prevista no teste é 0,40 contra 26,5% de churn real, e 38,9% dos clientes são sinalizados. Use `churn_propensity` para **ranquear**, não como probabilidade.
- **Limiar fixo em 0,5:** a precisão é de 54%, ou seja, cerca de 1 em cada 2 alertas é falso. O limiar deve ser definido pelo custo da ação de retenção contra o valor do cliente retido.
- **Generalização:** os dados são de um trimestre e de um estado. Sazonalidade, mudança de preços ou de concorrência exigem monitoramento de drift e retreino periódico.
- **Colinearidade:** `Total Charges` tem associação de 0,89 com `Tenure Months` (Spearman). O modelo usa as duas, então a importância individual de cada uma é ambígua.
- **`CLTV`:** é calculado por um modelo corporativo cuja origem não está documentada. Se ele embute informação de churn, há risco de vazamento oculto.
- **Entrada:** o schema da API valida as categorias vistas no treino, e valores novos são rejeitados (422). `Total Charges` em branco deve ser enviado como `0`.
- **Sem explicabilidade individual** na API.

## 5. Possíveis vieses (teste, por subgrupo)

| Grupo | N | Churn real | Sinalizados | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Gender=Female | 687 | 28,1% | 38,3% | 0,762 | 0,855 |
| Gender=Male | 722 | 25,1% | 39,5% | 0,812 | 0,851 |
| Senior Citizen=No | 1187 | 23,3% | 34,2% | 0,746 | 0,856 |
| Senior Citizen=Yes | 222 | 44,1% | 64,0% | 0,898 | 0,781 |
| Dependents=Yes | 326 | 8,3% | 9,8% | 0,481 | 0,904 |
| Contract=Month-to-month | 773 | 42,6% | 68,8% | 0,875 | 0,757 |
| Contract=One year | 300 | 12,0% | 5,3% | 0,167 | 0,798 |
| Contract=Two year | 336 | 2,7% | 0,0% | 0,000 | 0,812 |

- **Atributos sensíveis como feature:** `Gender` e `Senior Citizen` entram no modelo. O gênero tem taxa de sinalização parecida entre os grupos (38,3% e 39,5%), mas o recall difere em cerca de 5 p.p. (0,762 contra 0,812). Com ~700 clientes por grupo, essa diferença não é conclusiva. Recomenda-se um teste de ablação sem `Gender`.
- **Idosos:** são sinalizados com o dobro da frequência (64% contra 34%). Parte disso reflete a taxa real de churn (44% contra 23%), mas o ranqueamento é pior para esse grupo (ROC-AUC 0,781 contra 0,856). Ações de retenção baseadas só no modelo podem tratar idosos de forma diferente.
- **Contrato:** o modelo concentra os alertas em contratos mensais. Clientes de contrato de 1 ou 2 anos praticamente nunca são sinalizados (recall de 0,167 e 0,000), então o churn raro nesses grupos passa despercebido.
- **Clientes com dependentes:** recall de 0,481, ou seja, mais de metade dos churns desse grupo não é detectada.
- **Representatividade:** só há dados da Califórnia. Não é possível avaliar viés regional ou socioeconômico.

## 6. Recomendações
1. Usar a propensão para priorização (ex.: top-N por capacidade da equipe de retenção), com limiar escolhido por custo-benefício.
2. Reavaliar sem `Gender` e monitorar recall e taxa de alertas por subgrupo (idade, contrato, dependentes).
3. Manter revisão humana em qualquer ação que afete condições comerciais do cliente.
4. Monitorar drift das features e do churn real, e retreinar periodicamente.
5. Validar a origem do `CLTV` antes de uso em produção.

## 7. Reprodutibilidade
Seed 42 (split, CV, modelos e Optuna), dependências em `pyproject.toml` / `uv.lock` e dataset versionado via DVC (`data/raw/Telco_customer_churn.xlsx.dvc`). O treinamento está no notebook `notebooks/01_eda_baseline.ipynb` (seções 9 a 12).
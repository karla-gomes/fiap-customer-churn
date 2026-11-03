import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Carrega variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# =====================================================================
# 1. DIRETÓRIOS E CAMINHOS
# =====================================================================
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Nomes de arquivos alinhados com o salvamento do Notebook V3
MODEL_FILENAME = "campeao.pkl"
METADATA_FILENAME = "campeao_metadata.json"
DATA_RAW_PATH = RAW_DATA_DIR / "Telco_customer_churn.xlsx"

# =====================================================================
# 2. CONFIGURAÇÕES DE MODELAGEM E REPRODUTIBILIDADE
# =====================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COL = "Churn Value"

# =====================================================================
# 3. PRÉ-PROCESSAMENTO E LIMPEZA
# =====================================================================
# Colunas identificadas na EDA com variância zero, identificadores, 
# alta cardinalidade geográfica ou vazamento de dados (leakage).
COLUMNS_TO_DROP = [
    "Count", 
    "Country", 
    "State", 
    "CustomerID", 
    "City", 
    "Lat Long", 
    "Zip Code", 
    "Latitude", 
    "Longitude", 
    "Churn Label", 
    "Churn Score", 
    "Churn Reason",
    'Total Charges', 
    'Phone Service'
]

# Variável adicionada para evitar erro de importação no dataset.py
FEATURES_TO_DROP = []

# =====================================================================
# 4. FUNÇÕES UTILITÁRIAS
# =====================================================================
def get_model_path() -> Path:
    """Caminho do modelo; sobrescrevível por CHURN_MODEL_PATH (lido na hora da chamada)."""
    return Path(os.getenv("CHURN_MODEL_PATH", MODELS_DIR / MODEL_FILENAME))

def get_metadata_path() -> Path:
    default = get_model_path().with_name(METADATA_FILENAME)
    return Path(os.getenv("CHURN_METADATA_PATH", default))
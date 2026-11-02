import shutil
from pathlib import Path
import kagglehub


def get_project_root() -> Path:
    """Localiza a raiz do projeto procurando por arquivos marcadores (.git ou .dvc)."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / ".git").exists() or (parent / ".dvc").exists():
            return parent
    # Fallback para o diretório de execução atual
    return Path.cwd()


PROJECT_ROOT = get_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_IDENTIFIER = "yeanzc/telco-customer-churn-ibm-dataset"


def download_dataset_from_kaggle(dataset_name: str) -> Path:
    """Faz o download da versão mais recente do dataset via kagglehub e devolve o caminho do cache."""
    print(f"Baixando dataset '{dataset_name}' via kagglehub...")
    cache_path = kagglehub.dataset_download(dataset_name)
    print(f"Download concluído no cache temporário: {cache_path}")
    return Path(cache_path)


def copy_files_to_raw(source_path: Path, destination_path: Path = RAW_DATA_DIR) -> list[Path]:
    """Copia os arquivos do cache temporário para a pasta data/raw oficial do repositório."""
    destination_path.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []

    print(f"Destino dos dados: {destination_path.resolve()}")

    for file_path in source_path.glob("*"):
        if file_path.is_file():
            target_file = destination_path / file_path.name
            shutil.copy2(file_path, target_file)
            copied_files.append(target_file)
            print(f"Arquivo sincronizado com sucesso: {target_file.name}")

    return copied_files


def main() -> None:
    """Executa o pipeline de obtenção e posicionamento dos dados brutos."""
    print(f"Raiz detectada do projeto: {PROJECT_ROOT}")
    downloaded_cache_dir = download_dataset_from_kaggle(DATASET_IDENTIFIER)
    copy_files_to_raw(downloaded_cache_dir, RAW_DATA_DIR)


if __name__ == "__main__":
    main()
import os
import zipfile
import pandas as pd
import requests
from abc import ABC, abstractmethod
from src.Get_Logging_Config import get_logger
from src.config import Settings

logger = get_logger(__name__)

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, source_path: str) -> pd.DataFrame:
        pass

class APIIngestor(DataIngestor):
    def ingest(self, source_path: str) -> pd.DataFrame:
        logger.info(f"Fetching data from API endpoint: {source_path}")
        try:
            response = requests.get(source_path)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Received {len(data)} records from API")
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            logger.exception(f"Failed to fetch data from API: {source_path}")
            raise

class CSVIngestor(DataIngestor):
    def ingest(self, source_path: str) -> pd.DataFrame:
        if not os.path.exists(source_path):
            logger.error(f"CSV file not found at {source_path}")
            raise FileNotFoundError(f"CSV file not found at {source_path}")
        logger.info(f"Reading CSV file {source_path}")
        return pd.read_csv(source_path)

class ZIPIngestor(DataIngestor):
    def ingest(self, source_path: str) -> pd.DataFrame:
        if not os.path.exists(source_path):
            logger.error(f"ZIP file not found at {source_path}")
            raise FileNotFoundError(f"ZIP file not found at {source_path}")

        extract_dir = os.path.dirname(source_path)
        logger.info(f"Extracting ZIP file {source_path} into {extract_dir}")
        with zipfile.ZipFile(source_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error("No CSV file found in extracted archive")
            raise FileNotFoundError("No CSV file found in archive")
        if len(csv_files) > 1:
            logger.warning("Multiple CSV files found, using the first one")

        csv_path = os.path.join(extract_dir, csv_files[0])
        logger.info(f"Reading extracted CSV file {csv_path}")
        return pd.read_csv(csv_path)

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(source_path: str) -> DataIngestor:
        if source_path.lower().startswith(('http://', 'https://')):
            logger.debug("Using APIIngestor")
            return APIIngestor()

        ext = os.path.splitext(source_path)[1].lower()
        if ext == ".zip":
            logger.debug("Using ZIPIngestor")
            return ZIPIngestor()
        elif ext == ".csv":
            logger.debug("Using CSVIngestor")
            return CSVIngestor()

        logger.error(f"No ingestor available for {source_path}")
        raise ValueError(f"No ingestor available for {source_path}")

if __name__ == "__main__":
    api_train_url = "http://127.0.0.1:8000/train"
    api_test_url = "http://127.0.0.1:8000/test"

    try:
        ingestor = DataIngestorFactory.get_data_ingestor(api_train_url)
        df_train = ingestor.ingest(api_train_url)
        logger.info(f"Training DataFrame shape: {df_train.shape}")

        df_test = ingestor.ingest(api_test_url)
        logger.info(f"Test DataFrame shape: {df_test.shape}")

    except Exception:
        logger.exception("Data ingestion failed")

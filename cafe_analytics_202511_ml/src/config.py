from google.cloud import bigquery
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_id: str = "jr-data-training"
    dataset: str = "dbt_medallion_dev_gold"          # e.g., 'analytics_gold'
    model_name: str = "fact_order_items"        # e.g., 'dim_sales_summary'
    test_size: float = 0.2
    random_state: int = 42

settings = Settings()

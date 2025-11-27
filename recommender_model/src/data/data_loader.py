"""
Data Loader
===========
Loads raw transaction data from various sources.

Refactored from: A_data_preparation.py
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RawData:
    """Container for raw loaded data."""
    transactions: pd.DataFrame
    file_path: str
    n_rows: int
    n_customers: int
    n_orders: int
    n_products: int
    date_range: tuple


class DataLoader:
    """
    Loads transaction data from CSV, database, or cloud storage.
    
    Supported sources:
    - Local CSV files
    - Azure Blob Storage (with azure-storage-blob)
    - PostgreSQL (with psycopg2)
    
    Example:
        >>> loader = DataLoader()
        >>> raw_data = loader.load_csv("data/raw/transactions.csv")
        >>> print(f"Loaded {raw_data.n_rows} rows")
    """
    
    def __init__(self):
        logger.info("DataLoader initialized")

        
    
    def load_csv(
        self, 
        file_path: Union[str, Path],
        date_columns: list = None
    ) -> RawData:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            date_columns: Columns to parse as dates
            
        Returns:
            RawData object with loaded transactions
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Default date columns for cafe data
        if date_columns is None:
            date_columns = ['order_date', 'first_order_date', 'last_order_date']
        
        # Load CSV
        df = pd.read_csv(file_path, parse_dates=date_columns)
        
        # Basic stats
        n_rows = len(df)
        n_customers = df['customer_id'].nunique()
        n_orders = df['order_id'].nunique()
        
        # Create product key (name + variant)
        df['product'] = df['product_name'] + ' (' + df['product_variant'].fillna('Regular') + ')'
        n_products = df['product'].nunique()
        
        # Date range
        date_range = (df['order_date'].min(), df['order_date'].max())
        
        logger.info(f"Loaded {n_rows:,} rows")
        logger.info(f"  Customers: {n_customers:,}")
        logger.info(f"  Orders: {n_orders:,}")
        logger.info(f"  Products: {n_products:,}")
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
        
        return RawData(
            transactions=df,
            file_path=str(file_path),
            n_rows=n_rows,
            n_customers=n_customers,
            n_orders=n_orders,
            n_products=n_products,
            date_range=date_range
        )
    
    
    def load_from_bigquery(
        self,
        project_id: str,
        query: str,
        credentials_path: Optional[str] = None
    ) -> RawData:
        """
        Load transaction data from Google BigQuery using a SQL query.

        Args:
            project_id: GCP Project ID
            query: SQL query to execute
            credentials_path: Optional path to a service account JSON key file.
                              If not provided, uses environment default credentials.

        Returns:
            RawData object with loaded transactions
        """
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "Install BigQuery dependencies: pip install google-cloud-bigquery"
            )

        logger.info("Loading data from BigQuery...")

        # Authenticate client
        if credentials_path:
            logger.info(f"Using credentials from: {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client = bigquery.Client(project=project_id, credentials=credentials)
        else:
            logger.info("Using default GCP credentials")
            client = bigquery.Client(project=project_id)

        # Run query
        logger.info("Executing SQL query on BigQuery")
        df = client.query(query).to_dataframe()

        # Parse typical date column if present
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"])

        # Product key (same as CSV/database loader)
        if "product_name" in df.columns and "product_variant" in df.columns:
            df["product"] = (
                df["product_name"]
                + " ("
                + df["product_variant"].fillna("Regular")
                + ")"
            )

        # Compute stats
        n_rows = len(df)
        n_customers = df["customer_id"].nunique() if "customer_id" in df else None
        n_orders = df["order_id"].nunique() if "order_id" in df else None
        n_products = df["product"].nunique() if "product" in df else None

        # Date range
        if "order_date" in df.columns:
            date_range = (df["order_date"].min(), df["order_date"].max())
        else:
            date_range = (None, None)

        logger.info(f"Loaded {n_rows:,} rows from BigQuery")
        logger.info(f"  Customers: {n_customers}")
        logger.info(f"  Orders: {n_orders}")
        logger.info(f"  Products: {n_products}")
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")

        return RawData(
            transactions=df,
            file_path=f"bigquery://{project_id}",
            n_rows=n_rows,
            n_customers=n_customers,
            n_orders=n_orders,
            n_products=n_products,
            date_range=date_range
        )

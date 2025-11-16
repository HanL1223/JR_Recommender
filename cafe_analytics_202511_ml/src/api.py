from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.config import settings
from src.Get_Logging_Config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="BigQuery ML API", version="1.0")

# Initialize BigQuery client
bq_client = bigquery.Client(project=settings.project_id)


def load_and_split_data():
    """Load data from BigQuery and split into train/test sets."""
    try:
        query = f"""
        SELECT *
        FROM `{settings.project_id}.{settings.dataset}.{settings.model_name}`
        """
        logger.info(f"Executing BigQuery: {query}")
        df = bq_client.query(query).to_dataframe()
        logger.info(f"BigQuery data loaded successfully. Shape: {df.shape}")

        if df.empty:
            raise ValueError("BigQuery returned no data.")

        train_df, test_df = train_test_split(
            df,
            test_size=settings.test_size,
            random_state=settings.random_state
        )
        logger.info(f"Train/Test split complete: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    except Exception as e:
        logger.exception(f"Error loading data from BigQuery: {e}")
        raise RuntimeError("Data Ingestion Failure") from e


@app.on_event("startup")
async def startup_event():
    """Load BigQuery data at startup and store in app state."""
    logger.info("Loading data from BigQuery...")
    train_df, test_df = load_and_split_data()
    app.state.train_df = train_df
    app.state.test_df = test_df
    logger.info("Data successfully loaded into app state.")


def clean_dataframe(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Clean dataframe for JSON serialization and limit rows."""
    df_sample = df.head(limit).copy()
    # Replace infinities and NaNs
    df_sample.replace([np.inf, -np.inf], 0.0, inplace=True)
    df_sample.fillna(0.0, inplace=True)
    # Convert timestamps to string (ISO format)
    for col in df_sample.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df_sample[col] = df_sample[col].astype(str)
    return df_sample


@app.get("/train")
async def get_train_data(limit: int = 5):
    """Return sample of training data."""
    df = getattr(app.state, "train_df", None)
    if df is None:
        return JSONResponse(status_code=500, content={"error": "Training data not loaded"})
    df_sample = clean_dataframe(df, limit)
    return JSONResponse(content=df_sample.to_dict(orient="records"))


@app.get("/test")
async def get_test_data(limit: int = 5):
    """Return sample of test data."""
    df = getattr(app.state, "test_df", None)
    if df is None:
        return JSONResponse(status_code=500, content={"error": "Test data not loaded"})
    df_sample = clean_dataframe(df, limit)
    return JSONResponse(content=df_sample.to_dict(orient="records"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

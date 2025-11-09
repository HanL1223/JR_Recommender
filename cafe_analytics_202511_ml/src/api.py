from fastapi import FastAPI
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
import pandas as pd
from contextlib import asynccontextmanager
from src.config import settings
from src.Get_Logging_Config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="BigQuery ML API", version="1.0")

# Initialize BigQuery client
bq_client = bigquery.Client(project=settings.project_id)


def load_and_split_data():
    """
    Load data from BigQuery and split into train/test sets.
    """
    try:
        # SQL query dynamically built from settings
        query = f"""
        SELECT *
        FROM `{settings.project_id}.{settings.dataset}.{settings.model_name}`
        """
        logger.info(f"Executing BigQuery: {query}")

        df = bq_client.query(query).to_dataframe()
        logger.info(f"BigQuery data loaded. Shape: {df.shape}")

        if df.empty:
            raise ValueError("BigQuery returned no data.")

        # Split into train/test sets
        train_df, test_df = train_test_split(
            df,
            test_size=settings.test_size,
            random_state=settings.random_state
        )

        logger.info("Train/test split completed.")
        return train_df, test_df

    except Exception as e:
        logger.exception(f"Error loading data from BigQuery: {e}")
        raise RuntimeError("Data Ingestion Failure") from e



@app.on_event("startup")
async def startup_event():
    # Load BigQuery data at startup
    client = bigquery.Client(project=settings.project_id)
    query = f"SELECT * FROM `{settings.project_id}.{settings.dataset}.{settings.model_name}`"
    df = client.query(query).to_dataframe()

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=settings.test_size, random_state=settings.random_state)

    # Store in app state
    app.state.train_df = train_df
    app.state.test_df = test_df
    print(f"Data loaded: train {len(train_df)} rows, test {len(test_df)} rows")

@app.get("/train")
async def get_train_data(limit: int = 5):
    df = getattr(app.state, "train_df", None)
    if df is None:
        return {"error": "Training data not loaded"}, 500
    df_sample = df.head(limit).copy()
    numeric_cols = df_sample.select_dtypes(include=['number']).columns
    df_sample[numeric_cols] = df_sample[numeric_cols].astype(float).replace([np.inf, -np.inf], 0.0)
    return df_sample.to_dict(orient="records")

@app.get("/test")
async def get_test_data(limit: int = 5):
    df = getattr(app.state, "test_df", None)
    if df is None:
        return {"error": "Test data not loaded"}, 500
    df_sample = df.head(limit).copy()
    numeric_cols = df_sample.select_dtypes(include=['number']).columns
    df_sample[numeric_cols] = df_sample[numeric_cols].astype(float).replace([np.inf, -np.inf], 0.0)
    return df_sample.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

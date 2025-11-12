import os
from datetime import datetime
from cosmos import DbtDag, ProjectConfig, ProfileConfig, ExecutionConfig
from cosmos.profiles import GoogleCloudOauthProfileMapping

DBT_PROJECT_PATH = f"{os.getenv('AIRFLOW_HOME')}/dags/dbt/dbt_cafe_analytics"
DBT_EXECUTABLE_PATH = f"{os.getenv('AIRFLOW_HOME')}/dbt_venv/bin/dbt"
PROFILE_NAME = os.getenv('PROFILE_NAME')
TARGET_NAME = os.getenv('TARGET_NAME')
PROJECT_NAME = os.getenv('GOOGLE_CLOUD_PROJECT')
DATASET_NAME = os.getenv('DATASET_NAME')
ENABLE_FULL_REFRESH = (
    os.getenv('ENABLE_FULL_REFRESH').lower() == 'true'
)

_project_config = ProjectConfig(
    dbt_project_path=DBT_PROJECT_PATH,
    install_dbt_deps=True,
)

_profile_config = ProfileConfig(
    profile_name=PROFILE_NAME,
    target_name=TARGET_NAME,
    profile_mapping=GoogleCloudOauthProfileMapping(
        conn_id="gcp_conn",  # A connection for Google Cloud needs to be created in Airflow
        profile_args={
            "project": PROJECT_NAME, "dataset": DATASET_NAME
        },
    ),
)

_execution_config = ExecutionConfig(
    dbt_executable_path=DBT_EXECUTABLE_PATH,
)

dbt_cafe_analytics_dag = DbtDag(
    project_config=_project_config,
    profile_config=_profile_config,
    execution_config=_execution_config,
    operator_args={
        "full_refresh": ENABLE_FULL_REFRESH,  # This enables the full refresh
    },
    # normal dag parameters
    schedule="@monthly",
    start_date=datetime(2025, 11, 1),
    catchup=False,
    max_active_tasks=1,
    dag_id="dbt_cafe_analytics_dag",
    default_args={"retries": 2},
)
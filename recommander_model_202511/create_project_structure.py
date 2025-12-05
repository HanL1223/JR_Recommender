import os

# Define your folder structure inside src
PROJECT_STRUCTURE = {
    "src/data_ingestion": [
        "__init__.py",
        "data_loader.py",
        "data_validator.py",
        "preprocessor.py"
    ],
    "src/features": [
        "__init__.py",
        "product_features.py",
        "customer_features.py",
        "training_data_builder.py"
    ],
    "src/models": [
        "__init__.py",
        "base_model.py",
        "baseline_models.py",
        "lightgbm_ranker.py"
    ],
    "src/training": [
        "__init__.py",
        "data_splitter.py",
        "trainer.py",
        "hyperparameter_tuning.py"
    ],
    "src/evaluation": [
        "__init__.py",
        "metrics.py"
    ],
    "src/inference": [
        "__init__.py",
        "predictor.py",
        "cold_start.py"
    ]
}

# New: Data folders
DATA_FOLDERS = [
    "data/raw",
    "data/clean"
]


def create_structure(base_path="recommender_model_202511"):
    print(f"Creating project structure under: {os.path.abspath(base_path)}")

    for folder, files in PROJECT_STRUCTURE.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f" Created folder: {folder_path}")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    if file_name == "__init__.py":
                        f.write("")  # empty
                    else:
                        f.write(f"# {file_name}\n# Auto-generated file\n")
                print(f" Created file: {file_path}")

    for folder in DATA_FOLDERS:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f" Created folder: {folder_path}")


if __name__ == "__main__":
    create_structure()

import os

# Define your folder structure
PROJECT_STRUCTURE = {
    "data": [
        "__init__.py",
        "data_loader.py",
        "data_validator.py",
        "preprocessor.py"
    ],
    "features": [
        "__init__.py",
        "product_features.py",
        "customer_features.py",
        "training_data_builder.py"
    ],
    "models": [
        "__init__.py",
        "base_model.py",
        "baseline_models.py",
        "lightgbm_ranker.py"
    ],
    "training": [
        "__init__.py",
        "data_splitter.py",
        "trainer.py",
        "hyperparameter_tuning.py"
    ],
    "evaluation": [
        "__init__.py",
        "metrics.py"
    ],
    "inference": [
        "__init__.py",
        "predictor.py",
        "cold_start.py"
    ]
}


def create_structure(base_path="recommender_model"):
    print(f"Creating project structure under: {os.path.abspath(base_path)}")

    for folder, files in PROJECT_STRUCTURE.items():
        folder_path = os.path.join(base_path, folder)

        # Create folder
        os.makedirs(folder_path, exist_ok=True)
        print(f" Created folder: {folder_path}")

        # Create files
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    if file_name == "__init__.py":
                        f.write("")   # keep empty
                    else:
                        f.write(f"# {file_name}\n# Auto-generated file\n")
                print(f" Created file: {file_path}")


if __name__ == "__main__":
    create_structure("src")

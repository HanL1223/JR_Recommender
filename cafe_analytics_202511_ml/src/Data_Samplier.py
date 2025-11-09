import pandas as pd
from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from src.Get_Logging_Config import get_logger

logger = get_logger(__name__)


class DataSampler(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Abstract method to resample dataset."""
        pass


#Oversampling Method
#Note: It can lead to the creation of noisy and irrelevant synthetic samples in regions of feature space where the minority class is highly concentrated.
class SMOTESampler(DataSampler):
    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logger.info(f"Applying SMOTE oversampling. Input size: {df.shape}")

        X, y = df.drop(columns=[target_col]), df[target_col]
        smote = SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target_col] = y_res
        logger.info(f"SMOTE complete. Output size: {df_resampled.shape}")
        logger.info(f"y Class: {df_resampled[target_col].value_counts()}")
        return df_resampled


#Undersampling
class UnderSampler(DataSampler):
    def __init__(self, random_state: int = 42, sampling_strategy: str = "auto"):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logger.info(f"Applying Random Undersampling. Input size: {df.shape}")

        X, y = df.drop(columns=[target_col]), df[target_col]
        rus = RandomUnderSampler(random_state=self.random_state,
                                 sampling_strategy=self.sampling_strategy)
        X_res, y_res = rus.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target_col] = y_res
        logger.info(f"Undersampling complete. Output size: {df_resampled.shape}")
        logger.info(f"y Class: {df_resampled[target_col].value_counts()}")
        return df_resampled




#SMOTE-ENN
#a variant of SMOTE that addresses this limitation by combining SMOTE with the Edited Nearest Neighbors (ENN) technique
class SMOTEENNSampler(DataSampler):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logger.info(f"Applying SMOTEENN hybrid resampling. Input size: {df.shape}")

        X, y = df.drop(columns=[target_col]), df[target_col]
        smoteenn = SMOTEENN(random_state=self.random_state)
        X_res, y_res = smoteenn.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target_col] = y_res
        logger.info(f"SMOTEENN complete. Output size: {df_resampled.shape}")
        logger.info(f"y Class: {df_resampled[target_col].value_counts()}")
        return df_resampled

class SamplerFactory:
    @staticmethod
    def create(sampler_type: str, **kwargs) -> DataSampler:
        sampler_type = sampler_type.lower()
        if sampler_type == "smote":
            return SMOTESampler(**kwargs)
        elif sampler_type == "undersample":
            return UnderSampler(**kwargs)
        elif sampler_type == "smoteenn":
            return SMOTEENNSampler(**kwargs)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")


if __name__ == "__main__":
    pass

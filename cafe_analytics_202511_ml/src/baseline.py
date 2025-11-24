import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------- Model Type -------------------
class ModelType(Enum):
    POPULARITY = "popularity"
    ITEM_CF = "item_cf"
    SVD = "svd"
    GRU = "gru"

# ------------------- Base Recommender -------------------
class BaseRecommender(ABC):

    def __init__(self, config: Dict, data: Dict):
        self.config = config
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self, user_idx: int, top_k: int = 10, exclude_interacted: bool = True) -> List[int]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def _validate_trained(self):
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before prediction")

# ------------------- Popularity -------------------
class PopularityRecommender(BaseRecommender):

    @property
    def model_name(self) -> str:
        return "Popularity Baseline"

    def train(self) -> None:
        self.logger.info(f"Training {self.model_name}")
        user_item_matrix = self.data['user_item_matrix']
        item_counts = np.array(user_item_matrix.sum(axis=0)).flatten()
        self.item_popularity = np.argsort(item_counts)[::-1]
        self.is_trained = True
        self.logger.info(f"{self.model_name} trained successfully")

    def predict(self, user_idx: int, top_k: int = 10, exclude_interacted: bool = True) -> List[int]:
        self._validate_trained()
        if exclude_interacted:
            user_items = np.where(self.data['user_item_matrix'].iloc[user_idx] > 0)[0]
            top_items = [i for i in self.item_popularity if i not in user_items]
        else:
            top_items = self.item_popularity
        return top_items[:top_k].tolist()

    def get_model_info(self) -> Dict:
        return {
            'model_type': self.model_name,
            'n_items': len(self.item_popularity),
            'is_trained': self.is_trained,
            'config': self.config
        }

# ------------------- ItemCF -------------------
class ItemCFRecommender(BaseRecommender):

    def __init__(self, config: Dict, data: Dict):
        super().__init__(config, data)
        self.item_similarity = None
        self.n_neighbors = config.get('n_neighbors', 50)

    @property
    def model_name(self) -> str:
        return "Item-Based Collaborative Filtering"

    def train(self) -> None:
        self.logger.info(f"Training {self.model_name} (n_neighbors={self.n_neighbors})")
        user_item_matrix = self.data['user_item_matrix'].values
        item_matrix_binary = (user_item_matrix.T > 0).astype(float)
        self.item_similarity = cosine_similarity(item_matrix_binary)
        np.fill_diagonal(self.item_similarity, 0)
        self.is_trained = True
        self.logger.info(f"{self.model_name} trained successfully")

    def predict(self, user_idx: int, top_k: int = 10, exclude_interacted: bool = True) -> List[int]:
        self._validate_trained()
        user_item_matrix = self.data['user_item_matrix'].values
        user_items = np.where(user_item_matrix[user_idx] > 0)[0]
        if len(user_items) == 0:
            pop_model = PopularityRecommender(self.config, self.data)
            pop_model.train()
            return pop_model.predict(user_idx, top_k, exclude_interacted)

        scores = np.zeros(user_item_matrix.shape[1])
        for item in user_items:
            similar_items = np.argsort(self.item_similarity[item])[::-1][:self.n_neighbors]
            scores[similar_items] += self.item_similarity[item, similar_items]

        if exclude_interacted:
            scores[user_items] = -np.inf
        recommendations = np.argsort(scores)[::-1][:top_k]
        return recommendations.tolist()

    def get_model_info(self) -> Dict:
        return {
            'model_type': self.model_name,
            'n_neighbors': self.n_neighbors,
            'similarity_shape': self.item_similarity.shape,
            'is_trained': self.is_trained,
            'config': self.config
        }

# ------------------- SVD -------------------
class SVDRecommender(BaseRecommender):

    def __init__(self, config: Dict, data: Dict):
        super().__init__(config, data)
        self.n_factors = config.get('n_factors', 50)
        self.svd_predictions = None

    @property
    def model_name(self) -> str:
        return "Matrix Factorization (SVD)"

    def train(self) -> None:
        self.logger.info(f"Training {self.model_name} (n_factors={self.n_factors})")
        matrix = self.data['user_item_matrix'].values
        matrix_centered = matrix - np.mean(matrix, axis=0)
        U, sigma, Vt = svds(matrix_centered, k=self.n_factors)
        self.svd_predictions = np.dot(U, np.diag(sigma)).dot(Vt) + np.mean(matrix, axis=0)
        self.is_trained = True
        self.logger.info(f"{self.model_name} trained successfully")

    def predict(self, user_idx: int, top_k: int = 10, exclude_interacted: bool = True) -> List[int]:
        self._validate_trained()
        user_preds = self.svd_predictions[user_idx].copy()
        if exclude_interacted:
            user_items = np.where(self.data['user_item_matrix'].iloc[user_idx] > 0)[0]
            user_preds[user_items] = -np.inf
        recommendations = np.argsort(user_preds)[::-1][:top_k]
        return recommendations.tolist()

    def get_model_info(self) -> Dict:
        return {
            'model_type': self.model_name,
            'n_factors': self.n_factors,
            'prediction_shape': self.svd_predictions.shape,
            'is_trained': self.is_trained,
            'config': self.config
        }

# ------------------- GRU -------------------
class GRURecommenderModel(BaseRecommender, nn.Module):

    def __init__(self, config: Dict, data: Dict):
        BaseRecommender.__init__(self, config, data)
        nn.Module.__init__(self)
        self.n_items = data['n_items']
        self.embedding_dim = config.get('embedding_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.n_layers = config.get('n_layers', 2)
        self.dropout = config.get('dropout', 0.3)

        self.embedding = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.n_layers,
                          dropout=self.dropout if self.n_layers > 1 else 0, batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, self.n_items)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @property
    def model_name(self) -> str:
        return "GRU Deep Learning"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        last_hidden = hidden[-1]
        out = self.dropout_layer(last_hidden)
        out = F.relu(self.bn1(self.fc1(out)))
        out = self.dropout_layer(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout_layer(out)
        return self.fc_out(out)

    def train(self) -> None:
        self.logger.info(f"Training {self.model_name}")

        class RecommendationDataset(Dataset):
            def __init__(self, sequences, targets):
                self.sequences = torch.LongTensor(sequences)
                self.targets = torch.LongTensor(targets)
            def __len__(self):
                return len(self.sequences)
            def __getitem__(self, idx):
                return self.sequences[idx], self.targets[idx]

        train_dataset = RecommendationDataset(self.data['X_train_seq'], self.data['y_train'])
        val_dataset = RecommendationDataset(self.data['X_val_seq'], self.data['y_val'])
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        epochs = self.config.get('epochs', 30)
        patience = self.config.get('patience', 10)

        for epoch in range(epochs):
            self.train_one_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate_one_epoch(val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = self.state_dict()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.load_state_dict(self.best_state)
        self.is_trained = True
        self.logger.info(f"{self.model_name} trained successfully")

    def train_one_epoch(self, loader, criterion, optimizer):
        self.train()
        total_loss = 0
        for sequences, targets in loader:
            sequences, targets = sequences.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate_one_epoch(self, loader, criterion):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for sequences, targets in loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.forward(sequences)
                total_loss += criterion(outputs, targets).item()
        return total_loss / len(loader)

    def predict(self, user_idx: int, top_k: int = 10, exclude_interacted: bool = True) -> List[int]:
        self._validate_trained()
        sequence = self.data['current_sequence']
        self.eval()
        with torch.no_grad():
            seq_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
            output = self.forward(seq_tensor)
            top_items = torch.argsort(output[0], descending=True)[:top_k].cpu().numpy()
        return top_items.tolist()

    def get_model_info(self) -> Dict:
        return {
            'model_type': self.model_name,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'config': self.config
        }

# ------------------- Factory -------------------
class RecommenderFactory:

    @staticmethod
    def create_model(model_type: ModelType, config: Dict, data: Dict) -> BaseRecommender:
        logger.info(f"Creating model: {model_type.value}")
        if model_type == ModelType.POPULARITY:
            return PopularityRecommender(config, data)
        elif model_type == ModelType.ITEM_CF:
            return ItemCFRecommender(config, data)
        elif model_type == ModelType.SVD:
            return SVDRecommender(config, data)
        elif model_type == ModelType.GRU:
            return GRURecommenderModel(config, data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_multiple_models(model_types: List[ModelType], configs: Dict[ModelType, Dict], data: Dict) -> Dict[str, BaseRecommender]:
        models = {mt.value: RecommenderFactory.create_model(mt, configs.get(mt, {}), data) for mt in model_types}
        logger.info(f"Created {len(models)} models successfully")
        return models

    @staticmethod
    def train_all_models(models: Dict[str, BaseRecommender]) -> None:
        for name, model in models.items():
            logger.info(f"Training {name}")
            model.train()
        logger.info("All models trained successfully")

# ------------------- Usage Example -------------------
if __name__ == "__main__":
    logger.info("REFACTORED: Model Factory with ABC Pattern")

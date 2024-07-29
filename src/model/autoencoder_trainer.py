import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import logging
from typing import Tuple, Dict
from autoencoder_model import AutoencoderModel
from src.utils.decorator_loggins import log_execution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    def __init__(self, df: pd.DataFrame, embedding_col: str = 'combined_embeddings',
                 label_col: str = 'numeric_label', test_size: float = 0.2,
                 random_state: int = 42, batch_size: int = 32, encoding_dim: int = 32):
        self.df = df
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.X_test = None
        self.y_test = None

    def prepare_data(self):
        logger.info("Preparando datos para el autoencoder...")

        X = np.array(self.df[self.embedding_col].tolist())
        y = self.df[self.label_col].values

        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X_resampled, y_resampled,
                                                                      test_size=self.test_size,
                                                                      random_state=self.random_state)

        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(self.y_test)

        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1. / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.batch_size,
                                       sampler=sampler)
        self.test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=self.batch_size,
                                      shuffle=False)

        logger.info(
            f"Datos preparados. Tamaño del conjunto de entrenamiento: {len(X_train)}, Tamaño del conjunto de prueba: {len(self.X_test)}")

        return self.train_loader, self.test_loader

    def create_model(self):
        input_dim = len(self.df[self.embedding_col].iloc[0])
        self.model = AutoencoderModel(input_dim, self.encoding_dim)
        logger.info(
            f"Modelo creado con dimensión de entrada {input_dim} y dimensión de codificación {self.encoding_dim}")
        return self.model

    def train_model(self, num_epochs: int = 300, learning_rate: float = 0.001):
        if self.model is None or self.train_loader is None:
            raise ValueError("El modelo no ha sido creado o los datos no han sido preparados.")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info("Iniciando entrenamiento del modelo...")
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_features, _ in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_features)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(self.train_loader):.4f}')

        logger.info("Entrenamiento completado.")

    def find_optimal_threshold(self, step: float = 0.001):
        all_losses = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in self.test_loader:
                outputs = self.model(batch_features)
                loss = nn.MSELoss(reduction='none')(outputs, batch_features)
                loss = loss.mean(axis=1)
                all_losses.extend(loss.tolist())
                all_labels.extend(batch_labels.tolist())

        thresholds = np.arange(min(all_losses), max(all_losses), step)
        f1_scores = []

        for threshold in thresholds:
            predictions = [1 if loss < threshold else 0 for loss in all_losses]
            f1 = f1_score(all_labels, predictions)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold

    def evaluate_model(self, threshold: float = None):
        if self.model is None or self.test_loader is None:
            raise ValueError("El modelo no ha sido creado o entrenado, o los datos de prueba no han sido preparados.")

        if threshold is None:
            threshold = self.find_optimal_threshold()

        self.model.eval()
        all_losses = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in self.test_loader:
                outputs = self.model(batch_features)
                loss = nn.MSELoss(reduction='none')(outputs, batch_features)
                loss = loss.mean(axis=1)
                all_losses.extend(loss.tolist())
                all_labels.extend(batch_labels.tolist())

        all_predictions = [1 if loss < threshold else 0 for loss in all_losses]

        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc_roc = roc_auc_score(all_labels, [-loss for loss in all_losses])
        cm = confusion_matrix(all_labels, all_predictions)

        logger.info(f"Evaluación con umbral óptimo: {threshold:.4f}")
        logger.info(f"Precisión: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        logger.info(f"Matriz de Confusión:\n{cm}")

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": cm
        }
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from src.utils.decorator_loggins import log_execution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Modelo {model_name} cargado exitosamente.")

    def generate_embeddings(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        logger.info("Generando embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info("Embeddings generados exitosamente.")
        return embeddings

    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"La columna {text_column} no existe en el DataFrame.")

        embeddings = self.generate_embeddings(df[text_column])
        embeddings_list = embeddings.tolist()
        df['combined_embeddings'] = embeddings_list
        logger.info(f"DataFrame procesado. Embeddings combinados en una sola columna.")
        return df

    @staticmethod
    def get_embedding_dim(df: pd.DataFrame) -> int:
        if 'combined_embeddings' not in df.columns:
            raise ValueError("El DataFrame no contiene la columna 'combined_embeddings'.")
        embedding_dim = len(df['combined_embeddings'].iloc[0])
        return embedding_dim
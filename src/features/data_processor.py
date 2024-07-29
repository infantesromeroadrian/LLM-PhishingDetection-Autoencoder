import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import logging
from src.utils.decorator_loggins import log_execution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Descarga de recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def process_data(self) -> 'DataProcessor':
        logger.info("Iniciando procesamiento de datos...")
        self.df['cleaned_text'] = self.df['Message_body'].apply(self.clean_text)
        logger.info("Texto limpiado exitosamente.")
        self.df['numeric_label'] = (self.df['Label'] == 'Spam').astype(int)
        logger.info("Etiquetas convertidas a valores numéricos.")
        self.df.drop(['Message_body', 'Label'], axis=1, inplace=True)
        logger.info("Columnas originales eliminadas.")
        logger.info("Procesamiento de datos completado.")
        return self

    def get_processed_data(self) -> pd.DataFrame:
        if 'cleaned_text' not in self.df.columns or 'numeric_label' not in self.df.columns:
            raise ValueError("Los datos aún no han sido procesados. Ejecute process_data() primero.")
        return self.df
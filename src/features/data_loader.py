import pandas as pd
import logging
from functools import wraps
from typing import Optional
from src.utils.decorator_loggins import log_execution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Ejecutando {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finalizado {func.__name__}")
        return result

    return wrapper


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None

    @log_execution
    def load_data(self, encoding: str = 'utf-8') -> None:
        encodings_to_try = [encoding, 'iso-8859-1', 'latin1', 'cp1252']

        for enc in encodings_to_try:
            try:
                self.data = pd.read_csv(self.file_path, encoding=enc)
                logger.info(f"Datos cargados exitosamente desde {self.file_path} con codificación {enc}")
                return
            except UnicodeDecodeError:
                logger.warning(f"No se pudo cargar el archivo con la codificación {enc}. Probando otra...")
            except Exception as e:
                logger.error(f"Error al cargar los datos: {str(e)}")
                raise

        logger.error("No se pudo cargar el archivo con ninguna de las codificaciones probadas.")
        raise ValueError("No se pudo determinar la codificación correcta del archivo.")

    @log_execution
    def get_info(self) -> None:
        if self.data is not None:
            logger.info("Información del dataset:")
            print(self.data.info())
        else:
            logger.warning("No hay datos cargados. Ejecute load_data() primero.")

    @log_execution
    def check_nulls(self) -> pd.DataFrame:
        if self.data is not None:
            null_info = self.data.isnull().sum().reset_index()
            null_info.columns = ['Columna', 'Nulos']
            null_info['Porcentaje'] = (null_info['Nulos'] / len(self.data)) * 100
            logger.info("Información de valores nulos:")
            print(null_info)
            return null_info
        else:
            logger.warning("No hay datos cargados. Ejecute load_data() primero.")
            return pd.DataFrame()
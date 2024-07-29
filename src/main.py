import os
import sys

# Añade el directorio raíz del proyecto al PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.gradio_interface import PhishingDetectorInterface
from src.utils.decorator_loggins import log_execution, log_error, get_logger

logger = get_logger(__name__)

@log_execution
@log_error
def main():
    autoencoder_path = os.path.join(project_root, "models", "autoencoder_model.pth")
    distilbert_path = 'distilbert-base-uncased'  # Esto descargará el modelo preentrenado

    if not os.path.exists(autoencoder_path):
        logger.error(f"El modelo autoencoder no se encuentra en la ruta especificada: {autoencoder_path}")
        return

    logger.info("Iniciando la interfaz de detección de phishing...")

    interface = PhishingDetectorInterface(autoencoder_path, distilbert_path)
    interface.launch()

if __name__ == "__main__":
    main()

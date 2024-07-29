import logging
from functools import wraps
import time

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_execution(func):
    """
    Decorador para registrar la ejecución de una función.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Ejecutando {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Finalizado {func.__name__}. Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        return result
    return wrapper

def log_error(func):
    """
    Decorador para registrar errores que ocurran durante la ejecución de una función.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en {func.__name__}: {str(e)}")
            raise
    return wrapper

def get_logger(name):
    """
    Obtiene un logger configurado para un módulo específico.
    """
    return logging.getLogger(name)

# Ejemplo de uso
if __name__ == "__main__":
    @log_execution
    @log_error
    def example_function():
        logger.info("Esta es una función de ejemplo")
        # Simular un error
        raise ValueError("Este es un error de ejemplo")

    try:
        example_function()
    except ValueError:
        pass  # Ignoramos el error para este ejemplo
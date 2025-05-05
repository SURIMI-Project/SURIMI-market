import logging
import sys


def configure_logging():
    """Configures the logger and redirects print statements to the logger.
    Used for opentelemetry tracing.
    """
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("SURIMI-market")

    # Redirect print statements to the logger
    class PrintToLogger:
        def write(self, message):
            if message.strip():  # Avoid logging empty lines
                logger.info(message.strip())

        def flush(self):
            pass  # Required for compatibility with sys.stdout and sys.stderr

    # Redirect stdout and stderr to the logger
    sys.stdout = PrintToLogger()
    sys.stderr = PrintToLogger()

    return logger
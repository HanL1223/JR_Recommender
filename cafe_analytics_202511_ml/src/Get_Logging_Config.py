import logging
import sys


def get_logger(name: str = __name__) -> logging.Logger:
    """Configure and return a logger instance for the given module name."""

    # Create a custom logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO) 

        # Formatter with timestamp, level, and module name
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler("data_ingestion.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

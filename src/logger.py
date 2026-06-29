import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("PlacementPrepAI")



LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger.setLevel(getattr(logging, LOG_LEVEL))

# Avoid duplicate handlers
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Log to terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
    "logs/placementprep.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
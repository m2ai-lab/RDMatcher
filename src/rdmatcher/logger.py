import logging
import os
from typing import Literal, Optional

class RDMatchLogger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RDMatchLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not RDMatchLogger._initialized:
            self._setup_base_logger()
            RDMatchLogger._initialized = True

    def _setup_base_logger(self):
        """Initialize base logging configuration"""
        self.base_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.base_formatter)

    def get_logger(
        self,
        name: str,
        level: Literal["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        log_file: Optional[str] = None,
        console: bool = True
    ) -> logging.Logger:
        """
        Get a logger with specified name, level, and output destination.

        Args:
            name: Name of the logger
            level: Logging level
            log_file: Path to log file. If provided, logs are written here.
            console: If True, logs are written to notebook/console. Set False to silence output.
        """
        logger = logging.getLogger(name)

        level_val = getattr(logging, level.upper())
        logger.setLevel(level_val)

        if logger.handlers:
            logger.handlers.clear()

        logger.propagate = False

        if console:
            logger.addHandler(self.stream_handler)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.base_formatter)
            logger.addHandler(file_handler)

        return logger

    def set_level(self, logger_name: str, level: str):
        """Change logging level for a specific logger"""
        logger = logging.getLogger(logger_name)
        level_val = getattr(logging, level.upper())
        logger.setLevel(level_val)

rd_logger = RDMatchLogger()

def rdlogger(
    name: str,
    level: Literal["INFO", "DEBUG"] = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    return rd_logger.get_logger(name, level, log_file, console)

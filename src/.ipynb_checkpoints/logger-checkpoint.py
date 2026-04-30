import logging
import os
from typing import Literal



class EpiMatchLogger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EpiMatchLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not EpiMatchLogger._initialized:
            self._setup_base_logger()
            EpiMatchLogger._initialized = True
            
    def _setup_base_logger(self):
        """Initialize base logging configuration"""
        self.base_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.base_formatter)
    
    def get_logger(
        self, 
        name: str, 
        level: Literal["INFO", "DEBUG"] = "INFO"
    ) -> logging.Logger:
        """
        Get a logger with specified name and level.
        
        Args:
            name: Name of the logger (typically __name__ from calling module)
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        logger = logging.getLogger(name)
        
        # Set level
        level = getattr(logging, level.upper())
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            logger.addHandler(self.handler)
            logger.propagate = False
            
        return logger
    
    def set_level(self, logger_name: str, level: str):
        """Change logging level for a specific logger"""
        logger = logging.getLogger(logger_name)
        level = getattr(logging, level.upper())
        logger.setLevel(level)



epi_logger = EpiMatchLogger()

def epilogger(name: str, level: Literal["INFO", "DEBUG"] = "INFO") -> logging.Logger:
    return epi_logger.get_logger(name, level)
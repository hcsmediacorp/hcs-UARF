"""
UARF Logging System - Strukturiertes Logging mit Leveln
Production-ready Logging für alle Plattformen
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """Farbiger Formatter für Console Output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON Formatter für strukturierte Logs"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Extra Felder hinzufügen
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Exception Info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class UARFLogger:
    """
    Production-ready Logger für UARF
    
    Features:
    - Console Logging mit Farben
    - File Logging (optional)
    - JSON Format für MLflow/W&B Integration
    - Log Level Steuerung
    - Experiment Tracking Vorbereitung
    """
    
    _instance: Optional['UARFLogger'] = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = "uarf",
        level: int = logging.INFO,
        log_dir: Optional[str] = None,
        json_format: bool = False,
        console_output: bool = True,
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.name = name
        self.log_dir = log_dir
        self.json_format = json_format
        
        # Haupt-Logger erstellen
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console Handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if json_format:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter(
                    fmt='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
            
            self.logger.addHandler(console_handler)
        
        # File Handler (optional)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = Path(log_dir) / f"uarf_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter() if json_format else logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"Log file created: {log_file}")
        
        self._loggers[name] = self.logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get logger instance"""
        if name and name != self.name:
            if name not in self._loggers:
                logger = logging.getLogger(name)
                # Inherit handlers from main logger
                for handler in self.logger.handlers:
                    logger.addHandler(handler)
                logger.setLevel(self.logger.level)
                self._loggers[name] = logger
            return self._loggers[name]
        return self.logger
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra={'extra_data': kwargs} if kwargs else None)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra={'extra_data': kwargs} if kwargs else None)
    
    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra={'extra_data': kwargs} if kwargs else None)
    
    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={'extra_data': kwargs} if kwargs else None)
    
    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, extra={'extra_data': kwargs} if kwargs else None)
    
    def exception(self, msg: str, exc_info: bool = True, **kwargs):
        self.logger.exception(msg, exc_info=exc_info, extra={'extra_data': kwargs} if kwargs else None)
    
    def set_level(self, level: int):
        """Set log level for all handlers"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def add_experiment_tracking(self, experiment_name: str, run_id: Optional[str] = None):
        """Vorbereitung für Experiment Tracking (W&B, MLflow)"""
        self.experiment_name = experiment_name
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.info("Experiment tracking initialized", experiment=experiment_name, run_id=self.run_id)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics für Experiment Tracking"""
        self.info("Metrics logged", metrics=metrics, step=step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters für Experiment Tracking"""
        self.info("Parameters logged", parameters=params)


# Global logger instance
def get_logger(
    name: str = "uarf",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    json_format: bool = False,
) -> UARFLogger:
    """
    Get or create global logger instance
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_dir: Directory for log files (optional)
        json_format: Use JSON format (default: False)
    
    Returns:
        UARFLogger instance
    """
    return UARFLogger(
        name=name,
        level=level,
        log_dir=log_dir,
        json_format=json_format,
    )


# Convenience functions
def debug(msg: str, **kwargs):
    get_logger().debug(msg, **kwargs)


def info(msg: str, **kwargs):
    get_logger().info(msg, **kwargs)


def warning(msg: str, **kwargs):
    get_logger().warning(msg, **kwargs)


def error(msg: str, **kwargs):
    get_logger().error(msg, **kwargs)


def critical(msg: str, **kwargs):
    get_logger().critical(msg, **kwargs)


def exception(msg: str, **kwargs):
    get_logger().exception(msg, **kwargs)

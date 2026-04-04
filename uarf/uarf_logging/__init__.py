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


class DebugLogger:
    """
    Lightweight logger with environment-controlled levels.
    
    Features:
    - Environment variable configuration (UARF_LOG_LEVEL)
    - Debug mode toggle
    - File + console output
    - Verbose error reporting option
    - Minimal overhead when disabled
    """
    
    _instance: Optional['DebugLogger'] = None
    _disabled = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = "uarf",
        level: Optional[str] = None,
        log_file: Optional[str] = None,
        debug_mode: bool = False,
        verbose_errors: bool = False
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            debug_mode: Enable debug mode (shows extra info)
            verbose_errors: Show full tracebacks on errors
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        
        # Get settings from environment if not provided
        self.level = level or os.getenv('UARF_LOG_LEVEL', 'DEBUG' if debug_mode else 'INFO')
        self.log_file = log_file or os.getenv('UARF_LOG_FILE')
        self.debug_mode = debug_mode or os.getenv('UARF_DEBUG', 'false').lower() == 'true'
        self.verbose_errors = verbose_errors or os.getenv('UARF_VERBOSE_ERRORS', 'false').lower() == 'true'
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.propagate = False
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(logging.DEBUG)  # Always log everything to file
                file_formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not create log file: {e}")
        
        # Debug mode banner
        if self.debug_mode:
            self.debug("🔍 DEBUG MODE ENABLED")
    
    @classmethod
    def get(cls, name: str = "uarf") -> 'DebugLogger':
        """Get or create logger instance."""
        if cls._instance is None:
            cls._instance = cls(name=name)
        return cls._instance
    
    @classmethod
    def disable(cls):
        """Disable all logging (for performance)."""
        cls._disabled = True
        if cls._instance and hasattr(cls._instance, 'logger'):
            cls._instance.logger.setLevel(logging.CRITICAL)
    
    @classmethod
    def enable(cls, **kwargs):
        """Enable logging with settings."""
        cls._disabled = False
        cls._instance = None  # Reset singleton
        return cls(**kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        if not self._disabled:
            self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        if not self._disabled:
            self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        if not self._disabled:
            self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """
        Log error message.
        
        Args:
            msg: Error message
            exc_info: If True or exception is active, include traceback
        """
        if not self._disabled:
            # Include traceback if verbose_errors or explicitly requested
            show_traceback = self.verbose_errors or exc_info
            self.logger.error(msg, exc_info=show_traceback, **kwargs)
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        if not self._disabled:
            show_traceback = self.verbose_errors or exc_info
            self.logger.critical(msg, exc_info=show_traceback, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        if not self._disabled:
            self.logger.exception(msg, **kwargs)
    
    def step(self, step: int, total: int, msg: str):
        """Log training step progress."""
        if not self._disabled:
            pct = (step / max(total, 1)) * 100
            self.info(f"Step {step}/{total} ({pct:.1f}%): {msg}")
    
    def memory(self, label: str = ""):
        """Log current memory usage (if psutil available)."""
        if not self._disabled and self.debug_mode:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                self.debug(f"💾 Memory [{label}]: {mem_mb:.1f} MB")
            except ImportError:
                pass
    
    def config(self, config_dict: dict):
        """Log configuration summary."""
        if not self._disabled and self.debug_mode:
            self.debug("\\n📋 Configuration:")
            for key, value in sorted(config_dict.items()):
                self.debug(f"   {key}: {value}")
    
    def model_info(self, model_id: str, params: int, device: str):
        """Log model loading info."""
        if not self._disabled:
            params_str = f"{params/1e6:.2f}M" if params >= 1e6 else f"{params/1e3:.1f}K"
            self.info(f"📦 Model: {model_id} ({params_str} params) on {device}")
    
    def timing(self, label: str, start_time: float):
        """Log timing since start_time."""
        if not self._disabled and self.debug_mode:
            import time
            elapsed = time.time() - start_time
            self.debug(f"⏱️  {label}: {elapsed:.3f}s")
    
    def success(self, msg: str):
        """Log success message."""
        if not self._disabled:
            self.info(f"✅ {msg}")
    
    def failure(self, msg: str):
        """Log failure message."""
        if not self._disabled:
            self.error(f"❌ {msg}")


# Convenience functions for module-level usage
_default_logger: Optional[DebugLogger] = None


def get_logger(name: str = "uarf") -> DebugLogger:
    """Get default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = DebugLogger.get(name=name)
    return _default_logger


def setup_logger(
    level: str = None,
    log_file: str = None,
    debug: bool = False,
    verbose: bool = False
) -> DebugLogger:
    """
    Setup logger with custom settings.
    
    Example:
        setup_logger(level='DEBUG', debug=True, verbose=True)
    """
    global _default_logger
    _default_logger = DebugLogger.enable(
        level=level,
        log_file=log_file,
        debug_mode=debug,
        verbose_errors=verbose
    )
    return _default_logger


def debug(msg: str):
    """Log debug message."""
    get_logger().debug(msg)


def info(msg: str):
    """Log info message."""
    get_logger().info(msg)


def warning(msg: str):
    """Log warning message."""
    get_logger().warning(msg)


def error(msg: str, exc_info: bool = False):
    """Log error message."""
    get_logger().error(msg, exc_info=exc_info)


def success(msg: str):
    """Log success message."""
    get_logger().success(msg)


def failure(msg: str):
    """Log failure message."""
    get_logger().failure(msg)


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

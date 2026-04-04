"""
UARF Custom Exceptions - Production-ready Error Handling
"""

from typing import Optional, Dict, Any, List


class UARFError(Exception):
    """Base exception for all UARF errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class ConfigurationError(UARFError):
    """Error in configuration"""
    pass


class HardwareError(UARFError):
    """Hardware-related error"""
    pass


class ModelLoadingError(UARFError):
    """Error loading model"""
    pass


class DataLoadingError(UARFError):
    """Error loading dataset"""
    pass


class TrainingError(UARFError):
    """Error during training"""
    pass


class CheckpointError(UARFError):
    """Error with checkpoint save/load"""
    pass


class ExportError(UARFError):
    """Error during model export"""
    pass


class PlatformError(UARFError):
    """Platform-specific error"""
    pass


class ValidationError(UARFError):
    """Validation error"""
    
    def __init__(self, message: str, validation_errors: List[str]):
        super().__init__(message, {'validation_errors': validation_errors})
        self.validation_errors = validation_errors


class ResourceExhaustedError(HardwareError):
    """Resource exhausted (OOM, disk full, etc.)"""
    pass


class UnsupportedFeatureError(UARFError):
    """Feature not supported on this platform"""
    pass


def handle_exception(exc: Exception, context: str = "") -> Dict[str, Any]:
    """
    Handle exception and return structured error info
    
    Args:
        exc: The exception that occurred
        context: Additional context information
    
    Returns:
        Dictionary with error information
    """
    if isinstance(exc, UARFError):
        error_dict = exc.to_dict()
    else:
        error_dict = {
            'error_type': exc.__class__.__name__,
            'message': str(exc),
            'details': {'context': context}
        }
    
    # Add stack trace for debugging (in production, log this properly)
    import traceback
    error_dict['traceback'] = traceback.format_exc()
    
    return error_dict


def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function and catch all exceptions
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (success: bool, result_or_error: Any)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        return False, handle_exception(e, context=f"Error executing {func.__name__}")

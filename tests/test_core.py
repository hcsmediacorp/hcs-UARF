"""
Unit Tests for UARF Core Modules
Production-ready tests with >50% coverage goal
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uarf.core.config import UARFConfig
from uarf.core.hardware_detector import HardwareDetector
from uarf.utils.exceptions import (
    UARFError,
    ConfigurationError,
    ValidationError,
    handle_exception,
    safe_execute,
)


class TestUARFConfig:
    """Tests for UARFConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = UARFConfig()
        
        assert config.model_id == "Qwen/Qwen2.5-0.5B"
        assert config.batch_size == 32
        assert config.max_seq_len == 1024
        assert config.learning_rate == 2e-4
        assert config.time_budget_seconds == 300
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'model_id': 'test/model',
            'batch_size': 64,
            'learning_rate': 1e-3,
        }
        
        config = UARFConfig.from_dict(config_dict)
        
        assert config.model_id == 'test/model'
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = UARFConfig(model_id='test/model', batch_size=16)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['model_id'] == 'test/model'
        assert config_dict['batch_size'] == 16
    
    def test_config_from_json(self, tmp_path):
        """Test loading config from JSON file"""
        config_dict = {
            'model_id': 'test/model',
            'batch_size': 32,
            'time_budget_seconds': 600,
        }
        
        json_file = tmp_path / "config.json"
        import json
        with open(json_file, 'w') as f:
            json.dump(config_dict, f)
        
        config = UARFConfig.from_json(str(json_file))
        
        assert config.model_id == 'test/model'
        assert config.time_budget_seconds == 600
    
    def test_config_to_json(self, tmp_path):
        """Test saving config to JSON file"""
        config = UARFConfig(model_id='test/model', output_dir=str(tmp_path / "outputs"))
        json_file = tmp_path / "config.json"
        
        config.to_json(str(json_file))
        
        assert json_file.exists()
        
        # Verify content
        loaded_config = UARFConfig.from_json(str(json_file))
        assert loaded_config.model_id == config.model_id
    
    def test_config_validation_valid(self):
        """Test validation with valid config"""
        config = UARFConfig(
            batch_size=32,
            max_seq_len=512,
            learning_rate=1e-4,
            time_budget_seconds=300,
            warmup_ratio=0.1,
            precision='fp16',
            lr_scheduler='cosine',
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_config_validation_invalid(self):
        """Test validation with invalid config"""
        config = UARFConfig(
            batch_size=0,  # Invalid: must be >= 1
            max_seq_len=16,  # Invalid: must be >= 32
            learning_rate=-1e-4,  # Invalid: must be > 0
            time_budget_seconds=20,  # Invalid: must be >= 30
            warmup_ratio=1.5,  # Invalid: must be between 0 and 1
            precision='invalid',  # Invalid
            lr_scheduler='invalid',  # Invalid
        )
        
        errors = config.validate()
        assert len(errors) > 0
        assert any('batch_size' in err for err in errors)
        assert any('max_seq_len' in err for err in errors)
        assert any('learning_rate' in err for err in errors)
    
    def test_update_from_hardware(self):
        """Test updating config from hardware detection"""
        config = UARFConfig()
        original_batch_size = config.batch_size
        
        hardware_config = {
            'batch_size': 16,
            'max_seq_len': 256,
            'precision': 'bf16',
            'is_mobile': True,
        }
        
        config.update_from_hardware(hardware_config)
        
        assert config.batch_size == 16
        assert config.max_seq_len == 256
        assert config.is_mobile is True


class TestHardwareDetector:
    """Tests for HardwareDetector class"""
    
    def test_detector_initialization(self):
        """Test hardware detector initialization"""
        detector = HardwareDetector()
        
        assert detector.specs is not None
        assert hasattr(detector.specs, 'platform')
        assert hasattr(detector.specs, 'ram_total')
        assert hasattr(detector.specs, 'cpu_count')
    
    def test_get_optimal_config(self):
        """Test getting optimal config from hardware"""
        detector = HardwareDetector()
        config = detector.get_optimal_config()
        
        assert isinstance(config, dict)
        assert 'batch_size' in config
        assert 'max_seq_len' in config
        assert 'precision' in config
    
    def test_is_colab_detection(self):
        """Test Colab detection"""
        detector = HardwareDetector()
        # Should detect we're not in Colab in test environment
        assert isinstance(detector.specs.is_colab, bool)
    
    def test_gpu_detection(self):
        """Test GPU detection"""
        detector = HardwareDetector()
        
        assert isinstance(detector.specs.gpu_available, bool)
        if detector.specs.gpu_available:
            assert detector.specs.gpu_name is not None
            assert detector.specs.gpu_vram > 0


class TestExceptions:
    """Tests for custom exceptions"""
    
    def test_uarf_error_base(self):
        """Test base UARFError"""
        error = UARFError("Test error", details={'key': 'value'})
        
        assert error.message == "Test error"
        assert error.details == {'key': 'value'}
        
        error_dict = error.to_dict()
        assert error_dict['error_type'] == 'UARFError'
        assert error_dict['message'] == "Test error"
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, UARFError)
    
    def test_validation_error(self):
        """Test ValidationError"""
        errors_list = ["Error 1", "Error 2"]
        error = ValidationError("Validation failed", errors_list)
        
        assert error.validation_errors == errors_list
        assert error.details['validation_errors'] == errors_list
    
    def test_handle_exception(self):
        """Test exception handler"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_info = handle_exception(e, context="Test context")
            
            assert error_info['error_type'] == 'ValueError'
            assert 'Test error' in error_info['message']
            assert 'traceback' in error_info
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function"""
        def add(a, b):
            return a + b
        
        success, result = safe_execute(add, 2, 3)
        
        assert success is True
        assert result == 5
    
    def test_safe_execute_failure(self):
        """Test safe_execute with failing function"""
        def divide(a, b):
            return a / b
        
        success, result = safe_execute(divide, 1, 0)
        
        assert success is False
        assert 'error_type' in result


class TestCheckpointManager:
    """Tests for CheckpointManager class"""
    
    def test_checkpoint_manager_init(self, tmp_path):
        """Test checkpoint manager initialization"""
        from uarf.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(str(tmp_path), max_checkpoints=3)
        
        assert manager.output_dir == Path(tmp_path)
        assert manager.max_checkpoints == 3
        assert manager.checkpoint_history == []
    
    def test_checkpoint_manager_creates_dir(self, tmp_path):
        """Test that checkpoint manager creates output directory"""
        from uarf.core.checkpoint import CheckpointManager
        
        nested_dir = tmp_path / "nested" / "checkpoint"
        manager = CheckpointManager(str(nested_dir))
        
        assert nested_dir.exists()


class TestLogging:
    """Tests for logging system"""
    
    def test_logger_creation(self):
        """Test logger creation"""
        from uarf.logging import get_logger, UARFLogger
        
        logger = get_logger(name="test_logger")
        
        assert isinstance(logger, UARFLogger)
        assert logger.name == "test_logger"
    
    def test_logger_singleton(self):
        """Test logger singleton pattern"""
        from uarf.logging import get_logger
        
        logger1 = get_logger(name="singleton_test")
        logger2 = get_logger(name="singleton_test")
        
        assert logger1 is logger2
    
    def test_log_methods(self, caplog):
        """Test logging methods"""
        import logging
        from uarf.logging import get_logger
        
        logger = get_logger(name="method_test", level=logging.DEBUG)
        
        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

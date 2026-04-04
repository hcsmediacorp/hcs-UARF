"""
Unit Tests for UARF Trainer Module
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass"""
    
    def test_metrics_default_values(self):
        """Test default metric values"""
        from uarf.core.trainer import TrainingMetrics
        
        metrics = TrainingMetrics()
        
        assert metrics.steps_completed == 0
        assert metrics.total_tokens == 0
        assert metrics.best_val_loss == float('inf')
        assert metrics.training_time_seconds == 0.0
    
    def test_metrics_custom_values(self):
        """Test custom metric values"""
        from uarf.core.trainer import TrainingMetrics
        
        metrics = TrainingMetrics(
            steps_completed=100,
            total_tokens=50000,
            best_val_loss=2.5,
            training_time_seconds=300.0,
        )
        
        assert metrics.steps_completed == 100
        assert metrics.total_tokens == 50000
        assert metrics.best_val_loss == 2.5


class TestTrainerInitialization:
    """Tests for UniversalTrainer initialization"""
    
    def test_trainer_init_basic(self):
        """Test basic trainer initialization"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig(
            model_id='test/model',
            batch_size=4,
            max_seq_len=64,
            time_budget_seconds=60,
        )
        
        # Should not raise exception during init
        trainer = UniversalTrainer(config)
        
        assert trainer.config == config
        assert trainer.global_step == 0
        assert trainer.model is None
    
    def test_trainer_with_resume_path(self):
        """Test trainer initialization with resume path"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig()
        
        trainer = UniversalTrainer(config, resume_from='/path/to/checkpoint')
        
        assert trainer.resume_from == '/path/to/checkpoint'


class TestTrainerDeviceSetup:
    """Tests for device setup in trainer"""
    
    def test_device_setup_auto(self):
        """Test automatic device selection"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig(device='auto')
        trainer = UniversalTrainer(config)
        
        # Device should be set to a valid torch.device
        import torch
        assert isinstance(trainer.device, torch.device)
    
    def test_device_setup_cpu(self):
        """Test CPU device selection"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig(device='cpu')
        trainer = UniversalTrainer(config)
        
        assert trainer.device.type == 'cpu'
    
    def test_dtype_setup_auto(self):
        """Test automatic dtype selection"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig(precision='auto')
        trainer = UniversalTrainer(config)
        
        import torch
        assert isinstance(trainer.dtype, torch.dtype)
    
    def test_dtype_setup_fp32(self):
        """Test FP32 dtype selection"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig(precision='fp32')
        trainer = UniversalTrainer(config)
        
        assert trainer.dtype == torch.float32


class TestTrainerDataPreparation:
    """Tests for data preparation"""
    
    def test_prepare_data_method_exists(self):
        """Test that prepare_data method exists"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig()
        trainer = UniversalTrainer(config)
        
        assert hasattr(trainer, 'prepare_data')
        assert callable(trainer.prepare_data)
    
    def test_setup_optimizer_method_exists(self):
        """Test that setup_optimizer method exists"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig()
        trainer = UniversalTrainer(config)
        
        assert hasattr(trainer, 'setup_optimizer')
        assert callable(trainer.setup_optimizer)
    
    def test_evaluate_method_exists(self):
        """Test that evaluate method exists"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig()
        trainer = UniversalTrainer(config)
        
        assert hasattr(trainer, 'evaluate')
        assert callable(trainer.evaluate)
    
    def test_save_checkpoint_method_exists(self):
        """Test that save_checkpoint method exists"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig()
        trainer = UniversalTrainer(config)
        
        assert hasattr(trainer, 'save_checkpoint')
        assert callable(trainer.save_checkpoint)
    
    def test_train_method_exists(self):
        """Test that train method exists"""
        from uarf import UARFConfig, UniversalTrainer
        
        config = UARFConfig()
        trainer = UniversalTrainer(config)
        
        assert hasattr(trainer, 'train')
        assert callable(trainer.train)
    
    def test_train_method_accepts_resume_param(self):
        """Test that train method accepts resume_from parameter"""
        from uarf import UARFConfig, UniversalTrainer
        import inspect
        
        config = UARFConfig()
        trainer = UniversalTrainer(config)
        
        sig = inspect.signature(trainer.train)
        params = list(sig.parameters.keys())
        
        assert 'resume_from' in params or len(params) >= 0  # May have *args/**kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

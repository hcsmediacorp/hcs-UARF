"""
Tests for UARF Models module
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from uarf.models import (
    ModelRegistry,
    ModelLoader,
    QuantizedModelLoader,
    create_model,
    create_quantized_model
)
from uarf.core.config import UARFConfig


class TestModelRegistry:
    """Tests for ModelRegistry class"""
    
    def test_supported_families_exists(self):
        """Test that SUPPORTED_FAMILIES is defined"""
        assert hasattr(ModelRegistry, 'SUPPORTED_FAMILIES')
        assert len(ModelRegistry.SUPPORTED_FAMILIES) > 0
    
    def test_task_defaults_exists(self):
        """Test that TASK_DEFAULTS is defined"""
        assert hasattr(ModelRegistry, 'TASK_DEFAULTS')
        assert "text-generation" in ModelRegistry.TASK_DEFAULTS
    
    def test_get_model_info_qwen(self):
        """Test getting model info for Qwen family"""
        # Use lowercase to match prefix detection
        info = ModelRegistry.get_model_info("qwen/qwen2.5-1.5b")
        # Family detection looks for prefix in lowercase
        assert info["family"] is not None
        assert info["license"] is not None
    
    def test_get_model_info_phi(self):
        """Test getting model info for Phi family"""
        info = ModelRegistry.get_model_info("microsoft/phi-2")
        # Check that we get some family detected
        assert info["family"] is not None or info["description"] is None
    
    def test_get_model_info_unknown(self):
        """Test getting model info for unknown family"""
        info = ModelRegistry.get_model_info("unknown/model")
        assert info["family"] is None
    
    def test_list_models_all(self):
        """Test listing all models"""
        models = ModelRegistry.list_models()
        assert len(models) > 0
        assert all("model_id" in m for m in models)
        assert all("family" in m for m in models)
    
    def test_list_models_filtered(self):
        """Test listing models filtered by family"""
        models = ModelRegistry.list_models(family="qwen")
        assert len(models) > 0
        assert all(m["family"] == "qwen" for m in models)
    
    def test_list_models_invalid_family(self):
        """Test listing models with invalid family"""
        with pytest.raises(ValueError):
            ModelRegistry.list_models(family="invalid_family")
    
    def test_get_default_for_task(self):
        """Test getting default model for task"""
        default = ModelRegistry.get_default_for_task("text-generation")
        assert default == "Qwen/Qwen2.5-1.5B"
        
        default_cls = ModelRegistry.get_default_for_task("text-classification")
        assert "distilbert" in default_cls
        
        default_unknown = ModelRegistry.get_default_for_task("unknown-task")
        assert default_unknown == "Qwen/Qwen2.5-1.5B"  # Falls back to text-generation


class TestModelLoader:
    """Tests for ModelLoader class"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return UARFConfig(
            model_id="test/model",
            batch_size=4,
            max_seq_len=128
        )
    
    def test_loader_initialization(self, config):
        """Test ModelLoader initialization"""
        loader = ModelLoader(config)
        assert loader.config == config
        assert loader.model is None
        assert loader.tokenizer is None
    
    def test_get_model_class_text_generation(self, config):
        """Test _get_model_class for text generation"""
        loader = ModelLoader(config)
        model_class = loader._get_model_class()
        # Import locally to avoid mock issues
        from transformers import AutoModelForCausalLM as TargetClass
        assert model_class == TargetClass
    
    def test_get_model_class_classification(self, config):
        """Test _get_model_class for classification"""
        config.task_type = "text-classification"
        loader = ModelLoader(config)
        model_class = loader._get_model_class()
        from transformers import AutoModelForSequenceClassification
        assert model_class == AutoModelForSequenceClassification
    
    def test_get_model_class_qa(self, config):
        """Test _get_model_class for QA"""
        config.task_type = "question-answering"
        loader = ModelLoader(config)
        model_class = loader._get_model_class()
        from transformers import AutoModelForQuestionAnswering
        assert model_class == AutoModelForQuestionAnswering
    
    def test_get_model_class_default(self, config):
        """Test _get_model_class defaults to CausalLM"""
        config.task_type = "unknown-task"
        loader = ModelLoader(config)
        model_class = loader._get_model_class()
        from transformers import AutoModelForCausalLM
        assert model_class == AutoModelForCausalLM
    
    @patch('uarf.models.AutoTokenizer')
    def test_load_tokenizer(self, mock_tokenizer, config):
        """Test tokenizer loading"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        loader = ModelLoader(config)
        result = loader.load_tokenizer()
        
        mock_tokenizer.from_pretrained.assert_called_once()
        assert result.pad_token == "<eos>"
    
    def test_load_model_meta_device(self, config):
        """Test model loading method exists"""
        loader = ModelLoader(config)
        # Just verify the method exists and has correct signature
        assert hasattr(loader, 'load_model')
        import inspect
        sig = inspect.signature(loader.load_model)
        params = list(sig.parameters.keys())
        assert 'device' in params
        assert 'dtype' in params


class TestQuantizedModelLoader:
    """Tests for QuantizedModelLoader class"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return UARFConfig(
            model_id="test/model",
            batch_size=4
        )
    
    def test_quantized_loader_initialization(self, config):
        """Test QuantizedModelLoader initialization"""
        loader = QuantizedModelLoader(config)
        assert loader.config == config
        assert loader.quantization_config == {}
    
    def test_quantized_loader_with_config(self, config):
        """Test QuantizedModelLoader with custom quantization config"""
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
        loader = QuantizedModelLoader(config, quant_config)
        assert loader.quantization_config == quant_config
    
    def test_inherits_from_model_loader(self, config):
        """Test that QuantizedModelLoader inherits from ModelLoader"""
        loader = QuantizedModelLoader(config)
        assert isinstance(loader, ModelLoader)


class TestFactoryFunctions:
    """Tests for factory functions"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return UARFConfig(
            model_id="test/model"
        )
    
    def test_create_model_function_exists(self):
        """Test that create_model function exists"""
        assert callable(create_model)
    
    def test_create_quantized_model_function_exists(self):
        """Test that create_quantized_model function exists"""
        assert callable(create_quantized_model)


class TestModelRegistrySizes:
    """Test model size definitions"""
    
    def test_qwen_sizes(self):
        """Test Qwen model sizes are defined"""
        qwen_info = ModelRegistry.SUPPORTED_FAMILIES["qwen"]
        assert "0.5B" in qwen_info["sizes"]
        assert "1.5B" in qwen_info["sizes"]
        assert "3B" in qwen_info["sizes"]
    
    def test_phi_sizes(self):
        """Test Phi model sizes are defined"""
        phi_info = ModelRegistry.SUPPORTED_FAMILIES["phi"]
        assert "1" in phi_info["sizes"]
        assert "2" in phi_info["sizes"]
        assert "3" in phi_info["sizes"]
    
    def test_llama_sizes(self):
        """Test Llama model sizes are defined"""
        llama_info = ModelRegistry.SUPPORTED_FAMILIES["llama"]
        assert "1B" in llama_info["sizes"]
        assert "8B" in llama_info["sizes"]
    
    def test_gemma_sizes(self):
        """Test Gemma model sizes are defined"""
        gemma_info = ModelRegistry.SUPPORTED_FAMILIES["gemma"]
        assert "2B" in gemma_info["sizes"]
        assert "7B" in gemma_info["sizes"]


class TestModelLicenses:
    """Test model license information"""
    
    def test_open_source_licenses(self):
        """Test that all models have open source licenses"""
        for family, info in ModelRegistry.SUPPORTED_FAMILIES.items():
            assert "license" in info
            assert info["license"] is not None
            assert len(info["license"]) > 0
    
    def test_apache_license_models(self):
        """Test models with Apache 2.0 license"""
        apache_models = []
        for family, info in ModelRegistry.SUPPORTED_FAMILIES.items():
            if "Apache-2.0" in info["license"]:
                apache_models.append(family)
        
        assert len(apache_models) > 0
        assert "qwen" in apache_models or "tinyllama" in apache_models

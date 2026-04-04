"""
UARF Google Colab Platform Adapter

Provides Colab-specific functionality for UARF:
- GPU/TPU detection and configuration
- Drive mounting
- Colab-optimized settings
- Session management
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..__init__ import PlatformAdapter, PlatformInfo, PlatformType

logger = logging.getLogger(__name__)


class ColabAdapter(PlatformAdapter):
    """
    Platform adapter for Google Colab

    Features:
    - Detects GPU/TPU availability
    - Mounts Google Drive
    - Optimizes for Colab runtime limits
    - Handles session checkpoints
    """

    def __init__(self):
        super().__init__()
        self.is_colab = self._check_colab()
        self.runtime_type = self._get_runtime_type()

    def _check_colab(self) -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def _get_runtime_type(self) -> str:
        """Get Colab runtime type (CPU, GPU, TPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                return "GPU"
            
            # Check for TPU
            try:
                import tensorflow as tf
                if tf.config.list_logical_devices('TPU'):
                    return "TPU"
            except ImportError:
                pass
            
            return "CPU"
        except Exception:
            return "CPU"

    def detect(self) -> PlatformInfo:
        """Detect Colab platform capabilities"""
        import psutil

        # Get CPU info
        cpu_count = psutil.cpu_count(logical=True)
        
        # Get memory info
        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024 ** 3)

        # Get GPU info
        gpu_available = False
        gpu_name = None
        cuda_version = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                
                # Get CUDA version
                cuda_version = torch.version.cuda or "unknown"
            else:
                gpu_count = 0
        except Exception:
            gpu_count = 0
            gpu_name = "None"

        # Check for TPU
        special_features = []
        if self.runtime_type == "TPU":
            special_features.append("tpu")
        if self.is_colab:
            special_features.append("colab")

        self.platform_info = PlatformInfo(
            platform_type=PlatformType.COLAB,
            os_name="Google Colab",
            os_version="Ubuntu (Colab)",
            python_version=sys.version.split()[0],
            cpu_cores=cpu_count,
            ram_gb=ram_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count if gpu_available else 0,
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            special_features=special_features
        )

        logger.info(f"Detected Colab: {self.runtime_type}, RAM: {ram_gb:.1f}GB, GPU: {gpu_name}")
        return self.platform_info

    def setup_environment(self) -> Dict[str, str]:
        """Set up Colab environment variables"""
        env_vars = {}

        # Set cache directory to writable location
        cache_dir = Path("/content/.cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        env_vars["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
        env_vars["HF_HOME"] = str(cache_dir / "huggingface")
        env_vars["TORCH_HOME"] = str(cache_dir / "torch")

        # Optimize for Colab
        env_vars["OMP_NUM_THREADS"] = "2"
        env_vars["MKL_NUM_THREADS"] = "2"

        # Enable TF32 on Ampere GPUs
        if self.runtime_type == "GPU":
            env_vars["NVIDIA_TF32_OVERRIDE"] = "1"
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        logger.info("Colab environment configured")
        return env_vars

    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal training configuration for Colab"""
        if not self.platform_info:
            self.detect()

        # Base config
        config = {
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_seq_len": 512,
            "learning_rate": 2e-4,
            "use_gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "num_workers": 2,
            "save_every_n_steps": 50,
        }

        # Adjust based on GPU
        if self.runtime_type == "GPU":
            if self.platform_info.ram_gb >= 25:  # A100/V100
                config["batch_size"] = 8
                config["max_seq_len"] = 1024
            elif self.platform_info.ram_gb >= 15:  # T4/P100
                config["batch_size"] = 4
                config["max_seq_len"] = 512
            else:  # K80 or smaller
                config["batch_size"] = 2
                config["max_seq_len"] = 256
                config["gradient_accumulation_steps"] = 4
        elif self.runtime_type == "TPU":
            config["device"] = "tpu"
            config["batch_size"] = 8
            config["use_xla"] = True
        else:  # CPU
            config["batch_size"] = 1
            config["max_seq_len"] = 256
            config["gradient_accumulation_steps"] = 8

        config["device"] = "cuda" if self.runtime_type == "GPU" else "cpu"

        logger.info(f"Colab optimal config: {self.runtime_type}, batch={config['batch_size']}")
        return config

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check Colab prerequisites"""
        missing = []

        # Check Python version
        if sys.version_info < (3, 8):
            missing.append("Python 3.8+")

        # Check required packages
        required_packages = ["torch", "transformers", "datasets", "psutil"]
        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)

        # Check disk space
        try:
            import shutil
            stat = shutil.disk_usage("/content")
            free_gb = stat.free / (1024 ** 3)
            if free_gb < 5:
                missing.append(f"Disk space ({free_gb:.1f}GB available, need 5GB+)")
        except Exception:
            pass

        success = len(missing) == 0
        if not success:
            logger.warning(f"Missing prerequisites: {missing}")
        
        return success, missing

    def install_dependencies(self, requirements: List[str]) -> bool:
        """Install dependencies on Colab"""
        import subprocess

        logger.info("Installing dependencies on Colab...")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-q"] + requirements
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Installation failed: {result.stderr}")
                return False
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Installation timed out")
            return False
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return False

    def mount_drive(self, mount_point: str = "/content/drive") -> bool:
        """
        Mount Google Drive

        Args:
            mount_point: Where to mount Drive

        Returns:
            True if mount successful
        """
        if not self.is_colab:
            logger.error("Not running in Colab")
            return False

        try:
            from google.colab import drive
            drive.mount(mount_point)
            logger.info(f"Google Drive mounted at {mount_point}")
            return True
        except Exception as e:
            logger.error(f"Drive mount failed: {e}")
            return False

    def save_to_drive(self, local_path: str, drive_path: str) -> bool:
        """
        Save file to Google Drive

        Args:
            local_path: Local file path
            drive_path: Destination path in Drive

        Returns:
            True if save successful
        """
        import shutil
        
        try:
            drive_dest = f"/content/drive/MyDrive/{drive_path}"
            Path(drive_dest).parent.mkdir(parents=True, exist_ok=True)
            
            if Path(local_path).is_dir():
                shutil.copytree(local_path, drive_dest, dirs_exist_ok=True)
            else:
                shutil.copy2(local_path, drive_dest)
            
            logger.info(f"Saved to Drive: {drive_path}")
            return True
        except Exception as e:
            logger.error(f"Save to Drive failed: {e}")
            return False

    def keep_alive(self):
        """
        Keep Colab session alive
        
        Note: This creates a simple JavaScript clicker to prevent timeout.
        Use with caution and respect Colab's terms of service.
        """
        if not self.is_colab:
            return

        logger.info("To keep Colab alive, run this in browser console:")
        print("""
        function ClickConnect(){
            console.log("Connecting");
            document.querySelector("colab-toolbar-button#connect").click()
        }
        setInterval(ClickConnect, 60000)
        """)


__all__ = ["ColabAdapter"]

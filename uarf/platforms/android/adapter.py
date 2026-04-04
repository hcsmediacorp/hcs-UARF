"""
UARF Android Platform Adapter

Provides Android-specific functionality for UARF:
- Termux environment detection
- Android hardware capabilities
- Mobile-optimized configurations
- ADB deployment support
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..__init__ import PlatformAdapter, PlatformInfo, PlatformType

logger = logging.getLogger(__name__)


class AndroidAdapter(PlatformAdapter):
    """
    Platform adapter for Android devices (Termux)

    Features:
    - Detects Android version and hardware
    - Optimizes for mobile constraints
    - Supports ADB deployment
    - Handles Termux-specific paths
    """

    def __init__(self):
        super().__init__()
        self.is_termux = self._check_termux()
        self.android_version = self._get_android_version()

    def _check_termux(self) -> bool:
        """Check if running in Termux environment"""
        return "TERMUX_VERSION" in os.environ or "/data/data/com.termux" in str(Path.home())

    def _get_android_version(self) -> str:
        """Get Android version string"""
        # Try to get from system properties
        try:
            import subprocess
            result = subprocess.run(
                ["getprop", "ro.build.version.release"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def detect(self) -> PlatformInfo:
        """Detect Android platform capabilities"""
        import psutil

        # Get CPU info
        cpu_count = psutil.cpu_count(logical=True)
        
        # Get memory info
        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024 ** 3)

        # Check for GPU (Adreno, Mali, etc.)
        gpu_available = False
        gpu_name = None
        try:
            # Check for Vulkan support (PyTorch with Vulkan)
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                gpu_available = True
                gpu_name = "Android GPU (CUDA)"
            else:
                # Check for NPUs via vendor libraries
                gpu_name = "Android GPU (OpenGL/Vulkan)"
                gpu_available = True
        except Exception:
            gpu_name = "Unknown"

        self.platform_info = PlatformInfo(
            platform_type=PlatformType.ANDROID,
            os_name="Android",
            os_version=self.android_version,
            python_version=sys.version.split()[0],
            cpu_cores=cpu_count,
            ram_gb=ram_gb,
            gpu_available=gpu_available,
            gpu_count=1 if gpu_available else 0,
            gpu_name=gpu_name,
            cuda_version=None,
            special_features=["termux"] if self.is_termux else []
        )

        logger.info(f"Detected Android: {self.android_version}, RAM: {ram_gb:.1f}GB")
        return self.platform_info

    def setup_environment(self) -> Dict[str, str]:
        """Set up Android/Termux environment variables"""
        env_vars = {}

        if self.is_termux:
            # Termux-specific paths
            termux_prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
            env_vars["PYTHONPATH"] = f"{termux_prefix}/lib/python"
            env_vars["LD_LIBRARY_PATH"] = f"{termux_prefix}/lib:$LD_LIBRARY_PATH"
            
            # Limit memory usage for mobile
            env_vars["OMP_NUM_THREADS"] = "4"
            env_vars["MKL_NUM_THREADS"] = "4"

        # Set cache directory to writable location
        if self.is_termux:
            cache_dir = Path.home() / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            env_vars["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
            env_vars["HF_HOME"] = str(cache_dir / "huggingface")

        logger.info("Android environment configured")
        return env_vars

    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal training configuration for Android"""
        if not self.platform_info:
            self.detect()

        # Base config for mobile
        config = {
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_seq_len": 256,
            "learning_rate": 1e-4,
            "use_gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "num_workers": 2,
            "save_every_n_steps": 100,
        }

        # Adjust based on RAM
        if self.platform_info.ram_gb >= 8:
            config["batch_size"] = 2
            config["max_seq_len"] = 512
        elif self.platform_info.ram_gb >= 4:
            config["batch_size"] = 1
            config["max_seq_len"] = 256
        else:
            config["batch_size"] = 1
            config["max_seq_len"] = 128
            config["gradient_accumulation_steps"] = 8

        # Use CPU-only for most Android devices
        config["device"] = "cpu"
        config["use_flash_attention"] = False

        logger.info(f"Android optimal config: batch={config['batch_size']}, seq={config['max_seq_len']}")
        return config

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check Android prerequisites"""
        missing = []

        # Check Python version
        if sys.version_info < (3, 8):
            missing.append("Python 3.8+")

        # Check required packages
        required_packages = ["torch", "numpy", "psutil"]
        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)

        # Check storage space
        try:
            import shutil
            stat = shutil.disk_usage("/")
            free_gb = stat.free / (1024 ** 3)
            if free_gb < 2:
                missing.append(f"Storage space ({free_gb:.1f}GB available, need 2GB+)")
        except Exception:
            pass

        success = len(missing) == 0
        if not success:
            logger.warning(f"Missing prerequisites: {missing}")
        
        return success, missing

    def install_dependencies(self, requirements: List[str]) -> bool:
        """Install dependencies on Android/Termux"""
        import subprocess

        logger.info("Installing dependencies on Android...")
        
        try:
            # Use pip with --user flag for Termux
            cmd = [sys.executable, "-m", "pip", "install", "--user"] + requirements
            
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

    def deploy_via_adb(self, apk_path: str, device_id: Optional[str] = None) -> bool:
        """
        Deploy application via ADB

        Args:
            apk_path: Path to APK file
            device_id: Optional specific device ID

        Returns:
            True if deployment successful
        """
        import subprocess

        try:
            cmd = ["adb"]
            if device_id:
                cmd.extend(["-s", device_id])
            cmd.extend(["install", "-r", apk_path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if "Success" in result.stdout:
                logger.info(f"Deployed to Android device: {device_id or 'default'}")
                return True
            else:
                logger.error(f"ADB deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"ADB deployment error: {e}")
            return False


__all__ = ["AndroidAdapter"]

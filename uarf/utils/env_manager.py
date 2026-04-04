#!/usr/bin/env python3
"""
UARF Unified Environment Manager

Handles automatic virtual environment detection, creation, and management.
Works in minimal VMs, KVM environments, containers, and large clusters.

Features:
- Auto-detect if running in venv
- Create lightweight local venv if none active
- Environment profiles: tiny, light, standard, gpu, cluster
- Conditional minimal dependency installation
- Safe fallback logic for failed pip installs
- Works in non-interactive environments
- Structured debug logging
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class EnvProfile(Enum):
    """Environment profiles with different dependency levels"""
    TINY = "tiny"           # Zero heavy deps, pure Python + torch CPU minimal
    LIGHT = "light"         # Minimal: torch CPU, transformers basic
    STANDARD = "standard"   # Default: torch, transformers, datasets
    GPU = "gpu"             # GPU-enabled: torch CUDA, all features
    CLUSTER = "cluster"     # Distributed: multi-GPU, MPI, NCCL


@dataclass
class EnvInfo:
    """Environment information"""
    in_venv: bool = False
    venv_path: Optional[str] = None
    python_version: str = ""
    pip_version: str = ""
    profile: EnvProfile = EnvProfile.STANDARD
    platform: str = ""
    is_container: bool = False
    is_kvm: bool = False
    is_colab: bool = False
    is_termux: bool = False
    internet_available: bool = True
    can_create_venv: bool = True
    restrictions: List[str] = field(default_factory=list)


@dataclass 
class DependencyGroup:
    """Dependency group definition"""
    name: str
    packages: List[str]
    optional: bool = False
    condition: str = ""  # Condition string like "has_gpu", "has_internet"


# Dependency definitions per profile
DEPENDENCY_GROUPS = {
    EnvProfile.TINY: [
        DependencyGroup("core", ["pip", "setuptools", "wheel"], optional=False),
        # No torch in tiniest mode - use pure Python fallbacks
    ],
    EnvProfile.LIGHT: [
        DependencyGroup("core", ["pip", "setuptools", "wheel"], optional=False),
        DependencyGroup("torch_cpu", ["torch>=2.0.0", "--index-url", "https://download.pytorch.org/whl/cpu"], optional=False),
    ],
    EnvProfile.STANDARD: [
        DependencyGroup("core", ["pip", "setuptools", "wheel"], optional=False),
        DependencyGroup("ml_basic", [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.14.0",
        ], optional=False),
        DependencyGroup("utils", ["tqdm", "psutil", "huggingface_hub"], optional=True),
    ],
    EnvProfile.GPU: [
        DependencyGroup("core", ["pip", "setuptools", "wheel"], optional=False),
        DependencyGroup("torch_cuda", ["torch>=2.0.0"], optional=False),  # Auto-detects CUDA
        DependencyGroup("ml_full", [
            "transformers>=4.30.0",
            "datasets>=2.14.0",
            "accelerate>=0.20.0",
        ], optional=False),
        DependencyGroup("utils", ["tqdm", "psutil", "huggingface_hub"], optional=True),
    ],
    EnvProfile.CLUSTER: [
        DependencyGroup("core", ["pip", "setuptools", "wheel"], optional=False),
        DependencyGroup("torch_distributed", ["torch>=2.0.0"], optional=False),
        DependencyGroup("distributed", [
            "deepspeed",
            "mpi4py",
            "nccl",
        ], optional=True, condition="has_mpi"),
        DependencyGroup("ml_full", [
            "transformers>=4.30.0",
            "datasets>=2.14.0",
            "accelerate>=0.20.0",
        ], optional=False),
    ],
}


class UnifiedEnvManager:
    """
    Unified Environment Manager for UARF
    
    Automatically detects, creates, and manages Python environments
    across all deployment targets.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.venv_path = self.project_root / ".venv"
        self.info = self._detect_environment()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging for environment operations"""
        self.log_level = os.environ.get('UARF_ENV_LOG_LEVEL', 'INFO')
        self.log_file = self.project_root / "uarf_env.log"
    
    def log(self, level: str, message: str, **extra):
        """Structured log output"""
        timestamp = subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], 
                                   capture_output=True, text=True).stdout.strip()
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "component": "env_manager",
            **extra
        }
        
        # Console output
        color_codes = {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[32m',   # Green
            'WARNING': '\033[33m', # Yellow
            'ERROR': '\033[31m',   # Red
            'SUCCESS': '\033[35m'  # Magenta
        }
        reset = '\033[0m'
        color = color_codes.get(level, '')
        
        print(f"{color}[{level}]{reset} {message}", file=sys.stderr)
        
        # File logging
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass
    
    def _detect_environment(self) -> EnvInfo:
        """Detect current environment characteristics"""
        info = EnvInfo()
        
        # Python version
        info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Check if in virtual environment
        info.in_venv = (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        if info.in_venv:
            info.venv_path = sys.prefix
        
        # Get pip version
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.pip_version = result.stdout.split()[1]
        except Exception:
            info.pip_version = "unknown"
        
        # Detect platform
        import platform
        info.platform = platform.system().lower()
        
        # Detect containerization
        info.is_container = (
            os.path.exists('/.dockerenv') or
            os.path.exists('/run/.containerenv') or
            'KUBERNETES_SERVICE_HOST' in os.environ
        )
        
        # Detect KVM
        try:
            if os.path.exists('/dev/kvm'):
                info.is_kvm = True
        except Exception:
            pass
        
        # Detect Google Colab
        info.is_colab = (
            'COLAB_GPU' in os.environ or
            os.path.exists('/content/')
        )
        
        # Detect Termux (Android)
        info.is_termux = (
            'TERMUX_VERSION' in os.environ or
            '/data/data/com.termux' in sys.prefix
        )
        
        # Check internet availability
        info.internet_available = self._check_internet()
        
        # Check if we can create venv
        info.can_create_venv = self._check_venv_capability()
        
        # Detect restrictions
        if info.is_container:
            info.restrictions.append("container")
        if info.is_kvm:
            info.restrictions.append("kvm")
        if not info.internet_available:
            info.restrictions.append("no_internet")
        if not info.can_create_venv:
            info.restrictions.append("no_venv")
        
        return info
    
    def _check_internet(self, timeout: int = 3) -> bool:
        """Check if internet is available"""
        import socket
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except Exception:
            return False
    
    def _check_venv_capability(self) -> bool:
        """Check if we can create virtual environments"""
        # Try to import venv module
        try:
            import venv
            return True
        except ImportError:
            pass
        
        # Try to create a test venv
        try:
            test_dir = self.project_root / ".venv_test"
            subprocess.run(
                [sys.executable, '-m', 'venv', str(test_dir)],
                capture_output=True, timeout=10
            )
            if test_dir.exists():
                shutil.rmtree(test_dir)
                return True
        except Exception:
            pass
        
        return False
    
    def get_recommended_profile(self) -> EnvProfile:
        """Get recommended environment profile based on detected capabilities"""
        # If already in venv with packages, use current setup
        if self.info.in_venv:
            # Check what's installed
            if self._is_package_installed('torch'):
                if self._has_gpu():
                    return EnvProfile.GPU
                return EnvProfile.STANDARD
            return EnvProfile.LIGHT
        
        # Resource-constrained environments
        if self.info.is_termux:
            return EnvProfile.TINY
        
        # Container with limited resources
        if self.info.is_container and not self.info.internet_available:
            return EnvProfile.TINY
        
        # GPU available
        if self._has_gpu():
            return EnvProfile.GPU
        
        # Cluster detection
        if self._is_cluster():
            return EnvProfile.CLUSTER
        
        # Default to standard
        return EnvProfile.STANDARD
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        # Check NVIDIA GPU
        if shutil.which('nvidia-smi'):
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return True
            except Exception:
                pass
        
        # Check CUDA
        cuda_paths = ['/usr/local/cuda', '/opt/cuda']
        for path in cuda_paths:
            if os.path.exists(path):
                return True
        
        return False
    
    def _is_cluster(self) -> bool:
        """Detect if running in a cluster environment"""
        cluster_indicators = [
            'SLURM_JOB_ID',
            'PBS_JOBID',
            'LSB_JOBID',
            'OMPI_COMM_WORLD_SIZE',
            'MV2_COMM_WORLD_SIZE',
        ]
        return any(ind in os.environ for ind in cluster_indicators)
    
    def _is_package_installed(self, package: str) -> bool:
        """Check if a package is installed"""
        try:
            import importlib.util
            spec = importlib.util.find_spec(package.replace('-', '_'))
            return spec is not None
        except Exception:
            return False
    
    def ensure_environment(self, profile: Optional[EnvProfile] = None) -> Tuple[bool, str]:
        """
        Ensure proper environment is active
        
        Args:
            profile: Desired environment profile (auto-detected if None)
        
        Returns:
            Tuple of (success, message)
        """
        if profile is None:
            profile = self.get_recommended_profile()
        
        self.log('INFO', f"Ensuring environment with profile: {profile.value}")
        self.log('DEBUG', "Environment detection results", **{
            'in_venv': self.info.in_venv,
            'platform': self.info.platform,
            'is_container': self.info.is_container,
            'is_kvm': self.info.is_kvm,
            'internet': self.info.internet_available,
            'restrictions': self.info.restrictions
        })
        
        # If already in suitable venv, just install dependencies
        if self.info.in_venv:
            self.log('INFO', "Virtual environment already active", 
                    venv_path=sys.prefix)
            success, msg = self._install_dependencies(profile)
            return success, msg
        
        # Need to create venv
        if not self.info.can_create_venv:
            self.log('WARNING', "Cannot create venv, using system Python")
            success, msg = self._install_dependencies(profile, use_system=True)
            return success, msg
        
        # Create new venv
        self.log('INFO', f"Creating virtual environment at {self.venv_path}")
        success, msg = self._create_venv()
        if not success:
            return False, msg
        
        # Install dependencies in new venv
        success, msg = self._install_dependencies(profile, venv_path=str(self.venv_path))
        if not success:
            return False, msg
        
        # Suggest re-execution in new venv
        if not self._is_in_new_venv():
            self.log('SUCCESS', "Environment setup complete!")
            self.log('INFO', f"To activate: source {self.venv_path}/bin/activate")
            self.log('INFO', "Or re-run this command to auto-activate")
            
            # Auto-reexec if possible
            return self._reexec_in_venv(profile)
        
        return True, "Environment ready"
    
    def _create_venv(self) -> Tuple[bool, str]:
        """Create virtual environment"""
        try:
            # Use python -m venv for compatibility
            result = subprocess.run(
                [sys.executable, '-m', 'venv', str(self.venv_path), 
                 '--clear', '--symlinks'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                # Try without symlinks
                result = subprocess.run(
                    [sys.executable, '-m', 'venv', str(self.venv_path), '--clear'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            if result.returncode == 0:
                self.log('SUCCESS', "Virtual environment created")
                return True, "venv created"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                self.log('ERROR', f"Failed to create venv: {error_msg}")
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            self.log('ERROR', "Timeout creating venv")
            return False, "Timeout"
        except Exception as e:
            self.log('ERROR', f"Exception creating venv: {str(e)}")
            return False, str(e)
    
    def _install_dependencies(self, profile: EnvProfile, 
                             venv_path: Optional[str] = None,
                             use_system: bool = False) -> Tuple[bool, str]:
        """Install dependencies for the given profile"""
        
        # Determine which Python/pip to use
        if venv_path:
            if os.name == 'nt':  # Windows
                pip_path = str(Path(venv_path) / 'Scripts' / 'pip')
                python_path = str(Path(venv_path) / 'Scripts' / 'python')
            else:
                pip_path = str(Path(venv_path) / 'bin' / 'pip')
                python_path = str(Path(venv_path) / 'bin' / 'python')
        elif use_system:
            pip_path = shutil.which('pip') or shutil.which('pip3')
            python_path = sys.executable
        else:
            pip_path = shutil.which('pip') or shutil.which('pip3')
            python_path = sys.executable
        
        if not pip_path:
            self.log('ERROR', "pip not found")
            return False, "pip not found"
        
        # Ensure pip_path is string
        pip_path = str(pip_path)
        python_path = str(python_path)
        
        # Upgrade pip first
        self.log('INFO', "Upgrading pip...")
        try:
            subprocess.run(
                [python_path, '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True,
                timeout=120
            )
        except Exception as e:
            self.log('WARNING', f"Could not upgrade pip: {e}")
        
        # Get dependency groups for profile
        dep_groups = DEPENDENCY_GROUPS.get(profile, [])
        
        for group in dep_groups:
            # Check conditions
            if group.condition:
                if group.condition == "has_mpi" and not self._has_mpi():
                    self.log('DEBUG', f"Skipping {group.name}: MPI not available")
                    continue
            
            # Skip optional groups if no internet
            if group.optional and not self.info.internet_available:
                self.log('DEBUG', f"Skipping optional {group.name}: no internet")
                continue
            
            self.log('INFO', f"Installing {group.name}...")
            success = self._pip_install(pip_path, group.packages, group.optional)
            
            if not success and not group.optional:
                return False, f"Failed to install {group.name}"
        
        self.log('SUCCESS', f"All dependencies for {profile.value} installed")
        return True, "Dependencies installed"
    
    def _pip_install(self, pip_path: str, packages: List[str], 
                     optional: bool = False) -> bool:
        """Run pip install with error handling"""
        try:
            # Build command
            cmd = [pip_path, 'install', '--no-input']
            
            # Add timeout via environment
            env = os.environ.copy()
            env['PIP_DEFAULT_TIMEOUT'] = '60'
            
            # For optional packages, don't fail hard
            if optional:
                cmd.append('--ignore-installed')
            
            cmd.extend(packages)
            
            self.log('DEBUG', f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )
            
            if result.returncode == 0:
                self.log('SUCCESS', f"Installed: {packages[:2]}...")
                return True
            else:
                error_msg = result.stderr or result.stdout
                if optional:
                    self.log('WARNING', f"Optional install failed: {error_msg[:200]}")
                    return True  # Don't fail for optional
                else:
                    self.log('ERROR', f"Install failed: {error_msg[:200]}")
                    return False
                    
        except subprocess.TimeoutExpired:
            msg = f"Timeout installing {packages}"
            if optional:
                self.log('WARNING', msg)
                return True
            else:
                self.log('ERROR', msg)
                return False
        except Exception as e:
            msg = f"Exception: {str(e)}"
            if optional:
                self.log('WARNING', msg)
                return True
            else:
                self.log('ERROR', msg)
                return False
    
    def _has_mpi(self) -> bool:
        """Check if MPI is available"""
        return shutil.which('mpirun') is not None
    
    def _is_in_new_venv(self) -> bool:
        """Check if currently running in the newly created venv"""
        if not self.venv_path.exists():
            return False
        return sys.prefix == str(self.venv_path)
    
    def _reexec_in_venv(self, profile: EnvProfile) -> Tuple[bool, str]:
        """Re-execute current script in new venv"""
        if os.name == 'nt':
            python_exe = str(self.venv_path / 'Scripts' / 'python.exe')
        else:
            python_exe = str(self.venv_path / 'bin' / 'python')
        
        if not os.path.exists(python_exe):
            return False, "New Python executable not found"
        
        self.log('INFO', f"Re-executing in new venv: {python_exe}")
        
        # Set environment to indicate we've already setup
        env = os.environ.copy()
        env['UARF_ENV_SETUP_DONE'] = '1'
        env['UARF_PROFILE'] = profile.value
        
        # Build proper command - use module execution
        # Get the script being executed or module name
        script_path = os.path.abspath(sys.argv[0])
        
        # If running as module (-m uarf.cli.uarf_cli), preserve that
        if '-m' in sys.argv:
            idx = sys.argv.index('-m')
            if idx + 1 < len(sys.argv):
                module_name = sys.argv[idx + 1]
                args = [python_exe, '-m', module_name] + sys.argv[1:]
            else:
                args = [python_exe] + sys.argv[1:]
        # If running as script directly
        elif script_path.endswith('.py'):
            args = [python_exe, script_path] + sys.argv[1:]
        else:
            # Fallback: just run with same args
            args = [python_exe] + sys.argv[1:]
        
        try:
            os.execve(python_exe, args, env)
        except Exception as e:
            self.log('ERROR', f"Re-exec failed: {e}")
            return False, str(e)
        
        return True, "Re-executed"
    
    def get_activation_command(self) -> str:
        """Get command to activate the environment"""
        if os.name == 'nt':
            return f"{self.venv_path}\\Scripts\\activate"
        else:
            return f"source {self.venv_path}/bin/activate"
    
    def print_summary(self):
        """Print environment summary"""
        print("\n" + "="*60)
        print("UARF Environment Summary")
        print("="*60)
        print(f"Python: {self.info.python_version}")
        print(f"Pip: {self.info.pip_version}")
        print(f"In Virtual Env: {'Yes' if self.info.in_venv else 'No'}")
        if self.info.in_venv:
            print(f"Venv Path: {self.info.venv_path}")
        print(f"Platform: {self.info.platform}")
        print(f"Container: {'Yes' if self.info.is_container else 'No'}")
        print(f"KVM: {'Yes' if self.info.is_kvm else 'No'}")
        print(f"Colab: {'Yes' if self.info.is_colab else 'No'}")
        print(f"Termux: {'Yes' if self.info.is_termux else 'No'}")
        print(f"Internet: {'Yes' if self.info.internet_available else 'No'}")
        print(f"Can Create Venv: {'Yes' if self.info.can_create_venv else 'No'}")
        if self.info.restrictions:
            print(f"Restrictions: {', '.join(self.info.restrictions)}")
        
        recommended = self.get_recommended_profile()
        print(f"\nRecommended Profile: {recommended.value}")
        print("="*60 + "\n")


def main():
    """CLI entry point for environment management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='UARF Environment Manager')
    parser.add_argument('--profile', choices=['tiny', 'light', 'standard', 'gpu', 'cluster'],
                       help='Environment profile')
    parser.add_argument('--info', action='store_true', help='Show environment info')
    parser.add_argument('--ensure', action='store_true', help='Ensure environment is ready')
    parser.add_argument('--activate', action='store_true', help='Print activation command')
    
    args = parser.parse_args()
    
    manager = UnifiedEnvManager()
    
    if args.info or not (args.ensure or args.activate):
        manager.print_summary()
    
    if args.ensure:
        profile = None
        if args.profile:
            profile = EnvProfile(args.profile)
        
        success, msg = manager.ensure_environment(profile)
        if success:
            print(f"\n✅ Environment ready: {msg}")
            sys.exit(0)
        else:
            print(f"\n❌ Environment setup failed: {msg}")
            sys.exit(1)
    
    if args.activate:
        if manager.info.in_venv:
            print("# Already in virtual environment")
        elif manager.venv_path.exists():
            print(manager.get_activation_command())
        else:
            print("# No virtual environment found. Run --ensure first.")


if __name__ == '__main__':
    main()

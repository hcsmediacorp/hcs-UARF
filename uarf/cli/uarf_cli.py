#!/usr/bin/env python3
"""
UARF Command Line Interface - Unified Entry Point

One command for all platforms. Automatic environment management.
Works in minimal VMs, KVM, containers, and clusters.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports when running as module
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


def _ensure_environment(profile: str = None, skip_env: bool = False):
    """Ensure proper environment is active before running commands"""
    if skip_env or os.environ.get('UARF_ENV_SETUP_DONE') == '1':
        return True
    
    from uarf.utils.env_manager import UnifiedEnvManager, EnvProfile
    
    manager = UnifiedEnvManager()
    
    # Convert profile string to enum
    env_profile = None
    if profile:
        try:
            env_profile = EnvProfile(profile)
        except ValueError:
            print(f"⚠️  Unknown profile '{profile}', using auto-detection")
    
    # Ensure environment
    success, msg = manager.ensure_environment(env_profile)
    
    if not success:
        print(f"⚠️  Environment setup incomplete: {msg}")
        print("   Continuing with limited functionality...")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        prog='uarf',
        description="Universal AutoResearch Framework - LLM Training on ANY device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uarf auto                      Full auto mode: detect + suggest + train
  uarf run                       Train with defaults
  uarf run --model mistralai/Mistral-7B-v0.1 --time 600
  uarf suggest                   Show model recommendations for your hardware
  uarf detect --json             Output hardware info as JSON
  uarf env --info                Show environment status
  uarf env --ensure --profile gpu  Setup GPU environment
  uarf export --checkpoint ./outputs/model --format gguf

Environment Profiles:
  tiny      - Minimal deps, pure Python fallbacks (<512MB RAM)
  light     - CPU-only torch, basic transformers (512MB-2GB RAM)
  standard  - Full features, CPU or GPU (2GB+ RAM)
  gpu       - CUDA-enabled, all optimizations
  cluster   - Distributed training support

Quick Start:
  Just run: uarf auto
  That's it. The framework handles everything.
        """
    )
    
    # Global options
    parser.add_argument('--version', action='version', version='UARF v2.0.0')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    parser.add_argument('--no-env', action='store_true',
                       help='Skip automatic environment setup')
    parser.add_argument('--profile', choices=['tiny', 'light', 'standard', 'gpu', 'cluster'],
                       default=None,
                       help='Environment profile (default: auto-detect)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # auto command - full automation
    auto_parser = subparsers.add_parser('auto', 
                                        help='Full auto mode: detect + suggest + train')
    auto_parser.add_argument('--text', type=str, default=None,
                            help='Training text (or use demo data)')
    auto_parser.add_argument('--time', type=int, default=5,
                            help='Training time in minutes (default: 5)')
    auto_parser.add_argument('--output-dir', type=str, default='./outputs',
                            help='Output directory')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Start training')
    run_parser.add_argument('--model', type=str, default=None,
                           help='Hugging Face Model ID (auto-selected if not provided)')
    run_parser.add_argument('--dataset', type=str, default=None,
                           help='Dataset path or Hugging Face name')
    run_parser.add_argument('--time', type=int, default=300,
                           help='Time budget in seconds (default: 300)')
    run_parser.add_argument('--batch-size', type=int, default=None,
                           help='Batch size (auto if not specified)')
    run_parser.add_argument('--max-seq-len', type=int, default=None,
                           help='Max sequence length')
    run_parser.add_argument('--lr', type=float, default=2e-4,
                           help='Learning rate')
    run_parser.add_argument('--output-dir', type=str, default='./outputs',
                           help='Output directory')
    run_parser.add_argument('--config', type=str, default=None,
                           help='Path to config JSON file')
    run_parser.add_argument('--device', type=str, default='auto',
                           choices=['auto', 'cuda', 'cpu', 'mps'],
                           help='Device selection')
    run_parser.add_argument('--precision', type=str, default='auto',
                           choices=['auto', 'fp32', 'fp16', 'bf16', 'int8'],
                           help='Precision')
    run_parser.add_argument('--demo', action='store_true',
                           help='Run demo with synthetic data')
    
    # suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Show model recommendations')
    suggest_parser.add_argument('--task', type=str, default='text-generation',
                               choices=['text-generation', 'classification', 
                                       'qa', 'summarization', 'translation'],
                               help='Task type')
    suggest_parser.add_argument('--limit', type=int, default=5,
                               help='Max recommendations')
    suggest_parser.add_argument('--min-ram', type=float, default=None,
                               help='Minimum RAM in GB')
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export trained model')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to checkpoint')
    export_parser.add_argument('--format', type=str, required=True,
                              choices=['gguf', 'onnx', 'tflite', 'safetensors'],
                              help='Export format')
    export_parser.add_argument('--output', type=str, default=None,
                              help='Output file path')
    export_parser.add_argument('--quantization', type=str, default='Q4_K_M',
                              help='Quantization for GGUF')
    
    # detect command
    detect_parser = subparsers.add_parser('detect', help='Detect hardware')
    detect_parser.add_argument('--json', action='store_true',
                              help='Output as JSON')
    detect_parser.add_argument('--verbose', action='store_true',
                              help='Detailed output')
    
    # env command - environment management
    env_parser = subparsers.add_parser('env', help='Environment management')
    env_parser.add_argument('--info', action='store_true',
                           help='Show environment info')
    env_parser.add_argument('--ensure', action='store_true',
                           help='Ensure environment is ready')
    env_parser.add_argument('--activate', action='store_true',
                           help='Print activation command')
    env_parser.add_argument('--profile', choices=['tiny', 'light', 'standard', 'gpu', 'cluster'],
                           help='Environment profile')
    env_parser.add_argument('--clean', action='store_true',
                           help='Remove virtual environment')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Handle env command separately (doesn't need env setup first)
    if args.command == 'env':
        _handle_env_command(args)
        return
    
    # Ensure environment for other commands
    if not args.no_env:
        _ensure_environment(args.profile, skip_env=False)
    
    # Execute commands
    if args.command == 'auto':
        _handle_auto_command(args)
    
    elif args.command == 'run':
        _handle_run_command(args)
    
    elif args.command == 'suggest':
        _handle_suggest_command(args)
    
    elif args.command == 'export':
        _handle_export_command(args)
    
    elif args.command == 'detect':
        _handle_detect_command(args)


def _handle_env_command(args):
    """Handle environment management command"""
    from uarf.utils.env_manager import UnifiedEnvManager, EnvProfile
    
    manager = UnifiedEnvManager()
    
    if args.clean:
        # Remove venv
        import shutil
        if manager.venv_path.exists():
            print(f"Removing virtual environment at {manager.venv_path}...")
            shutil.rmtree(manager.venv_path)
            print("✅ Environment removed")
        else:
            print("No virtual environment found")
        return
    
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
            print("# No virtual environment found. Run 'uarf env --ensure' first.")
        return
    
    # Default: show info
    manager.print_summary()


def _handle_auto_command(args):
    """Handle full auto mode command"""
    from uarf.controller import UARFController
    
    print("="*60)
    print("UARF Auto Mode - Full Automation")
    print("="*60)
    
    controller = UARFController(debug=os.environ.get('UARF_DEBUG', False))
    
    # Detect hardware
    print("\n🔍 Detecting hardware...")
    detect_result = controller.detect()
    if not detect_result.success:
        print(f"⚠️  Detection warning: {detect_result.message}")
    
    # Suggest model
    print("\n💡 Suggesting best model...")
    suggest_result = controller.suggest(task="text-generation")
    if suggest_result.success:
        model_info = suggest_result.data.get('suggested_model', {})
        print(f"   Recommended: {model_info.get('model', 'unknown')}")
        print(f"   Parameters: {model_info.get('params_millions', 'unknown')}M")
    
    # Configure training
    print("\n⚙️  Configuring training...")
    controller.update_config(
        time_budget_seconds=args.time * 60,
        output_dir=args.output_dir,
    )
    
    # Run training (or demo)
    print(f"\n🚀 Starting training for {args.time} minutes...")
    print(f"   Output: {args.output_dir}")
    
    # Note: Actual training implementation would go here
    result = controller.run_training(text=args.text)
    
    if result.success:
        print(f"\n✅ Training completed!")
        print(f"   Output: {result.data.get('output_path', 'N/A')}")
    else:
        print(f"\n⚠️  Training completed with warnings: {result.message}")
    
    print("="*60)


def _handle_run_command(args):
    """Handle run command"""
    from uarf.core.hardware_detector import HardwareDetector
    from uarf.core.config import UARFConfig
    from uarf.core.trainer import UniversalTrainer
    
    print("="*60)
    print("UARF Training Run")
    print("="*60)
    
    # Detect hardware
    print("\n🔍 Detecting hardware...")
    detector = HardwareDetector()
    detector.print_summary()
    
    # Create or load config
    if args.config:
        print(f"\n📄 Loading config from {args.config}")
        config = UARFConfig.from_json(args.config)
    else:
        print("\n⚙️  Creating configuration...")
        config = UARFConfig(
            model_id=args.model or UARFConfig.model_id,
            dataset_name=args.dataset or UARFConfig.dataset_name,
            time_budget_seconds=args.time,
            device=args.device,
            output_dir=args.output_dir,
            learning_rate=args.lr,
        )
        
        # Apply hardware-based optimizations
        hardware_config = detector.get_optimal_config()
        
        if args.batch_size:
            config.batch_size = args.batch_size
        elif 'batch_size' in hardware_config:
            config.batch_size = hardware_config['batch_size']
        
        if args.max_seq_len:
            config.max_seq_len = args.max_seq_len
        elif 'max_seq_len' in hardware_config:
            config.max_seq_len = hardware_config['max_seq_len']
        
        # Apply low-RAM profile if needed
        if detector.specs.ram_available < 2:
            print("   📉 Low RAM detected, applying optimizations...")
            config.update_from_hardware(hardware_config)
    
    # Show config
    config.print_summary()
    
    # Demo mode
    if args.demo:
        print("\n🎭 Running in DEMO mode with synthetic data...")
        _run_demo_mode(config, detector)
        return
    
    # Start training
    print("\n🚀 Starting training...")
    try:
        trainer = UniversalTrainer(config)
        trainer.train()
        print("\n✅ Training completed!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if os.environ.get('UARF_DEBUG'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_demo_mode(config, detector):
    """Run demo mode with synthetic data"""
    import torch
    import time
    
    # Get device - use device_manager for this
    from uarf.core.device_manager import DeviceManager
    dm = DeviceManager(config.device or "auto")
    device = dm.device
    print(f"   Device: {device}")
    
    # Create tiny model
    from uarf_run import create_tiny_model
    
    vocab_size = 8192
    model = create_tiny_model(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        max_seq_len=config.max_seq_len or 64
    )
    model = model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Synthetic dataset
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, size, seq_len, vocab_size):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = vocab_size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
                'attention_mask': torch.ones(self.seq_len)
            }
    
    train_loader = torch.utils.data.DataLoader(
        SyntheticDataset(1000, config.max_seq_len or 64, vocab_size),
        batch_size=config.batch_size or 8,
        shuffle=True
    )
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps or 100
    )
    
    model.train()
    start_time = time.time()
    global_step = 0
    
    print("\n   Training...")
    for epoch in range(min(3, config.time_budget_seconds // 30)):
        for batch in train_loader:
            if time.time() - start_time > config.time_budget_seconds:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()
            
            loss = model(input_ids=input_ids, targets=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 10 == 0:
                print(f"   Step {global_step}: loss={loss.item():.4f}")
        
        if time.time() - start_time > config.time_budget_seconds:
            break
    
    elapsed = time.time() - start_time
    print(f"\n   ✅ Demo complete: {global_step} steps in {elapsed:.1f}s")


def _handle_suggest_command(args):
    """Handle suggest command"""
    from uarf.core.hardware_detector import HardwareDetector
    from uarf.core.model_selector import ModelSelector
    
    print("="*60)
    print("UARF Model Recommendations")
    print("="*60)
    
    detector = HardwareDetector()
    detector.print_summary()
    
    selector = ModelSelector(detector.specs)
    selector.print_suggestions(args.task)


def _handle_export_command(args):
    """Handle export command"""
    import torch
    from pathlib import Path
    
    print("="*60)
    print("UARF Model Export")
    print("="*60)
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n📤 Exporting model...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Format: {args.format}")
    print(f"   Quantization: {args.quantization}")
    
    # Find training state file
    training_state_file = checkpoint_path / 'training_state.pt'
    if not training_state_file.exists():
        # Try to find any .pt file
        pt_files = list(checkpoint_path.glob('*.pt'))
        if pt_files:
            training_state_file = pt_files[0]
        else:
            print(f"⚠️  No training state file found, using checkpoint directly")
    
    # Load model
    print("\n   Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.format == 'gguf':
            output_name = f"model-{args.quantization.lower()}.gguf"
        else:
            output_name = f"model.{args.format}"
        output_path = str(checkpoint_path.parent / output_name)
    
    # Export based on format
    if args.format == 'gguf':
        try:
            from uarf.exports.gguf import export_to_gguf
            
            training_state = {}
            if training_state_file.exists():
                training_state = torch.load(training_state_file, map_location='cpu')
            
            export_to_gguf(
                model_state=model.state_dict(),
                config=training_state.get('config', {}),
                output_path=output_path,
                quantization=args.quantization
            )
            print(f"\n✅ Export successful: {output_path}")
        except ImportError:
            print(f"\n⚠️  GGUF export module not available")
            print("   Install with: pip install llama-cpp-python")
        except Exception as e:
            print(f"\n❌ Export failed: {e}")
            sys.exit(1)
    
    elif args.format == 'safetensors':
        try:
            output_dir = Path(output_path).parent
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"\n✅ Export successful: {output_dir}")
        except Exception as e:
            print(f"\n❌ Export failed: {e}")
            sys.exit(1)
    
    else:
        print(f"\n⚠️  Format '{args.format}' is not fully supported yet.")
        print("   Supported: gguf, safetensors")


def _handle_detect_command(args):
    """Handle detect command"""
    from uarf.core.hardware_detector import HardwareDetector
    
    detector = HardwareDetector()
    
    if args.json:
        import json
        specs_dict = {
            'platform': detector.specs.platform,
            'cpu_count': detector.specs.cpu_count,
            'ram_total_gb': round(detector.specs.ram_total, 2),
            'ram_available_gb': round(detector.specs.ram_available, 2),
            'gpu_available': detector.specs.gpu_available,
            'gpu_name': detector.specs.gpu_name,
            'gpu_vram_gb': round(detector.specs.gpu_vram, 2) if detector.specs.gpu_vram else 0,
            'is_mobile': detector.specs.is_mobile,
            'is_colab': detector.specs.is_colab,
            'is_cluster': detector.specs.is_cluster,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        print(json.dumps(specs_dict, indent=2))
    else:
        detector.print_summary()


if __name__ == '__main__':
    main()

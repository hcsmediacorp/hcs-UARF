"""
UARF Checkpoint Manager - Resume Training Support
Production-ready checkpoint save/load with validation
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import shutil

from ..utils.exceptions import CheckpointError, ResourceExhaustedError


class CheckpointManager:
    """
    Manages training checkpoints with resume support
    
    Features:
    - Save model, optimizer, scheduler states
    - Save training metrics and config
    - Validate checkpoint integrity
    - Auto-cleanup of old checkpoints
    - Resume from any valid checkpoint
    """
    
    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoint_history = []
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        model,
        tokenizer,
        optimizer,
        scheduler,
        global_step: int,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        is_best: bool = False,
    ) -> str:
        """
        Save a complete training checkpoint
        
        Args:
            model: The model to save
            tokenizer: The tokenizer
            optimizer: Optimizer state
            scheduler: Scheduler state
            global_step: Current training step
            metrics: Training metrics
            config: Configuration dictionary
            is_best: Whether this is the best checkpoint so far
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint-{global_step:06d}_{timestamp}"
        checkpoint_path = self.output_dir / checkpoint_name
        
        try:
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save training state
            training_state = {
                'global_step': global_step,
                'metrics': metrics,
                'config': config,
                'timestamp': timestamp,
                'is_best': is_best,
            }
            
            if self.save_optimizer and optimizer is not None:
                training_state['optimizer'] = optimizer.state_dict()
            
            if scheduler is not None:
                training_state['scheduler'] = scheduler.state_dict()
            
            torch.save(training_state, checkpoint_path / 'training_state.pt')
            
            # Save metadata
            metadata = {
                'checkpoint_name': checkpoint_name,
                'global_step': global_step,
                'timestamp': timestamp,
                'metrics': metrics,
                'is_best': is_best,
            }
            with open(checkpoint_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'path': str(checkpoint_path),
                'step': global_step,
                'timestamp': timestamp,
                'metrics': metrics.copy(),
            })
            
            # Track best checkpoint
            if is_best or ('val_loss' in metrics and metrics['val_loss'] < self.best_val_loss):
                self.best_val_loss = metrics.get('val_loss', float('inf'))
                self.best_checkpoint_path = str(checkpoint_path)
                
                # Create symlink to best checkpoint
                best_link = self.output_dir / 'best'
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                os.symlink(checkpoint_path, best_link)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except torch.cuda.OutOfMemoryError:
            raise ResourceExhaustedError(
                "GPU out of memory while saving checkpoint",
                details={'checkpoint_path': str(checkpoint_path)}
            )
        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint: {str(e)}",
                details={'checkpoint_path': str(checkpoint_path)}
            )
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model=None,
        tokenizer=None,
        optimizer=None,
        scheduler=None,
    ) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        """
        Load a training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load weights into
            tokenizer: Tokenizer to load
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
        
        Returns:
            Tuple of (model, tokenizer, optimizer, scheduler, training_state)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise CheckpointError(
                f"Checkpoint path does not exist: {checkpoint_path}",
                details={'path': str(checkpoint_path)}
            )
        
        training_state_file = checkpoint_path / 'training_state.pt'
        if not training_state_file.exists():
            raise CheckpointError(
                f"Training state file not found: {training_state_file}",
                details={'path': str(checkpoint_path)}
            )
        
        try:
            # Load training state
            training_state = torch.load(training_state_file, map_location='cpu')
            
            # Load model and tokenizer
            if model is not None:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            
            if tokenizer is not None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
            # Load optimizer state
            if optimizer is not None and 'optimizer' in training_state:
                optimizer.load_state_dict(training_state['optimizer'])
            
            # Load scheduler state
            if scheduler is not None and 'scheduler' in training_state:
                scheduler.load_state_dict(training_state['scheduler'])
            
            return model, tokenizer, optimizer, scheduler, training_state
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint: {str(e)}",
                details={'checkpoint_path': str(checkpoint_path)}
            )
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the most recent checkpoint path"""
        if not self.checkpoint_history:
            # Scan directory for checkpoints
            checkpoints = sorted(self.output_dir.glob('checkpoint-*'))
            if checkpoints:
                return str(checkpoints[-1])
            return None
        
        return self.checkpoint_history[-1]['path']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the best checkpoint path (lowest validation loss)"""
        if self.best_checkpoint_path:
            return self.best_checkpoint_path
        
        # Try to find via symlink
        best_link = self.output_dir / 'best'
        if best_link.exists() and best_link.is_symlink():
            return str(best_link.resolve())
        
        return None
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_dir in self.output_dir.glob('checkpoint-*'):
            metadata_file = checkpoint_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
        
        return sorted(checkpoints, key=lambda x: x.get('global_step', 0))
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by step
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: x['step'],
            reverse=True
        )
        
        # Keep only max_checkpoints
        to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        for checkpoint in to_remove:
            try:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                
                self.checkpoint_history.remove(checkpoint)
            except Exception as e:
                print(f"Warning: Failed to remove old checkpoint {checkpoint['path']}: {e}")
    
    def export_for_inference(self, checkpoint_path: str, export_dir: str):
        """
        Export a checkpoint for inference (smaller size, no optimizer state)
        
        Args:
            checkpoint_path: Path to training checkpoint
            export_dir: Directory to export to
        """
        checkpoint_path = Path(checkpoint_path)
        export_dir = Path(export_dir)
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy model and tokenizer only
        for item in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
            src = checkpoint_path / item
            if src.exists():
                shutil.copy(src, export_dir / item)
        
        # Copy model weights
        for item in checkpoint_path.glob('*.safetensors'):
            shutil.copy(item, export_dir / item)
        
        # Also support pytorch_model.bin
        for item in checkpoint_path.glob('pytorch_model*.bin'):
            shutil.copy(item, export_dir / item)
        
        # Create inference config
        inference_config = {
            'exported_from': str(checkpoint_path),
            'export_date': datetime.now().isoformat(),
            'inference_ready': True,
        }
        
        with open(export_dir / 'inference_config.json', 'w') as f:
            json.dump(inference_config, f, indent=2)

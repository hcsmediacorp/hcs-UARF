#!/usr/bin/env python3
import os, sys, argparse, time, math
from pathlib import Path

try:
    import torch
except ImportError:
    print('PyTorch not installed!')
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def detect_hardware():
    ram_gb = 0
    gpu_available = False
    gpu_name = None
    gpu_vram = 0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except: pass
    if torch.cuda.is_available():
        gpu_available = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_available = True
        gpu_name = 'Apple Silicon'
    return {'ram_gb': ram_gb, 'gpu_available': gpu_available, 'gpu_name': gpu_name, 'gpu_vram': gpu_vram, 'device': 'cuda' if gpu_available else 'cpu'}

def create_tiny_model(vocab_size=8192, d_model=256, n_layers=4, n_heads=8, max_seq_len=512):
    import torch.nn as nn
    import torch.nn.functional as F
    class TinyTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, activation='gelu', batch_first=True, norm_first=True) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        def forward(self, input_ids, attention_mask=None, targets=None):
            B, T = input_ids.shape
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
            h = self.token_emb(input_ids) + self.pos_emb(positions)
            for layer in self.layers: h = layer(h)
            h = self.norm(h)
            logits = self.lm_head(h)
            if targets is not None: return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits
    return TinyTransformer(vocab_size, d_model, n_layers, n_heads, max_seq_len)

def train_simple(model, tokenizer, train_loader, val_loader, device, config):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_steps'])
    model.train()
    progress_bar = tqdm(total=config['max_steps'], desc='Training')
    global_step, best_val_loss, start_time = 0, float('inf'), time.time()
    data_iter = iter(train_loader)
    while global_step < config['max_steps']:
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(train_loader); batch = next(data_iter)
        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone()
        loss = model(input_ids=input_ids, targets=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        global_step += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        progress_bar.update(1)
        elapsed = time.time() - start_time
        if elapsed >= config['time_budget']: break
    progress_bar.close()
    return {'steps': global_step, 'best_val_loss': best_val_loss, 'training_time': time.time() - start_time}

def main():
    parser = argparse.ArgumentParser(description='UARF')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--time', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-seq-len', type=int, default=64)
    args = parser.parse_args()
    print('='*60)
    print('UARF - Universal AutoResearch Framework')
    print('='*60)
    hardware = detect_hardware()
    print(f"RAM: {hardware['ram_gb']:.1f} GB, Device: {hardware['device']}")
    print('
Creating tiny demo model...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    tokenizer.pad_token = tokenizer.eos_token
    model = create_tiny_model(vocab_size=min(tokenizer.vocab_size, 8192), d_model=256, n_layers=4, n_heads=8, max_seq_len=args.max_seq_len)
    print(f'Created model: {sum(p.numel() for p in model.parameters()):,} parameters')
    
    class SyntheticDataset:
        def __init__(self, size, seq_len, vocab_size):
            self.size, self.seq_len, self.vocab_size = size, seq_len, vocab_size
        def __len__(self): return self.size
        def __getitem__(self, idx): return {'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)), 'attention_mask': torch.ones(self.seq_len)}
    
    train_loader = DataLoader(SyntheticDataset(1000, args.max_seq_len, min(tokenizer.vocab_size, 8192)), batch_size=args.batch_size, shuffle=True)
    config = {'batch_size': args.batch_size, 'max_seq_len': args.max_seq_len, 'lr': 2e-4, 'max_steps': max(100, args.time*2), 'time_budget': args.time}
    print(f'
Starting training for {args.time}s...')
    metrics = train_simple(model, tokenizer, train_loader, None, hardware['device'], config)
    print(f"
Done! Steps: {metrics['steps']}, Time: {metrics['training_time']:.1f}s")
    print('='*60)

if __name__ == '__main__': main()

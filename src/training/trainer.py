import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import os

from src.data.normalization import DisplacementNormalizer
from .losses import combined_loss

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cpu',
        checkpoint_dir: str = 'models/saved',
        normalizer: DisplacementNormalizer = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.normalizer = normalizer
        
        self.epochs = config.get('epochs', 200)
        self.lr = config.get('lr', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.patience = config.get('patience', 20)
        self.physics_loss_weight = config.get('physics_loss_weight', 0.0)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            pred = self.model(batch)
            target = batch.y
            
            if self.normalizer is not None:
                target = self.normalizer.normalize(target)
            
            # Physics loss: 0 for baseline (Model 1), active for physics-informed (Models 3–4)
            loss = combined_loss(
                pred, 
                target, 
                physics_loss=None, 
                physics_weight=self.physics_loss_weight
            )
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        return {'train_loss': total_loss / len(self.train_loader.dataset)}
        
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch.y
                
                if self.normalizer is not None:
                    target = self.normalizer.normalize(target)
                
                loss = combined_loss(
                    pred, 
                    target, 
                    physics_loss=None, 
                    physics_weight=self.physics_loss_weight
                )
                total_loss += loss.item() * batch.num_graphs
                
        return {'val_loss': total_loss / len(self.val_loader.dataset)}
        
    def train(self) -> dict:
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            train_loss = train_metrics['train_loss']
            val_loss = val_metrics['val_loss']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(best_model_path, epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        history['best_val_loss'] = best_val_loss
        history['best_epoch'] = best_epoch
        return history

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint with optimizer state and normalizer stats."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        if self.normalizer is not None:
            checkpoint['normalizer_mean'] = self.normalizer.mean
            checkpoint['normalizer_std'] = self.normalizer.std
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'normalizer_mean' in checkpoint and self.normalizer is not None:
            self.normalizer.mean = checkpoint['normalizer_mean']
            self.normalizer.std = checkpoint['normalizer_std']
            
        return checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dataset_bbox import BBoxDataset

def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 Loss (Huber Loss)"""
    loss = nn.SmoothL1Loss(beta=beta, reduction='mean')
    return loss(pred, target)

def train_bbox_regressor(
    model,
    train_loader,
    val_loader,
    save_dir,
    epochs=20,
    lr=1e-2,
    weight_decay=1e-3,
    device='cuda',
    patience=10

):
    """
    Train BBox Regressor with early stopping and stronger L2 regularization.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss function
    criterion = nn.SmoothL1Loss(reduction='mean')
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"):
            images = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            
            # Forward
            pred_bboxes = model(images)
            loss = criterion(pred_bboxes, bboxes)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]"):
                images = batch['image'].to(device)
                bboxes = batch['bbox'].to(device)
                
                pred_bboxes = model(images)
                loss = criterion(pred_bboxes, bboxes)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/bbox_regressor_best_early_stopping.pth")
            print(" ✓ (saved)")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
        
        # Scheduler step
        scheduler.step(val_loss)
    
    print(f"\nTraining completed. Best Val Loss={best_val_loss:.4f}")

if __name__ == "__main__":
    from utils.utilities import load_zipped_pickle, make_train_test_split
    from models.BBoxRegressor import BBoxRegressor
    
    # Load data
    data = load_zipped_pickle('./data/raw/train.pkl')
    train_data, val_data = make_train_test_split(data, test_ratio=0.2)
    
    # Create datasets
    train_dataset = BBoxDataset(train_data, target_size=(256, 256))
    val_dataset = BBoxDataset(val_data, target_size=(256, 256))
    
    # Create dataloaders (tuned for throughput)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=pin_mem, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=pin_mem, prefetch_factor=2
    )
    
    # Create model
    model = BBoxRegressor(backbone='resnet18', pretrained=True, dropout=0.5)
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_bbox_regressor(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3,
        device=device,
        save_dir='weights',
        model_name='bbox_regressor'
    )
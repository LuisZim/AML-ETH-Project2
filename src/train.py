from tqdm.auto import tqdm
import torch


# Training scaffold (to be adapted to your dataset loader)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f'Training next epoch', leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

        pbar.set_postfix({'loss': loss.item()})

    return running_loss / len(dataloader.dataset)


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            running_loss += loss.item() * imgs.size(0)

            pbar.set_postfix({'loss': loss.item()})
    return running_loss / len(dataloader.dataset)


def train_model(model, train_loader, optimizer, criterion, device, epochs, save_path, val_loader=None, patience=5):
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss = eval_one_epoch(model, val_loader, criterion, device) if val_loader is not None else None

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}" if val_loss is not None else f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")

        # Early Stopping if we have validation set
        if val_loader is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{save_path}")
                print(" ✓ (saved)")
            else:
                patience_counter += 1
                print(f" (patience {patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
        # Record losses
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
    if val_loader is None:
        torch.save(model.state_dict(), f"{save_path}")
        print(" ✓ (saved final model)")
    return loss_history
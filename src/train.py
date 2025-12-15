from tqdm.auto import tqdm
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

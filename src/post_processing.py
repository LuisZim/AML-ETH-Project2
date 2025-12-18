import torch
def average_logits(logits_list):
    """Averages a list of logits tensors."""
    return sum(logits_list) / len(logits_list)

def get_averaged_logits(model, video, device, temp_window=2):
    """
    Given a model and an echocardiography video (H, W, num_frames),
    returns the predicted masks for each frame by averaging logits over a temporal window.
    
    Args:
        model: The trained segmentation model.
        video: A tensor of shape (H, W, num_frames) representing the echocardiography video.
        temp_window: Number of neighboring frames on each side to include in averaging.
        device: The device to run the model on."""
    model.eval()
    logits_list = []
    num_frames = video.shape[2]
    with torch.no_grad():
        for i in range(num_frames):
            frame = video[:, :, i].to(device)
            frame = frame.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            model = model.to(device)
            logits_list.append(model(frame).squeeze(0).squeeze(0).cpu())  # Remove batch and channel dims
    averaged_masks = []
    for i in range(num_frames):
        start_idx = max(0, i - temp_window)
        end_idx = min(num_frames, i + temp_window + 1)
        relevant_logits = logits_list[start_idx:end_idx]
        averaged_logit = average_logits(relevant_logits)
        averaged_masks.append(averaged_logit.unsqueeze(2))  # Add frame dim
    averaged_masks = torch.cat(averaged_masks, dim=2)  # (H, W, num_frames)
    return averaged_masks
import torch
import yaml
import logging
from tqdm import tqdm
import time
import os
import json

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    lengths = [w.size(1) for w in waveforms]
    max_len = max(lengths)
    padded_waveforms = []
    for w in waveforms:
        pad_len = max_len - w.size(1)
        padded_waveforms.append(torch.nn.functional.pad(w, (0, pad_len)))
    waveforms = torch.stack(padded_waveforms)
    return waveforms


def valid_step(model, loss_fn, val_dataloader, device):
    model.eval()
    total_loss = 0.0
    val_losses = []

    for wave in tqdm(val_dataloader):
        wave = wave.to(device)
        with torch.amp.autocast(device.type):
            preds, targets, negatives, softmax_outputs, mask_indices = model(wave)
            loss = loss_fn(preds, targets, negatives, softmax_outputs, mask_indices)
            val_losses.append(loss.item())

        total_loss += loss.item()
    
    return total_loss / len(val_dataloader), val_losses


def train_step(model, epochs, optimizer, loss_fn, scheduler, train_dataloader, val_dataloader, device, model_name, save_checkpoint_every, checkpoint_dir, start_epoch = 0):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    logging.info("Initializing model training...\n")

    all_train_batch_losses = []
    all_val_batch_losses = []

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        total_loss = 0.0
        batch_losses = []

        for wave in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            wave = wave.to(device)
            with torch.amp.autocast(device.type):
                preds, targets, negatives, softmax_outputs, mask_indices = model(wave)
                loss = loss_fn(preds, targets, negatives, softmax_outputs, mask_indices)
                batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - start_time
        avg_train_loss  = total_loss / len(train_dataloader)
        avg_val_loss, val_losses = valid_step(model, loss_fn, val_dataloader, device)
        all_train_batch_losses.append(batch_losses)
        all_val_batch_losses.append(val_losses)
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"{'Val Loss: {:.4f} - '.format(avg_val_loss) if avg_val_loss is not None else ''}"
            f"Time: {elapsed:.2f}s"
        )
 
        if (epoch + 1) % save_checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, model_name)
            os.makedirs(path, exist_ok=True)
            checkpoint_path = os.path.join(path, f"{model_name}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")

    with open(os.path.join(checkpoint_dir, model_name, f"{model_name}_batch_train_losses.json"), "w") as f:
        json.dump(all_train_batch_losses, f)

    with open(os.path.join(checkpoint_dir, model_name, f"{model_name}_batch_val_losses.json"), "w") as f:
        json.dump(all_val_batch_losses, f)

def data_split(dataset):
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset


def load_pretrained(model, optimizer, scheduler, checkpoint_path):

    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint
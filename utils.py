import torch
import yaml
import logging
from tqdm import tqdm
import time
import os

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

    for wave in tqdm(val_dataloader):
        wave = wave.to(device)
        with torch.amp.autocast(device.type):
            preds, targets, negatives, softmax_outputs, mask_indices = model(wave)
            loss = loss_fn(preds, targets, negatives, softmax_outputs, mask_indices)
        total_loss += loss.item()
    
    return total_loss / len(val_dataloader)


def train_step(model, epochs, optimizer, loss_fn, scheduler, train_dataloader, val_dataloader, device, model_name, save_checkpoint_every, checkpoint_dir):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    logging.info("Initializing model training...\n")

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        total_loss = 0.0

        for wave in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            wave = wave.to(device)
            with torch.amp.autocast(device.type):
                preds, targets, negatives, softmax_outputs, mask_indices = model(wave)
                loss = loss_fn(preds, targets, negatives, softmax_outputs, mask_indices)

            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - start_time
        avg_train_loss  = total_loss / len(train_dataloader)
        avg_val_loss = valid_step(model, loss_fn, val_dataloader, device)
        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"{'Val Loss: {:.4f} - '.format(avg_val_loss) if avg_val_loss is not None else ''}"
            f"Time: {elapsed:.2f}s"
        )
 
        if (epoch + 1) % save_checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")
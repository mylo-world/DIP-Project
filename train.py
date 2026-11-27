import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from losses import CombinedLoss, dice_coefficient



class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = -float("inf")
        self.stop      = False

    def step(self, val_dice: float) -> bool:
        """Returns True when training should stop."""
        if val_dice > self.best + self.min_delta:
            self.best    = val_dice
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_dice = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(logits, masks)

    n = len(loader)
    return total_loss / n, total_dice / n



def train(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    *,
    epochs:       int   = 100,
    lr:           float = 1e-4,
    weight_decay: float = 1e-5,
    patience:     int   = 15,
    checkpoint_dir: str = "checkpoints",
    device = None,
) -> dict:
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)
    model  = model.to(device)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion   = CombinedLoss(smooth=1.0, bce_weight=0.25, pos_weight=6.0)
    optimizer   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler   = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                    patience=7)
    early_stop  = EarlyStopping(patience=patience)

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "lr_history": [], "best_epoch": 0,
    }
    best_val_dice = -float("inf")

    print(f"Training on {device}  |  epochs={epochs}  |  lr={lr}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} "
          f"{'Train Dice':>11} {'Val Dice':>10} {'LR':>10}  {'Time':>6}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_dice = _run_epoch(model, train_loader, criterion,
                                            device, optimizer)
        val_loss,   val_dice   = _run_epoch(model, val_loader,   criterion,
                                            device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dice)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_dice"].append(round(train_dice, 6))
        history["val_dice"].append(round(val_dice, 6))
        history["lr_history"].append(current_lr)

        elapsed = time.time() - t0
        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} "
              f"{train_dice:>11.4f} {val_dice:>10.4f} {current_lr:>10.2e}  "
              f"{elapsed:>5.1f}s")

        # Checkpoint 
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            history["best_epoch"] = epoch
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice":   val_dice,
                    "val_loss":   val_loss,
                },
                ckpt_dir / "best_model.pth",
            )
            print(f"  ✓ checkpoint saved  (val_dice={val_dice:.4f})")

        # Early stopping 
        if early_stop.step(val_dice):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.  Best epoch: {history['best_epoch']}  "
          f"Best val Dice: {best_val_dice:.4f}")
    return history


# CLI entry point

def _parse_args():
    p = argparse.ArgumentParser(description="Train U-Net Mini on Kvasir-SEG")
    p.add_argument("--data_dir",      default="data/Kvasir-SEG")
    p.add_argument("--split_file",    default="data/split.json",
                   help="JSON produced by dataset.py split utility")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--checkpoint_dir",default="checkpoints")
    p.add_argument("--device",        default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from dataloader import get_dataloaders
    from model import UNetMini

    train_loader, val_loader, _ = get_dataloaders(
        data_dir   = args.data_dir,
        split_file = args.split_file,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )

    model = UNetMini(in_channels=9, out_channels=1)
    train(
        model, train_loader, val_loader,
        epochs        = args.epochs,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        patience      = args.patience,
        checkpoint_dir= args.checkpoint_dir,
        device        = args.device,
    )
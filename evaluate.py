import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader



def _compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                     smooth: float = 1.0) -> dict:
    pred = pred_mask.astype(np.float32).ravel()
    gt   = gt_mask.astype(np.float32).ravel()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()

    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    f2        = (5 * precision * recall + smooth) / (4 * precision + recall + smooth)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "dice":      float(dice),
        "iou":       float(iou),
        "recall":    float(recall),
        "precision": float(precision),
        "f2":        float(f2),
        "accuracy":  float(accuracy),
    }



@torch.no_grad()
def evaluate(
    model,
    test_loader: DataLoader,
    *,
    device: str | None = None,
    threshold: float = 0.5,
    results_dir: str = "results",
    sample_names: list[str] | None = None,
) -> dict:
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)
    model  = model.to(device).eval()

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics   = []
    all_images    = []   
    all_gt_masks  = []
    all_pred_masks= []

    idx = 0
    for images, masks in test_loader:
        images_gpu = images.to(device, non_blocking=True)
        logits     = model(images_gpu)
        probs      = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
        masks_np   = masks.numpy()                         # (B,1,H,W)
        images_np  = images.numpy()                        # (B,9,H,W)

        for b in range(images.size(0)):
            pred_bin = (probs[b, 0] > threshold).astype(np.uint8)
            gt_bin   = (masks_np[b, 0] > 0.5).astype(np.uint8)

            m = _compute_metrics(pred_bin, gt_bin)
            m["sample"] = (sample_names[idx] if sample_names
                           else f"sample_{idx:04d}")
            all_metrics.append(m)

            # Store first 3 RGB channels (original image) for visualisation
            rgb = images_np[b, :3].transpose(1, 2, 0)   # (H,W,3)
            rgb = np.clip(rgb, 0, 1)
            all_images.append(rgb)
            all_gt_masks.append(gt_bin)
            all_pred_masks.append(pred_bin)
            idx += 1

    csv_path = out_dir / "metrics_per_sample.csv"
    fieldnames = ["sample", "dice", "iou", "recall", "precision", "f2", "accuracy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: (f"{m[k]:.6f}" if k != "sample" else m[k])
                             for k in fieldnames})

    metric_keys = ["dice", "iou", "recall", "precision", "f2", "accuracy"]
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        summary[k] = {
            "mean": round(float(np.mean(vals)),  4),
            "std":  round(float(np.std(vals)),   4),
            "min":  round(float(np.min(vals)),   4),
            "max":  round(float(np.max(vals)),   4),
        }

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTest Set Metrics")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for k in metric_keys:
        s = summary[k]
        print(f"{k:<12} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['min']:>8.4f} {s['max']:>8.4f}")
    print(f"\nResults saved to: {out_dir}")

    _save_prediction_grid(all_images, all_gt_masks, all_pred_masks,
                          all_metrics, out_dir, n_samples=12)

    return summary



def _save_prediction_grid(images, gt_masks, pred_masks, metrics,
                          out_dir: Path, n_samples: int = 12):
    n     = min(n_samples, len(images))
    cols  = 3
    fig   = plt.figure(figsize=(cols * 3.5, n * 3.2))
    gs    = gridspec.GridSpec(n, cols, figure=fig,
                              hspace=0.35, wspace=0.05)

    col_titles = ["Image", "Ground Truth", "Prediction"]
    for ci, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    for i in range(n):
        dice = metrics[i]["dice"]
        iou  = metrics[i]["iou"]

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(images[i])
        ax0.set_ylabel(f"#{i+1}", fontsize=8, rotation=0,
                       labelpad=18, va="center")
        ax0.set_xticks([]); ax0.set_yticks([])

        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(gt_masks[i], cmap="gray", vmin=0, vmax=1)
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(pred_masks[i], cmap="gray", vmin=0, vmax=1)
        ax2.set_title(f"Dice={dice:.3f} IoU={iou:.3f}",
                      fontsize=7.5, pad=3, color="#444")
        ax2.set_xticks([]); ax2.set_yticks([])

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    path = plots_dir / "predictions_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction grid saved → {path}")


def plot_training_curves(history: dict, results_dir: str = "results"):
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    best_e = history.get("best_epoch", len(epochs))

    # Loss curve 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            color="#4A7DBF", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",
            color="#E07B54", linewidth=1.5, linestyle="--")
    ax.axvline(best_e, color="#666", linewidth=0.8, linestyle=":",
               label=f"Best epoch ({best_e})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (Dice + BCE)")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend(framealpha=0.8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Dice curve 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["train_dice"], label="Train Dice",
            color="#4A7DBF", linewidth=1.5)
    ax.plot(epochs, history["val_dice"],   label="Val Dice",
            color="#E07B54", linewidth=1.5, linestyle="--")
    ax.axvline(best_e, color="#666", linewidth=0.8, linestyle=":",
               label=f"Best epoch ({best_e})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Coefficient")
    ax.set_title("Training & Validation Dice", fontweight="bold")
    ax.set_ylim(0, 1); ax.legend(framealpha=0.8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_curve.png", dpi=150)
    plt.close(fig)

    print(f"Training curves saved → {plots_dir}")


# CLI entry point 

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate U-Net Mini on Kvasir-SEG test set")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to best_model.pth")
    p.add_argument("--history",     default="checkpoints/history.json",
                   help="Training history JSON (for curve plots)")
    p.add_argument("--data_dir",    default="data/Kvasir-SEG")
    p.add_argument("--split_file",  default="data/split.json")
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--device",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from dataloader import get_dataloaders
    from model import UNetMini

    _, _, test_loader = get_dataloaders(
        data_dir   = args.data_dir,
        split_file = args.split_file,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )

    model = UNetMini(in_channels=9, out_channels=1)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, "
          f"val_dice={ckpt['val_dice']:.4f}")

    evaluate(
        model, test_loader,
        device      = args.device,
        threshold   = args.threshold,
        results_dir = args.results_dir,
    )

    history_path = Path(args.history)
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, results_dir=args.results_dir)
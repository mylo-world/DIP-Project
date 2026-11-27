import argparse
import json
import os
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(
        description="Kvasir-SEG U-Net Mini pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "mode",
        choices=["prepare", "train", "evaluate", "all"],
        help="Pipeline stage to run",
    )

    # Data
    p.add_argument("--data_dir",       default="Kvasir-SEG",
                   help="Root directory of Kvasir-SEG dataset")
    p.add_argument("--split_file",     default="split_index.json",
                   help="JSON file for train/val/test splits")
    p.add_argument("--train_ratio",    type=float, default=0.80)
    p.add_argument("--val_ratio",      type=float, default=0.10)
    p.add_argument("--seed",           type=int,   default=42)

    # Training
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--device",         default=None,
                   help="cuda | mps | cpu (auto-detected if omitted)")

    # Evaluation
    p.add_argument("--checkpoint",     default="checkpoints/best_model.pth",
                   help="Checkpoint to evaluate (evaluate / all modes)")
    p.add_argument("--threshold",      type=float, default=0.5,
                   help="Sigmoid threshold for binarisation")
    p.add_argument("--results_dir",    default="results")

    return p



def stage_prepare(args):
    print("\n STAGE: PREPARE ")
    import dataset as ds

    if hasattr(ds, "build_split"):
        result = ds.build_split(
            data_dir    = args.data_dir,
            split_file  = args.split_file,
            train_ratio = args.train_ratio,
            val_ratio   = args.val_ratio,
            seed        = args.seed,
        )
        split = {"train": result["train"], "val": result["val"], "test": result["test"]}
    else:
        ds.BASE_DIR    = args.data_dir
        ds.IMAGE_DIR   = os.path.join(args.data_dir, "images")
        ds.MASK_DIR    = os.path.join(args.data_dir, "masks")
        ds.BBOX_JSON   = os.path.join(args.data_dir, "kavsir_bboxes.json")
        ds.OUTPUT_JSON = args.split_file
        result = ds.main()
        split  = {
            "train": result["train_ids"],
            "val":   result["val_ids"],
            "test":  result["test_ids"],
        }

    print(f"Split saved -> {args.split_file}")
    print(f"  Train: {len(split['train'])}  Val: {len(split['val'])}  "
          f"Test: {len(split['test'])}  "
          f"Total: {len(split['train']) + len(split['val']) + len(split['test'])}")


def stage_train(args):
    print("\n STAGE: TRAIN ")
    import torch
    from dataloader import get_dataloaders
    from model import UNetMini
    from train import train

    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = args.data_dir,
        split_file  = args.split_file,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    model = UNetMini(in_channels=9, out_channels=1)

    try:
        from torchinfo import summary
        summary(model, input_size=(1, 9, 256, 256), depth=3,
                col_names=["input_size", "output_size", "num_params"],
                verbose=1)
    except ImportError:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"UNetMini | trainable params: {n_params:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resuming from epoch {ckpt['epoch']} "
              f"(val_dice={ckpt['val_dice']:.4f})")

    history = train(
        model, train_loader, val_loader,
        epochs         = args.epochs,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        patience       = args.patience,
        checkpoint_dir = args.checkpoint_dir,
        device         = args.device,
    )
    return history


def stage_evaluate(args, history=None):
    print("\n STAGE: EVALUATE ")
    import torch
    from dataloader import get_dataloaders
    from model import UNetMini
    from evaluate import evaluate, plot_training_curves

    _, _, test_loader = get_dataloaders(
        data_dir    = args.data_dir,
        split_file  = args.split_file,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    model = UNetMini(in_channels=9, out_channels=1)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded  -> epoch {ckpt['epoch']}  val_dice={ckpt['val_dice']:.4f}")

    summary = evaluate(
        model, test_loader,
        device      = args.device,
        threshold   = args.threshold,
        results_dir = args.results_dir,
    )

    # Plot training curves
    if history is None:
        hist_path = Path(args.checkpoint_dir) / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)

    if history is not None:
        plot_training_curves(history, results_dir=args.results_dir)

    return summary



def main():
    parser  = build_parser()
    args    = parser.parse_args()
    history = None

    if args.mode == "prepare":
        stage_prepare(args)
    elif args.mode == "train":
        history = stage_train(args)
    elif args.mode == "evaluate":
        stage_evaluate(args)
    elif args.mode == "all":
        stage_prepare(args)
        history = stage_train(args)
        stage_evaluate(args, history=history)

    print("\nDone.")


if __name__ == "__main__":
    main()
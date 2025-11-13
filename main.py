"""
This module performs K-fold cross-validation training, validation, and testing
of an image segmentation model using PyTorch. It manages data loading,
training loops, model evaluation, and logging of per-fold metrics.

The training pipeline includes:
    - Reproducible seeding
    - K-fold dataset splitting
    - Model training and validation per epoch
    - Metrics logging and saving for each fold
    - Aggregation of fold-level test results
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from losses import DiceLoss, ComboLoss
from dataset import CSVDataset
from model import UNet, ResUNet1, ResUNet2, ResUNet3
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any


#model outputs 2 channel(s).
#mask has 1 channel(s).
def run_training(args: Any) -> None:
    """
    Execute K-fold cross-validation training and evaluation.

    Args:
        args: Object containing configuration parameters such as dataset directories and image settings.        
    Returns:
        None. Results and logs are written to disk.
    """
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    """


    TO DO


    """
    # Load dataset
    csv_path = os.path.join(args.dataset_dir, args.csv_file)
    df = pd.read_csv(csv_path)
    run_root = os.path.join(args.output_dir, args.version)
    os.makedirs(os.path.join(run_root, "logs"), exist_ok=True)

    # KFold split
    fold_results = []

    kf = KFold(n_splits=int(args.num_folds), shuffle=True, random_state=args.seed)
    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(df)):
        fold = fold_idx + 1
        print(f"\n===== Fold {fold}/{args.num_folds} =====")

        # Per-fold dataframes
        test_df = df.iloc[test_idx].reset_index(drop=True)
        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)

        # Non-stratified validation split from train portion
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.2,
            shuffle=False,
            random_state=args.seed,
            stratify=None,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # Initialize datasets and dataloaders
        train_ds = CSVDataset(args, train_df)
        val_ds   = CSVDataset(args, val_df)
        test_ds  = CSVDataset(args, test_df)

        train_dl = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True,  num_workers=1)
        val_dl   = DataLoader(val_ds,   batch_size=int(args.batch), shuffle=False, num_workers=1)
        test_dl  = DataLoader(test_ds,  batch_size=int(args.batch), shuffle=False, num_workers=1)

        # Initialize model, loss, and optimizer
        # model = UNet(num_classes=int(args.num_classes)).to(DEVICE)
        # model = ResUNet1(num_classes=int(args.num_classes)).to(DEVICE)
        model = ResUNet2(num_classes=int(args.num_classes)).to(DEVICE)
        # model = ResUNet3(num_classes=int(args.num_classes)).to(DEVICE)
        # criterion = DiceLoss()
        criterion = ComboLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(getattr(args, "weight_decay", 1e-4)))

        log_records = []

        # Training loop (no checkpoint saving; last-epoch weights are used for testing)
        for epoch in range(1, int(args.epochs) + 1):
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_dice, val_iou, val_fpr, val_fnr = validate(model, val_dl, criterion, DEVICE)

            log_records.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_dice": float(val_dice),
                "val_iou": float(val_iou),
                "val_fpr": float(val_fpr),
                "val_fnr": float(val_fnr),
            })

            print(
                f"Epoch [{epoch:02d}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | FPR: {val_fpr:.2f}% | FNR: {val_fnr:.2f}%"
            )

        # Save per-fold logs
        log_df = pd.DataFrame(log_records)
        log_path = os.path.join(str(args.output_dir), str(args.version), "logs", f"fold_{fold}_log.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_df.to_csv(log_path, index=False)

        # Final test evaluation using last-epoch model (no checkpoint load)
        preds_dir = os.path.join(str(args.output_dir), str(args.version), f"FOLD_{fold}", "predictions")
        acc, dice, iou, fpr, fnr = test(model, test_dl, DEVICE, save_image_dir=preds_dir)
        fold_results.append({
            "fold": fold,
            "test_acc": float(acc),
            "test_dice": float(dice),
            "test_iou": float(iou),
            "test_fpr": float(fpr),
            "test_fnr": float(fnr),
        })

        # Free memory and delete model to prevent leakage over folds
        del model
        torch.cuda.empty_cache()

    # Save aggregated test results
    results_path = os.path.join(run_root, "testing_results.csv")
    pd.DataFrame(fold_results).to_csv(results_path, index=False)


if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)

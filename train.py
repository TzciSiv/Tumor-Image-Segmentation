"""
This module provides utility functions for training, validating, and testing
a PyTorch segmentation model. It includes per-epoch training, validation with
metric computation, and final model evaluation.

Functions:
    train_one_epoch: Train the model for a single epoch.
    validate: Evaluate model performance on the validation set.
    test: Evaluate model performance on the test set and save the prediction along with the input for comparison.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from metrics import calculate_metrics, accuracy_score
from typing import Tuple
from torchvision.utils import save_image
import os

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch on the provided training set.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (nn.Module): Loss function used for optimization.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Average training loss and Accuracy for the epoch.
    """

    
    
    """


    TO DO


    """
    model.train()
    total_train_loss = 0.0
    all_pred, all_true  = [], []
    thr = 0.5

    for imgs, masks, _ in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * imgs.size(0)

        probs = torch.softmax(outputs, dim=1)         # (B,2,H,W)
        lesion_prob = probs[:, 1:2, ...]              # (B,1,H,W)
        preds = (lesion_prob >= thr).to(torch.uint8)  # (B,1,H,W) 0/1
        all_pred.append(preds)
        all_true.append((masks > 0).to(torch.uint8))

    avg_loss = total_train_loss / len(dataloader.dataset)
    acc = accuracy_score(all_true, all_pred)
    
    return avg_loss, acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """
    Validate the model on the provided validation set and compute detailed metrics.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        criterion (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float, float, float, float]:
            Average validation loss, Acc uracy, DICE, IoU, FPR and FNR.
    """

    
    
    """


    TO DO


    """
    model.eval()
    total_train_loss = 0.0
    all_pred, all_true  = [], []
    thr = 0.5

    with torch.no_grad():
        for imgs, masks, _ids in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            total_train_loss +=  loss.item() * imgs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1:2] > thr).float()
            all_pred.append(preds)
            all_true.append((masks > 0.5).float())

    avg_loss = total_train_loss / len(dataloader.dataset)
    yt = torch.cat(all_true, dim=0)
    yp = torch.cat(all_pred, dim=0)
    acc, dice, iou, fpr, fnr = calculate_metrics(yt, yp)    
    
    return avg_loss, acc, dice, iou, fpr, fnr

def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_image_dir: str
) -> Tuple[float, float, float, float, float]:
    """
    Test the model on the provided testing set and compute detailed metrics.
    Save the models' prediction with the input and ground truth.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        device (torch.device): Device to perform computations on.
        save_image_dir (str): directory to save the image, mask and predicted mask
                              outputs/model version/image_files/FOLD_n/image_id/[image/mask/pred]
    Returns:
        Tuple[float, float, float, float, float]:
            Accuracy, DICE, IoU, FPR and FNR.
    """

    
    
    """


    TO DO


    """
    model.eval()
    os.makedirs(save_image_dir, exist_ok=True)
    all_pred, all_true  = [], []
    thr = 0.5

    with torch.no_grad():
        for imgs, masks, ids in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
    
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1:2] > thr).float()

            all_true.append((masks > 0.5).float().view(-1))
            all_pred.append(preds.view(-1))

            # save imgs per-sample
            for b, img_id in enumerate(ids):
                out_dir = os.path.join(save_image_dir, str(img_id))
                os.makedirs(out_dir, exist_ok=True)

                img_to_save = imgs[b].detach().cpu()
                msk_to_save = masks[b].detach().cpu()
                prd_to_save = preds[b].detach().cpu()

                save_image(img_to_save, os.path.join(out_dir, "image.png"))
                save_image(msk_to_save, os.path.join(out_dir, "mask.png"))
                save_image(prd_to_save, os.path.join(out_dir, "pred.png"))

    yt = torch.cat(all_true, dim=0)
    yp = torch.cat(all_pred, dim=0)
    acc, dice, iou, fpr, fnr = calculate_metrics(yt, yp)

    return acc, dice, iou, fpr, fnr
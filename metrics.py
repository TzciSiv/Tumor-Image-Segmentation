"""
    This module calculates relevant metrics from ground truth and predicted labels.
"""
from typing import Sequence, Tuple
import torch
import torch.nn as nn
import numpy as np

def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[float, float, float, float, float]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - Accuracy
      - DICE
      - IoU
      - FPR
      - FNR

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (accuracy, dice, iou, fpr, fnr)
    """

    
    
    """


    TO DO


    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn

    acc  = (tp + tn) / (total)
    dice = (2 * tp) / (2 * tp + fp + fn)
    iou  = tp / (tp + fp + fn)
    fpr  = (fp / (fp + tn)) * 100.0
    fnr  = (fn / (fn + tp)) * 100.0
    
    return acc, dice, iou, fpr, fnr


def confusion_matrix(    
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - TN
      - FP
      - FN
      - TP

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (tn, fp, fn, tp)
    """

    
    """


    TO DO


    """    
    # shape (B,1,H,W)
    # Normalize to lists so torch.cat always works
    yt_list = y_true if isinstance(y_true, (list, tuple)) else [y_true]
    yp_list = y_pred if isinstance(y_pred, (list, tuple)) else [y_pred]

    # Concatenate, flatten, and binarize
    yt = torch.cat(yt_list, dim=0).reshape(-1).float()
    yp = torch.cat(yp_list, dim=0).reshape(-1).float()

    yt = yt > 0.5     # bool
    yp = yp > 0.5     # bool

    tp = (yp & yt).sum().item()
    tn = ((~yp) & (~yt)).sum().item()
    fp = (yp & (~yt)).sum().item()
    fn = ((~yp) & yt).sum().item()
    
    return tn, fp, fn, tp


def accuracy_score(    
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """
    This function computes the accuracy score based on ground truth labels and predicted labels.

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A float value of the accuracy score
    """

    
    """


    TO DO


    """    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total

    return acc
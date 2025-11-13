"""
    This module calculates DICE Loss and Combo Loss from logits and targets.
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]  (raw outputs from model)
        targets: [B, H, W]    (integer class labels)
        """
        
        
        
        """
        
        
        TO DO
        
        
        """
        B, C, H, W = logits.shape

        # targets → [B,1,H,W] and binary {0,1}
        targets = targets.view(targets.size(0), 1, targets.size(-2), targets.size(-1)).float()
        targets = (targets > 0).float()

        # foreground probability p
        probs = torch.softmax(logits, dim=1)[:, 1:2]   # foreground channel → [B,1,H,W]

        # flatten per-sample
        p = probs.view(B, -1)
        y = targets.view(B, -1)

        intersection = (p * y).sum(dim=1)
        p2 = (p * p).sum(dim=1)
        y2 = (y * y).sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (p2 + y2 + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss

class ComboLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss() 

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]  (raw outputs from model)
        targets: [B, H, W]    (integer class labels)
        """
        # Dice part 
        dice_loss = self.dice(logits, targets)

        # CE part: needs integer class ids [B,H,W]
        # Convert mask to {0,1} long
        ce_targets = (targets > 0).long()
        ce_targets = ce_targets[:, 0, ...]  # -> [B,H,W]
        ce_loss = self.ce(logits, ce_targets)

        return  0.5 * dice_loss + 0.5 * ce_loss

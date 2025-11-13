"""
    This module creates a custom PyTorch Dataset for loading prepared and pre-processed image data and corresponding mask from a CSV DataFrame.

    Each row in the DataFrame contains image_id which is the file name for the image and mask files.
    
"""
import os
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSVDataset(Dataset):
    """
    Attributes:
        df (pd.DataFrame): DataFrame containing image file names.
        args: Object containing configuration parameters such as dataset directories and image settings.
        img_transform (transforms.Compose): Composed transformation pipeline applied to each input image.
        mask_transform (transforms.Compose): Composed transformation pipeline applied to each mask.
    """

    def __init__(self, args, df: pd.DataFrame) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image and mask transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
            df (pd.DataFrame): DataFrame containing 'image_id' column.
        """
        
        """
        
        
        TO DO
        
        
        """
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.args = args
        self.image_size: int = int(args.image_size)
        self.use_aug: bool = bool(getattr(args, "augment", False))
        self.img_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.mask_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.PILToTensor()
        ])
        
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieve a single image_id from the dataset and use it to open input image and its mask.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Image tensor of shape (C, H, W) after transformations.
                - Mask tensor of shape (H, W) after transformation, typically 0 or 1 for binary classification.
                - Image ID of type string used to save the prediction output as the same name.
        """
        """
        
        
        TO DO
        
        
        """
        row = self.df.iloc[i]

        # paths
        image_path = os.path.join(
            self.args.dataset_dir,
            self.args.image_dir,
            f"{row['image_id']}.jpg",
        )
        mask_path = os.path.join(
            self.args.dataset_dir,
            self.args.mask_dir,
            f"{row['image_id']}_segmentation.png",
        )
        
        # image preparation
        img = Image.open(image_path).convert("RGB")
        x = self.img_transform(img)  # (C, H, W)

        # mask preparation
        msk = Image.open(mask_path).convert("L")
        y = self.mask_transform(msk)  # (H, W) int64 labels (0/1 for binary)

        # ---- label-preserving augmentation ----
        if torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        # vertical flip
        if torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)

        # random 0/90/180/270 rotation
        k = int(torch.randint(0, 4, (1,)).item())
        angle = 90 * k
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        msk = msk.rotate(angle, resample=Image.NEAREST,  fillcolor=0)
        # ---- end augmentation ----

        if y.ndim == 2:
            y = y.unsqueeze(0)
        elif y.ndim == 3 and y.size(0) != 1:
            y = y[:1, ...]
        # binarize to {0.,1.} and cast to float
        y = (y > 0).to(torch.float32)        # [1,H,W] float

        return x, y, str(row["image_id"])
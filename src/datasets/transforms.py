from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _normalize(mean, std):
    return A.Normalize(mean=mean, std=std, max_pixel_value=255.0)


def get_train_transforms(image_size: int, mean, std) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            _normalize(mean, std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int, mean, std) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            _normalize(mean, std),
            ToTensorV2(),
        ]
    )


def get_test_transforms(image_size: int, mean, std) -> A.Compose:
    # For now, val and test transforms are identical
    return get_val_transforms(image_size, mean, std)




import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF

CLASS_NAMES = {
    0: "Ограничение скорости (20 км/ч)",
    1: "Ограничение скорости (30 км/ч)",
    2: "Ограничение скорости (50 км/ч)",
    3: "Ограничение скорости (60 км/ч)",
    4: "Ограничение скорости (70 км/ч)",
    5: "Ограничение скорости (80 км/ч)",
    6: "Конец ограничения скорости (80 км/ч)",
    7: "Ограничение скорости (100 км/ч)",
    8: "Ограничение скорости (120 км/ч)",
    9: "Обгон запрещён",
    10: "Обгон грузовым автомобилям запрещён",
    11: "Преимущество перед встречным движением",
    12: "Главная дорога",
    13: "Уступите дорогу",
    14: "Движение без остановки запрещено (STOP)",
    15: "Движение запрещено",
    16: "Движение грузовых автомобилей запрещено",
    17: "Въезд запрещён",
    18: "Прочие опасности",
    19: "Опасный поворот налево",
    20: "Опасный поворот направо",
    21: "Извилистая дорога",
    22: "Неровная дорога",
    23: "Скользкая дорога",
    24: "Сужение дороги справа",
    25: "Дорожные работы",
    26: "Светофорное регулирование",
    27: "Пешеходный переход",
    28: "Дети",
    29: "Пересечение с велосипедной дорожкой",
    30: "Падение камней",
    31: "Дикие животные",
    32: "Конец всех ограничений",
    33: "Поворот направо",
    34: "Поворот налево",
    35: "Движение прямо",
    36: "Движение прямо или направо",
    37: "Движение прямо или налево",
    38: "Держитесь правее",
    39: "Держитесь левее",
    40: "Круговое движение",
    41: "Конец зоны запрещения обгона",
    42: "Конец зоны запрещения обгона грузовым автомобилям",
    43: "Неизвестный знак 43",
    44: "Неизвестный знак 44",
    45: "Неизвестный знак 45",
    46: "Неизвестный знак 46",
    47: "Неизвестный знак 47",
    48: "Неизвестный знак 48",
    49: "Неизвестный знак 49",
    50: "Неизвестный знак 50",
    51: "Неизвестный знак 51",
    52: "Неизвестный знак 52",
    53: "Неизвестный знак 53",
    54: "Неизвестный знак 54",
    55: "Неизвестный знак 55",
    56: "Неизвестный знак 56",
    57: "Неизвестный знак 57",
    58: "Неизвестный знак 58",
    59: "Неизвестный знак 59",
    60: "Неизвестный знак 60",
    61: "Неизвестный знак 61",
    62: "Неизвестный знак 62",
    63: "Неизвестный знак 63",
    64: "Неизвестный знак 64",
    65: "Неизвестный знак 65",
    66: "Неизвестный знак 66",
}

class TrafficSignDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[callable] = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def get_train_transforms(img_size: int = 48) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(img_size: int = 48) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_tta_transforms(img_size: int = 48) -> List[transforms.Compose]:
    tta_transforms = []

    tta_transforms.append(transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    tta_transforms.append(transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    for angle in [90, 180, 270]:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img, a=angle: TF.rotate(img, a)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    return tta_transforms

def create_stratified_split(
    dataset: TrafficSignDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    random.seed(seed)
    np.random.seed(seed)

    labels = dataset.df["class_number"].values
    classes = np.unique(labels)

    train_indices = []
    val_indices = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0].tolist()
        np.random.shuffle(cls_indices)

        n_val = max(1, int(len(cls_indices) * val_ratio))

        if len(cls_indices) <= 1:
            n_val = 0

        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"Train classes: {len(np.unique(labels[train_indices]))}, "
          f"Val classes: {len(np.unique(labels[val_indices]))}")

    return train_dataset, val_dataset

def get_data_loaders(
    batch_size: int = 64,
    img_size: int = 48,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, TrafficSignDataset]:
    train_dataset = TrafficSignDataset(
        csv_file="train.csv",
        img_dir="train/train",
        transform=get_train_transforms(img_size),
    )

    full_val_dataset = TrafficSignDataset(
        csv_file="train.csv",
        img_dir="train/train",
        transform=get_val_transforms(img_size),
    )

    train_subset, val_subset = create_stratified_split(
        train_dataset, val_ratio=val_ratio, seed=seed
    )

    val_indices = val_subset.indices
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset

def get_test_loader(
    batch_size: int = 64,
    img_size: int = 48,
    num_workers: int = 0,
) -> DataLoader:
    test_dataset = TrafficSignDataset(
        csv_file="sample_submission.csv",
        img_dir="test/test",
        transform=get_val_transforms(img_size),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader

def get_class_weights(dataset: TrafficSignDataset) -> torch.Tensor:
    labels = dataset.df["class_number"].values
    classes = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(classes)

    weights = np.zeros(n_classes)
    for cls in classes:
        n_cls = np.sum(labels == cls)
        weights[cls] = n_samples / (n_classes * n_cls)

    return torch.tensor(weights, dtype=torch.float32)

def apply_degradation(
    image: torch.Tensor,
    degradation_type: str = "blur",
    severity: float = 0.5,
) -> torch.Tensor:
    img = image.clone()

    if degradation_type == "blur":
        kernel_size = max(3, int(severity * 15) // 2 * 2 + 1)
        sigma_val = max(0.1, severity * 3)
        blur = transforms.GaussianBlur(kernel_size, sigma=sigma_val)
        img = blur(img)

    elif degradation_type == "noise":
        noise = torch.randn_like(img) * severity * 0.5
        img = torch.clamp(img + noise, 0, 1)

    elif degradation_type == "darken":
        img = img * (1.0 - severity * 0.7)
        img = torch.clamp(img, 0, 1)

    elif degradation_type == "small":
        scale = 1.0 - severity * 0.7
        h, w = img.shape[1:]
        new_h, new_w = max(8, int(h * scale)), max(8, int(w * scale))
        resize_down = transforms.Resize((new_h, new_w))
        resize_up = transforms.Resize((h, w))
        img = resize_up(resize_down(img))

    return img

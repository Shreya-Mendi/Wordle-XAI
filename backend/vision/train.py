"""
Training script for the Wordle tile state classifier.

Generates synthetic tile images for all 4 classes (correct/present/absent/empty)
and fine-tunes the ResNet18 backbone + custom classification head.

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 64 --output checkpoints/best.pth
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from model import WordleTileClassifier, STATE_CLASSES

# Wordle tile colors (RGB)
TILE_COLORS = {
    "correct": (83, 141, 78),    # #538d4e
    "present": (181, 159, 59),   # #b59f3b
    "absent":  (58, 58, 60),     # #3a3a3c
    "empty":   (18, 18, 19),     # #121213 (dark background)
}

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _get_font(size: int = 36):
    # Try system font paths in priority order (macOS, then Linux)
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    # Try matplotlib font manager as a portable fallback
    try:
        import matplotlib.font_manager as fm
        font_path = fm.findfont(fm.FontProperties(family="sans-serif", weight="bold"))
        if font_path:
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def make_tile(state: str, letter: str = "", size: int = 64, jitter: float = 0.15) -> Image.Image:
    """
    Generate a synthetic Wordle tile image.

    Args:
        state: one of STATE_CLASSES
        letter: uppercase letter to render (empty string for 'empty' tiles)
        size: tile size in pixels
        jitter: colour noise fraction (0 = no noise)
    """
    base_color = TILE_COLORS[state]

    # Apply colour jitter
    r, g, b = base_color
    r = int(np.clip(r + random.uniform(-jitter, jitter) * 255, 0, 255))
    g = int(np.clip(g + random.uniform(-jitter, jitter) * 255, 0, 255))
    b = int(np.clip(b + random.uniform(-jitter, jitter) * 255, 0, 255))

    img = Image.new("RGB", (size, size), (r, g, b))
    draw = ImageDraw.Draw(img)

    # Tile border (slightly lighter)
    border_c = tuple(min(255, c + 30) for c in (r, g, b))
    draw.rectangle([(0, 0), (size - 1, size - 1)], outline=border_c, width=2)

    if letter and state != "empty":
        font = _get_font(int(size * 0.58))
        text_color = (255, 255, 255)
        bbox = draw.textbbox((0, 0), letter, font=font)
        tx = (size - (bbox[2] - bbox[0])) // 2 - bbox[0]
        ty = (size - (bbox[3] - bbox[1])) // 2 - bbox[1]
        # Slight position jitter
        tx += random.randint(-2, 2)
        ty += random.randint(-2, 2)
        draw.text((tx, ty), letter, fill=text_color, font=font)

    return img


class SyntheticTileDataset(Dataset):
    def __init__(self, n_per_class: int = 2000, transform=None):
        self.samples: list[tuple[Image.Image, int]] = []
        self.transform = transform

        for class_idx, state in enumerate(STATE_CLASSES):
            for _ in range(n_per_class):
                letter = random.choice(LETTERS) if state != "empty" else ""
                tile = make_tile(state, letter)
                self.samples.append((tile, class_idx))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = str(_SCRIPT_DIR / "checkpoints" / "best_model.pth")


def train(
    epochs: int = 20,
    batch_size: int = 64,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-5,
    output_path: str = _DEFAULT_CHECKPOINT,
    n_per_class: int = 2000,
    val_split: float = 0.1,
):
    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    train_transform = T.Compose([
        T.Resize((64, 64)),
        T.RandomRotation(5),
        T.RandomHorizontalFlip(p=0.0),  # letters shouldn't flip
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        T.RandomCrop(64, padding=4),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = SyntheticTileDataset(n_per_class=n_per_class)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    # Apply transforms after split
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return self.transform(img), label

    train_loader = DataLoader(
        TransformDataset(train_set, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TransformDataset(val_set, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = WordleTileClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_acc * 100:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "val_acc": val_acc},
                output_path,
            )
            print(f"  ✓ Saved best model (val_acc={val_acc * 100:.1f}%) → {output_path}")

    print(f"\n[Train] Done. Best val accuracy: {best_val_acc * 100:.1f}%")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wordle tile classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-per-class", type=int, default=2000)
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pth")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_per_class=args.n_per_class,
        output_path=args.output,
    )

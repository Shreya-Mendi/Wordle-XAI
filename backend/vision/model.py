"""
ResNet18-based tile classifier for Wordle XAI.
Classifies each tile into one of 4 states: correct, present, absent, empty.
Uses transfer learning from ImageNet pretrained weights.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# State class mapping
STATE_CLASSES = ["correct", "present", "absent", "empty"]
NUM_CLASSES = len(STATE_CLASSES)

# Input transform matching ImageNet preprocessing
TILE_TRANSFORM = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WordleTileClassifier(nn.Module):
    """
    ResNet18 with a custom classification head for Wordle tile state detection.
    Target layer for Grad-CAM: self.backbone.layer4[-1]
    Feature map resolution at layer4 output: 2×2 for 64×64 input.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = models.resnet18(weights=weights)

        # Remove the original fully-connected head
        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Custom classification head for tile state (4 classes)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    @property
    def target_layer(self) -> nn.Module:
        """The last residual block — used as Grad-CAM hook target."""
        return self.backbone.layer4[-1]


class WordleVisionModel:
    """
    High-level wrapper around WordleTileClassifier.
    Handles loading, inference, and preprocessing.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WordleTileClassifier(pretrained=True).to(self.device)
        self.loaded_checkpoint = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.model.eval()

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.loaded_checkpoint = True
        print(f"[Vision] Loaded checkpoint: {path}")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert a PIL tile image to a model-ready tensor."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        tensor = TILE_TRANSFORM(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> tuple[str, float, np.ndarray]:
        """
        Predict the state of a single tile.

        Returns:
            (state_label, confidence, probabilities_array)
        """
        tensor = self.preprocess(pil_image)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        class_idx = int(np.argmax(probs))
        return STATE_CLASSES[class_idx], float(probs[class_idx]), probs

    def predict_with_grad(self, pil_image: Image.Image) -> tuple[str, float, np.ndarray, torch.Tensor, torch.Tensor, int]:
        """
        Predict tile state while retaining gradients (for Grad-CAM).

        Returns:
            (state_label, confidence, probabilities, input_tensor, logits, class_idx)
            input_tensor has requires_grad=True; logits retains the computation graph
            for backpropagation in Grad-CAM.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        tensor = TILE_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        class_idx = int(np.argmax(probs))

        return STATE_CLASSES[class_idx], float(probs[class_idx]), probs, tensor, logits, class_idx

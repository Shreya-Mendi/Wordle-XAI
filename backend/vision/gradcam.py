"""
Grad-CAM implementation for Wordle tile explainability.

Registers forward/backward hooks on ResNet18 layer4 to capture
activations and gradients, then computes the weighted activation map.
Produces per-tile heatmap overlays showing which image regions the model
attends to when classifying tile state (green/yellow/gray/empty).
"""

import io
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class TileGradCAM:
    """
    Grad-CAM for Wordle tile state classification.

    Usage:
        gcam = TileGradCAM(vision_model.model)
        heatmap_pil, overlay_pil = gcam.explain(tile_pil_image, target_class=None)
    """

    def __init__(self, model):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        # Always clear old handles before re-registering to prevent duplicate hooks
        self.remove_hooks()
        target = self.model.target_layer

        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles.append(target.register_forward_hook(fwd_hook))
        self._handles.append(target.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __del__(self) -> None:
        self.remove_hooks()

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        logits: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Run backprop and compute the Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed tile tensor [1, 3, H, W]
            logits: model output (from same forward pass)
            target_class: class index to explain

        Returns:
            cam: float32 numpy array in [0,1] with same spatial ratio as layer4 output,
                 upsampled to 64×64.
        """
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # activations: [1, C, h, w],  gradients: [1, C, h, w]
        grads = self._gradients   # e.g., [1, 512, 2, 2] for 64×64 input
        acts = self._activations  # same shape

        if grads is None or acts is None:
            # Fallback: uniform cam if hooks didn't fire
            return np.ones((64, 64), dtype=np.float32) * 0.5

        # Global average pool gradients over spatial dims → weights [1, C, 1, 1]
        weights = grads.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [1, h, w]
        cam = F.relu(cam).squeeze()                        # [h, w]

        cam_np = cam.cpu().numpy().astype(np.float32)

        # Upsample to 64×64
        if cam_np.ndim == 0:
            cam_np = np.ones((1, 1), dtype=np.float32) * float(cam_np)
        cam_resized = cv2.resize(cam_np, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        vmin, vmax = cam_resized.min(), cam_resized.max()
        if vmax - vmin > 1e-8:
            cam_resized = (cam_resized - vmin) / (vmax - vmin)
        else:
            cam_resized = np.zeros_like(cam_resized)

        return cam_resized

    def explain(
        self,
        pil_image: Image.Image,
        target_class: int | None = None,
    ) -> tuple[Image.Image, Image.Image, int, float]:
        """
        Full Grad-CAM pipeline for a single tile.

        Args:
            pil_image: RGB PIL image of the tile (any size, will be resized)
            target_class: class index to explain; if None, uses model prediction

        Returns:
            (heatmap_pil, overlay_pil, predicted_class_idx, confidence)
        """
        from .model import TILE_TRANSFORM, STATE_CLASSES

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        tile_64 = pil_image.resize((64, 64), Image.LANCZOS)

        # Prepare tensor with grad tracking
        tensor = TILE_TRANSFORM(tile_64).unsqueeze(0)
        tensor = tensor.to(next(self.model.parameters()).device)
        tensor.requires_grad_(True)

        # Enable gradients for this pass even in eval mode
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if target_class is None:
            target_class = pred_idx

        cam = self.compute_cam(tensor, logits, target_class)

        heatmap_pil = self._cam_to_heatmap_image(cam)
        overlay_pil = self._blend_overlay(tile_64, cam, alpha=0.5)

        return heatmap_pil, overlay_pil, pred_idx, confidence

    @staticmethod
    def _cam_to_heatmap_image(cam: np.ndarray) -> Image.Image:
        """Convert a [0,1] float CAM array to a JET colormap PIL image."""
        cam_uint8 = np.uint8(255 * cam)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb)

    @staticmethod
    def _blend_overlay(
        tile_pil: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Blend the JET heatmap over the original tile image.

        Args:
            tile_pil: original tile at 64×64
            cam: normalized [0,1] CAM array at 64×64
            alpha: heatmap opacity (0=tile only, 1=heatmap only)

        Returns:
            blended PIL image
        """
        tile_np = np.array(tile_pil.resize((64, 64))).astype(np.float32)

        cam_uint8 = np.uint8(255 * cam)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        blended = (alpha * heatmap_rgb + (1 - alpha) * tile_np).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    @staticmethod
    def pil_to_b64(pil_image: Image.Image) -> str:
        """Encode a PIL image to a base64 PNG string."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

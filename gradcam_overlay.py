"""
Grad-CAM++ Visualisation for YOLO spill detections.
Generates heatmap overlays highlighting regions the model attends to.
Uses pytorch-grad-cam with multi-layer fallback (thesis Section 4.7).
"""

import cv2
import numpy as np
from typing import Optional


def try_gradcam(model_obj, frame_rgb: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """
    Attempts to generate a Grad-CAM++ heatmap for the given frame and bbox.
    Returns BGR overlay image, or None if pytorch-grad-cam is unavailable.

    Args:
        model_obj: Ultralytics YOLO model
        frame_rgb: RGB numpy array (H, W, 3)
        bbox: (x1, y1, x2, y2) detection box to highlight
    """
    try:
        import torch
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None

    try:
        # Access YOLO backbone — try common layer paths
        backbone = model_obj.model.model
        target_layers = []

        # Try the last few Conv/C2f layers in the backbone
        for name, module in backbone.named_modules():
            if hasattr(module, 'conv') or module.__class__.__name__ in ('C2f', 'Conv'):
                target_layers = [module]

        if not target_layers:
            return None

        # Preprocess
        h, w = frame_rgb.shape[:2]
        img_float = frame_rgb.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_float.transpose(2, 0, 1)).unsqueeze(0)

        def reshape_transform(tensor):
            # Handle YOLO feature map shapes
            if len(tensor.shape) == 4:
                return tensor
            return tensor

        with GradCAMPlusPlus(
            model=backbone,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        ) as cam:
            grayscale_cam = cam(input_tensor=img_tensor)[0]

        # Crop heatmap to detection region, then resize back
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(x2, w), min(y2, h)

        heatmap = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        # Draw bounding box on heatmap
        cv2.rectangle(heatmap_bgr, (x1, y1), (x2, y2), (0, 255, 100), 2)
        return heatmap_bgr

    except Exception as e:
        print(f"[GradCAM] Failed: {e}")
        return None


def draw_detection_overlay(
    frame: np.ndarray,
    bbox: tuple,
    confidence: float,
    residual: float,
    status: str,           # "CONFIRMED", "PENDING", "REJECTED"
    alert_id: str = "",
    probability: float = None,   # P(physically plausible), if a threshold calibrator is fitted
    uncertainty: float = None,   # MC-Dropout residual std
    fluid_class: str = None,     # detected/assumed fluid class (from PINNResult.detected_fluid_class)
) -> np.ndarray:
    """
    Draws detection box + status labels on frame (BGR).
    Status colours:
        CONFIRMED → red
        PENDING   → orange
        REJECTED  → grey
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    colour_map = {
        "CONFIRMED": (0, 0, 220),
        "PENDING":   (0, 140, 255),
        "REJECTED":  (120, 120, 120),
        "PASSED":    (200, 160, 0),
    }
    colour = colour_map.get(status, (200, 200, 200))

    # Box
    thickness = 3 if status == "CONFIRMED" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    # Label background
    label = f"{status}  conf:{confidence:.2f}  res:{residual:.4f}"
    if uncertainty is not None:
        label += f"+/-{uncertainty:.4f}"
    if probability is not None:
        label += f"  P:{probability:.2f}"
    if fluid_class:
        label += f"  [{fluid_class}]"
    if alert_id:
        label += f"  [{alert_id}]"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
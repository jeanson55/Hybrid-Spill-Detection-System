"""
Fluid Identification Classifier
────────────────────────────────
A standalone model that predicts *what kind of fluid* a confirmed
spill-detection region contains — decoupled from the YOLO detector and
the PINN residual validator.

Why separate:
    YOLO answers "is there a spill-like region here?"
    PINN answers  "does this region behave like a fluid physically?"
    This module answers "which fluid?" — and its output (fluid_class)
    feeds back into the PINN validator as the viscosity lookup key
    (see PINNValidator.VISCOSITY_MAP), so a classifier error here
    degrades physics-plausibility checking, not detection sensitivity.
    That's a deliberately soft coupling: worth getting right, but not
    safety-critical the way stages 1–2 are.

Feature groups (thesis-aligned):
    Visual        — RGB stats, HSV stats, GLCM texture, LBP texture,
                     specular-reflectance proxy
    Geometric      — area, compactness, elongation, fractal dimension
                     (computed from a segmentation mask of the crop,
                     not just the raw bbox, so shape actually varies
                     between a slick and a compact drip)
    Environmental  — wind speed, current velocity, temperature
                     (NOT derivable from the image — must be supplied
                     by the caller from a sensor feed / metocean API;
                     sane defaults are used if omitted)

Backends (pick one via `method=`):
    "random_forest"  — default; always available (scikit-learn)
    "xgboost"         — optional; falls back to random_forest with a
                          warning if xgboost isn't installed
    "lightgbm"         — optional; same fallback behaviour
    "cnn"               — small PyTorch CNN over the cropped detection
                             image, with environmental features fused
                             in at the fully-connected head
    "vit"                — optional Vision Transformer backend (via
                             `timm`); only worth it once you have on the
                             order of a few thousand labelled crops per
                             class — see ViTFluidClassifierWrapper
                             docstring. Falls back to "cnn" otherwise.

All backends share the same outward interface via `FluidClassifier`:
    clf = FluidClassifier(method="random_forest", model_path="fluid_rf.joblib")
    pred = clf.predict(frame, bbox, env=EnvironmentalConditions(wind_speed_mps=4.2))
    pred.label, pred.confidence, pred.probabilities
"""

import warnings
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import cv2


def _win_extended_path(path: Path) -> str:
    """Bypasses Windows' legacy 260-char MAX_PATH without needing the
    registry LongPathsEnabled setting — see import_public_dataset.py for
    the full explanation."""
    s = str(path.resolve())
    if platform.system() == "Windows" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def _safe_imread(path: Path):
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        data = np.fromfile(_win_extended_path(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


# ── Classes & Constants ──────────────────────────────────────────────────────

# Matches PINNValidator.VISCOSITY_MAP keys (minus "unknown", which is a
# fallback the classifier emits when it isn't confident — never a
# training target).
FLUID_CLASSES = ["water", "light_hydrocarbon", "heavy_oil", "drilling_mud"]

# All crops are normalized to this size (aspect-preserving, reflect-padded)
# before feature extraction. Without this, absolute-scale features (area,
# and to a lesser extent GLCM/LBP texture granularity) would leak which
# *source dataset* an image came from rather than genuine fluid-appearance
# signal — different public datasets/exports are captured or resized at
# very different native resolutions, and that difference otherwise
# correlates with class label purely by which dataset happened to supply
# more of which class.
CANONICAL_CROP_SIZE = 128

# Typical calm-conditions defaults used only when environmental readings
# aren't supplied. These are placeholders, not calibrated values — wire
# up a real sensor/metocean feed and pass EnvironmentalConditions in
# explicitly whenever possible.
_DEFAULT_WIND_MPS = 3.0
_DEFAULT_CURRENT_MPS = 0.1
_DEFAULT_TEMP_C = 25.0

# Below this predicted-class probability, FluidClassifier reports
# "unknown" rather than a low-confidence guess (mirrors the PINN's
# VISCOSITY_MAP["unknown"] fallback of μ=10.0).
DEFAULT_MIN_CONFIDENCE = 0.40


@dataclass
class EnvironmentalConditions:
    """
    Environmental readings at detection time. All fields are optional —
    missing values are imputed with calm-conditions defaults so the
    feature vector always has a fixed length, but real readings should
    be supplied whenever available (a wrong wind/current reading will
    bias spreading-rate-sensitive classifications, e.g. light
    hydrocarbon vs. water under wind chop).
    """
    wind_speed_mps: Optional[float] = None
    current_velocity_mps: Optional[float] = None
    temperature_c: Optional[float] = None

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.wind_speed_mps if self.wind_speed_mps is not None else _DEFAULT_WIND_MPS,
            self.current_velocity_mps if self.current_velocity_mps is not None else _DEFAULT_CURRENT_MPS,
            self.temperature_c if self.temperature_c is not None else _DEFAULT_TEMP_C,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list:
        return ["wind_speed_mps", "current_velocity_mps", "temperature_c"]


@dataclass
class FluidPrediction:
    label: str
    probabilities: dict
    confidence: float


# ── Visual Feature Extraction ────────────────────────────────────────────────

def _rgb_hsv_stats(crop_bgr: np.ndarray) -> tuple:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    feats = [
        r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std(),
        h.mean(), h.std(),          # hue
        s.mean(), s.std(),          # saturation
        v.mean(), v.std(),          # brightness
    ]
    names = [
        "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
        "hue_mean", "hue_std",
        "saturation_mean", "saturation_std",
        "brightness_mean", "brightness_std",
    ]
    return feats, names


def _reflectance_features(crop_bgr: np.ndarray) -> tuple:
    """
    Specular-highlight proxy: bright, low-saturation pixels ("sheen")
    are common on hydrocarbon films catching light; water tends to
    scatter more diffusely. Cheap stand-in for a real reflectance
    sensor.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    specular_mask = (v > 0.85) & (s < 0.20)
    specular_fraction = float(specular_mask.mean())
    specular_intensity = float(v[specular_mask].mean()) if specular_mask.any() else 0.0
    feats = [specular_fraction, specular_intensity]
    names = ["specular_fraction", "specular_intensity"]
    return feats, names


def _glcm_features(gray: np.ndarray) -> tuple:
    """Gray-Level Co-occurrence Matrix texture descriptors."""
    from skimage.feature import graycomatrix, graycoprops

    # Quantize to 32 levels — keeps the co-occurrence matrix small and
    # stable on the small crops typical of a detection bbox.
    levels = 32
    q = (gray.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    q = np.clip(q, 0, levels - 1)

    glcm = graycomatrix(
        q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels, symmetric=True, normed=True
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats, names = [], []
    for p in props:
        val = graycoprops(glcm, p).mean()   # average over the 4 angles
        feats.append(float(val))
        names.append(f"glcm_{p}")
    return feats, names


def _lbp_features(gray: np.ndarray, n_points: int = 8, radius: int = 1) -> tuple:
    """Local Binary Pattern histogram (uniform patterns) — surface texture."""
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    names = [f"lbp_bin{i}" for i in range(n_bins)]
    return hist.astype(float).tolist(), names


def _normalize_crop_scale(crop_bgr: np.ndarray, size: int = CANONICAL_CROP_SIZE) -> np.ndarray:
    """
    Resizes (preserving aspect ratio) so the longer side equals `size`,
    then reflect-pads the shorter side up to a size×size canvas.
    Reflect padding (not black) is used so it doesn't introduce a hard
    artificial edge that would corrupt GLCM/LBP texture stats or bias
    the Otsu segmentation used for geometric features.
    """
    h, w = crop_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=interp)

    pad_h, pad_w = size - new_h, size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT101)


def extract_visual_features(crop_bgr: np.ndarray) -> tuple:
    """Returns (feature_list, name_list) for all visual descriptors."""
    if crop_bgr.size == 0:
        raise ValueError("Empty crop passed to extract_visual_features")

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    feats, names = [], []
    for fn in (_rgb_hsv_stats, _reflectance_features):
        f, n = fn(crop_bgr)
        feats += f
        names += n

    f, n = _glcm_features(gray)
    feats += f; names += n

    f, n = _lbp_features(gray)
    feats += f; names += n

    return feats, names


# ── Geometric Feature Extraction ─────────────────────────────────────────────

def _segment_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Rough foreground segmentation of the crop via Otsu thresholding on
    saturation (fluid regions are typically lower-saturation than
    surrounding structure/vegetation, but this is a heuristic — swap in
    a proper mask from the detector if one becomes available, e.g. a
    YOLO-seg output, for cleaner geometric features).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[..., 1]
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Clean up small speckle noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _fractal_dimension(mask: np.ndarray, n_scales: int = 5) -> float:
    """
    Box-counting fractal dimension of the mask boundary. Irregular,
    filamentous edges (typical of a spreading hydrocarbon sheen) box-count
    higher than the smooth, near-circular edge of a compact water pool.
    """
    edges = cv2.Canny(mask, 50, 150)
    if edges.sum() == 0:
        return 1.0

    h, w = edges.shape
    max_dim = max(h, w)
    sizes = np.geomspace(2, max(2, max_dim // 2), num=n_scales).astype(int)
    sizes = sorted(set(sizes))

    counts = []
    for size in sizes:
        if size < 1:
            continue
        n_rows = int(np.ceil(h / size))
        n_cols = int(np.ceil(w / size))
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                block = edges[i*size:(i+1)*size, j*size:(j+1)*size]
                if block.any():
                    count += 1
        counts.append(max(count, 1))

    if len(sizes) < 2 or len(counts) < 2:
        return 1.0

    log_sizes = np.log(1.0 / np.array(sizes[:len(counts)], dtype=np.float64))
    log_counts = np.log(np.array(counts, dtype=np.float64))
    # Slope of log(N) vs log(1/size) ≈ fractal dimension
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return float(np.clip(slope, 1.0, 2.0))


def extract_geometric_features(crop_bgr: np.ndarray) -> tuple:
    """Returns (feature_list, name_list): area_fill_fraction, compactness, elongation, fractal_dimension."""
    mask = _segment_crop(crop_bgr)
    crop_area = float(mask.shape[0] * mask.shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No segmentable region — fall back to bbox-derived defaults
        # rather than crashing the whole feature vector.
        h, w = crop_bgr.shape[:2]
        area_fill_fraction = 1.0
        compactness = 0.0
        elongation = float(max(w, h) / max(1.0, min(w, h)))
        fractal_dim = 1.0
    else:
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        # Fraction of the (canonical-sized) crop the segmented region
        # fills — scale-invariant, unlike a raw pixel count, so it
        # doesn't leak which source dataset's native resolution an
        # image came from.
        area_fill_fraction = min(area / crop_area, 1.0) if crop_area > 0 else 0.0
        perimeter = float(cv2.arcLength(largest, closed=True))

        # Compactness (isoperimetric ratio): 1.0 = perfect circle,
        # → 0 as the boundary gets more irregular/elongated.
        compactness = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        compactness = min(compactness, 1.0)

        if len(largest) >= 5:
            (_, _), (minor_ax, major_ax), _ = cv2.fitEllipse(largest)
            elongation = float(major_ax / max(minor_ax, 1e-3))
        else:
            x, y, w, h = cv2.boundingRect(largest)
            elongation = float(max(w, h) / max(1.0, min(w, h)))

        fractal_dim = _fractal_dimension(mask)

    feats = [area_fill_fraction, compactness, elongation, fractal_dim]
    names = ["area_fill_fraction", "compactness", "elongation", "fractal_dimension"]
    return feats, names


# ── Environmental Feature Extraction ─────────────────────────────────────────

def extract_environmental_features(env: Optional[EnvironmentalConditions]) -> tuple:
    env = env or EnvironmentalConditions()
    return env.to_vector().tolist(), EnvironmentalConditions.feature_names()


# ── Combined Feature Vector ──────────────────────────────────────────────────

def extract_features(
    crop_bgr: np.ndarray,
    env: Optional[EnvironmentalConditions] = None
) -> tuple:
    """
    Full feature vector for one detection crop: visual + geometric +
    environmental, in a fixed, documented order.
    Returns (np.ndarray[float32], list[str] feature_names).
    """
    norm_crop = _normalize_crop_scale(crop_bgr)
    v_feats, v_names = extract_visual_features(norm_crop)
    g_feats, g_names = extract_geometric_features(norm_crop)
    e_feats, e_names = extract_environmental_features(env)

    feats = np.array(v_feats + g_feats + e_feats, dtype=np.float32)
    names = v_names + g_names + e_names
    return feats, names


# ── Tabular Backend (Random Forest / XGBoost / LightGBM) ────────────────────

class TabularFluidClassifier:
    """
    Feature-vector classifier wrapping scikit-learn's estimator API.
    Default: RandomForestClassifier. Optionally XGBoost or LightGBM —
    both degrade gracefully to RandomForest (with a warning) if the
    package isn't installed, so `method=` is safe to set speculatively.
    """

    def __init__(self, method: str = "random_forest", **estimator_kwargs):
        self.method = method
        self.classes_ = list(FLUID_CLASSES)
        self.feature_names_: Optional[list] = None
        self.pipeline = self._build_pipeline(method, estimator_kwargs)
        self._fitted = False

    def _build_pipeline(self, method: str, kwargs: dict):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = self._build_estimator(method, kwargs)
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

    @staticmethod
    def _build_estimator(method: str, kwargs: dict):
        if method == "xgboost":
            try:
                from xgboost import XGBClassifier
                defaults = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="mlogloss")
                defaults.update(kwargs)
                return XGBClassifier(**defaults)
            except ImportError:
                warnings.warn("xgboost not installed — falling back to RandomForest. "
                               "`pip install xgboost` to use the requested backend.")
                method = "random_forest"

        if method == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                defaults = dict(n_estimators=300, max_depth=-1, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8)
                defaults.update(kwargs)
                return LGBMClassifier(**defaults)
            except ImportError:
                warnings.warn("lightgbm not installed — falling back to RandomForest. "
                               "`pip install lightgbm` to use the requested backend.")
                method = "random_forest"

        from sklearn.ensemble import RandomForestClassifier
        defaults = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                         class_weight="balanced", random_state=42, n_jobs=-1)
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)

    def fit(self, X: np.ndarray, y: list, feature_names: Optional[list] = None):
        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._fitted = True
        return self

    def get_feature_importances(self) -> Optional[list]:
        """
        Returns [(feature_name, importance), ...] sorted descending, or
        None if the underlying estimator doesn't expose importances.
        Useful for sanity-checking that predictions are driven by
        physically meaningful features (hue, texture, ...) rather than
        an artifact like crop resolution or source-dataset quirks.
        """
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            return None
        pairs = list(zip(self.feature_names_, clf.feature_importances_.tolist()))
        return sorted(pairs, key=lambda p: p[1], reverse=True)

    def predict_one(self, x: np.ndarray) -> tuple:
        """x: 1D feature vector. Returns (label, {class: prob})."""
        if not self._fitted:
            raise RuntimeError("TabularFluidClassifier.fit() must be called (or a "
                                "saved model loaded) before predict_one().")
        x = x.reshape(1, -1)
        probs = self.pipeline.predict_proba(x)[0]
        label = self.classes_[int(np.argmax(probs))]
        return label, dict(zip(self.classes_, probs.tolist()))

    def evaluate(self, X: np.ndarray, y: list) -> str:
        from sklearn.metrics import classification_report
        preds = self.pipeline.predict(X)
        return classification_report(y, preds, zero_division=0)

    def save(self, path: str):
        import joblib
        joblib.dump({"method": self.method, "pipeline": self.pipeline,
                     "classes_": self.classes_, "feature_names_": self.feature_names_}, path)

    @classmethod
    def load(cls, path: str) -> "TabularFluidClassifier":
        import joblib
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.method = blob["method"]
        obj.pipeline = blob["pipeline"]
        obj.classes_ = blob["classes_"]
        obj.feature_names_ = blob.get("feature_names_")
        obj._fitted = True
        return obj


# ── CNN Backend ───────────────────────────────────────────────────────────────

class _SmallFluidCNN:
    """Lazily builds the torch.nn.Module so `torch` is only required if
    the "cnn" or "vit" backend is actually used."""

    @staticmethod
    def build(n_classes: int, n_env_features: int, img_size: int = 64):
        import torch
        import torch.nn as nn

        class SmallFluidCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                )
                feat_dim = 64 * (img_size // 8) * (img_size // 8)
                self.env_proj = nn.Sequential(nn.Linear(n_env_features, 16), nn.ReLU())
                self.classifier = nn.Sequential(
                    nn.Linear(feat_dim + 16, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, n_classes),
                )

            def forward(self, img, env):
                x = self.features(img).flatten(1)
                e = self.env_proj(env)
                return self.classifier(torch.cat([x, e], dim=1))

        return SmallFluidCNN()


class CNNFluidClassifierWrapper:
    """
    Small end-to-end CNN over the cropped detection image, with
    environmental scalars fused in at the FC head. Reasonable choice
    once you have on the order of a few hundred labelled crops per
    class; below that, the tabular backends (which need far fewer
    samples per parameter) will likely generalise better.
    """

    IMG_SIZE = 64

    def __init__(self, device: str = "cpu", n_classes: int = len(FLUID_CLASSES)):
        import torch

        self.device = device
        self.classes_ = list(FLUID_CLASSES[:n_classes])
        self.model = _SmallFluidCNN.build(
            n_classes=n_classes, n_env_features=3, img_size=self.IMG_SIZE
        ).to(device)
        self._fitted = False
        self._torch = torch

    def _preprocess(self, crop_bgr: np.ndarray):
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5   # roughly [-1, 1]
        return img.transpose(2, 0, 1)   # CHW

    def fit(self, crops: list, env_vecs: list, labels: list,
             epochs: int = 30, lr: float = 1e-3, batch_size: int = 16,
             val_split: float = 0.15):
        """
        crops: list of BGR np.ndarray crops
        env_vecs: list of 3-element environmental feature arrays
        labels: list of class-name strings (subset of FLUID_CLASSES)
        """
        torch = self._torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader, random_split

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}

        class _CropDataset(Dataset):
            def __init__(outer_self, crops, env_vecs, labels, preprocess):
                outer_self.X_img = [preprocess(c) for c in crops]
                outer_self.X_env = env_vecs
                outer_self.y = [label_to_idx[l] for l in labels]

            def __len__(outer_self):
                return len(outer_self.y)

            def __getitem__(outer_self, idx):
                return (
                    torch.tensor(outer_self.X_img[idx], dtype=torch.float32),
                    torch.tensor(outer_self.X_env[idx], dtype=torch.float32),
                    outer_self.y[idx],
                )

        ds = _CropDataset(crops, env_vecs, labels, self._preprocess)
        n_val = max(1, int(len(ds) * val_split)) if len(ds) > 10 else 0
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val]) if n_val else (ds, None)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for img, env, y in train_loader:
                img, env, y = img.to(self.device), env.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img, env)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * img.size(0)

            msg = f"[FluidCNN] epoch {epoch+1}/{epochs} train_loss={total_loss/n_train:.4f}"
            if val_loader:
                acc = self._evaluate_loader(val_loader)
                msg += f" val_acc={acc:.3f}"
            print(msg)

        self.model.eval()
        self._fitted = True
        return self

    def _evaluate_loader(self, loader) -> float:
        torch = self._torch
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for img, env, y in loader:
                img, env, y = img.to(self.device), env.to(self.device), y.to(self.device)
                preds = self.model(img, env).argmax(dim=1)
                correct += int((preds == y).sum())
                total += y.size(0)
        self.model.train()
        return correct / max(total, 1)

    def predict_one(self, crop_bgr: np.ndarray, env_vec: np.ndarray) -> tuple:
        if not self._fitted:
            raise RuntimeError("CNNFluidClassifierWrapper must be trained or "
                                "loaded before predict_one().")
        torch = self._torch
        self.model.eval()
        img = torch.tensor(self._preprocess(crop_bgr), dtype=torch.float32).unsqueeze(0).to(self.device)
        env = torch.tensor(env_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img, env)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label = self.classes_[int(np.argmax(probs))]
        return label, dict(zip(self.classes_, probs.tolist()))

    def save(self, path: str):
        self._torch.save({"state_dict": self.model.state_dict(),
                           "classes_": self.classes_}, path)

    def load(self, path: str):
        blob = self._torch.load(path, map_location=self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.classes_ = blob["classes_"]
        self.model.eval()
        self._fitted = True
        return self


class ViTFluidClassifierWrapper(CNNFluidClassifierWrapper):
    """
    Vision Transformer backend via `timm`, sharing the CNN wrapper's
    training/inference plumbing but swapping the backbone.

    Only reaches for this once you have enough data — ViTs lack CNNs'
    built-in translation/locality bias, so they typically need on the
    order of several thousand labelled crops per class before they
    outperform the small CNN above; below that they tend to overfit or
    underperform it. If `timm` isn't installed, or the dataset looks
    too small, prefer `method="cnn"` instead.
    """

    def __init__(self, device: str = "cpu", n_classes: int = len(FLUID_CLASSES),
                 vit_model_name: str = "vit_tiny_patch16_224"):
        import torch
        try:
            import timm
        except ImportError:
            raise ImportError(
                "ViT backend requires `timm` (`pip install timm`). "
                "Falling back to method='cnn' is recommended unless you "
                "have a few thousand+ labelled crops per class."
            )

        self.device = device
        self.classes_ = list(FLUID_CLASSES[:n_classes])
        self.IMG_SIZE = 224
        self.model = timm.create_model(
            vit_model_name, pretrained=True, num_classes=0
        ).to(device)  # feature extractor; env fusion head below

        import torch.nn as nn
        feat_dim = self.model.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 16, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        ).to(device)
        self.env_proj = nn.Sequential(nn.Linear(3, 16), nn.ReLU()).to(device)
        self._fitted = False
        self._torch = torch

    # NOTE: fit()/predict_one() would need light overrides vs. the CNN
    # wrapper (backbone + head forward pass instead of a single
    # `self.model(img, env)` call) — left as a documented extension
    # point rather than duplicated here, since it's only relevant once
    # a large enough labelled dataset actually exists.


# ── Unified Facade ────────────────────────────────────────────────────────────

class FluidClassifier:
    """
    Single entry point used by the pipeline. Wraps whichever backend is
    selected and exposes a fixed predict/train/save/load interface.

        clf = FluidClassifier(method="random_forest")
        clf.fit(crops=[...], labels=[...], envs=[...])   # tabular backends
        clf.save("fluid_rf.joblib")

        clf = FluidClassifier(method="random_forest", model_path="fluid_rf.joblib")
        pred = clf.predict(frame, bbox, env=EnvironmentalConditions(wind_speed_mps=5.0))
    """

    TABULAR_METHODS = ("random_forest", "xgboost", "lightgbm")
    CROP_PADDING = 10

    def __init__(
        self,
        method: str = "random_forest",
        model_path: Optional[str] = None,
        device: str = "cpu",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.method = method
        self.min_confidence = min_confidence
        self.is_tabular = method in self.TABULAR_METHODS

        if self.is_tabular:
            self.backend = TabularFluidClassifier(method=method)
        elif method == "cnn":
            self.backend = CNNFluidClassifierWrapper(device=device)
        elif method == "vit":
            try:
                self.backend = ViTFluidClassifierWrapper(device=device)
            except ImportError as e:
                warnings.warn(f"{e}\nFalling back to method='cnn'.")
                self.method = "cnn"
                self.is_tabular = False
                self.backend = CNNFluidClassifierWrapper(device=device)
        else:
            raise ValueError(f"Unknown method '{method}'. "
                              f"Choose from: random_forest, xgboost, lightgbm, cnn, vit")

        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ── Inference ─────────────────────────────────────────────────────────

    def _crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        p = self.CROP_PADDING
        cx1, cy1 = max(0, int(x1) - p), max(0, int(y1) - p)
        cx2, cy2 = min(w, int(x2) + p), min(h, int(y2) + p)
        crop = frame[cy1:cy2, cx1:cx2]
        return crop

    def predict(
        self,
        frame: np.ndarray,
        bbox: tuple,
        env: Optional[EnvironmentalConditions] = None,
    ) -> FluidPrediction:
        crop = self._crop(frame, bbox)
        if crop.size == 0:
            return FluidPrediction("unknown", {}, 0.0)

        env = env or EnvironmentalConditions()

        try:
            if self.is_tabular:
                feats, _ = extract_features(crop, env)
                label, probs = self.backend.predict_one(feats)
            else:
                label, probs = self.backend.predict_one(crop, env.to_vector())
        except Exception as e:
            warnings.warn(f"[FluidClassifier] prediction failed, returning 'unknown': {e}")
            return FluidPrediction("unknown", {}, 0.0)

        confidence = probs.get(label, 0.0)
        if confidence < self.min_confidence:
            return FluidPrediction("unknown", probs, confidence)
        return FluidPrediction(label, probs, confidence)

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        crops: list,
        labels: list,
        envs: Optional[list] = None,
        **fit_kwargs,
    ) -> "FluidClassifier":
        """
        crops: list of BGR np.ndarray crops (already cropped to the
               detection region — this does NOT run the detector)
        labels: list of class-name strings (subset of FLUID_CLASSES)
        envs: optional list of EnvironmentalConditions, one per crop
              (defaults imputed per-sample if omitted)
        """
        envs = envs or [EnvironmentalConditions() for _ in crops]

        if self.is_tabular:
            X = np.stack([extract_features(c, e)[0] for c, e in zip(crops, envs)])
            _, feature_names = extract_features(crops[0], envs[0])
            self.backend.fit(X, labels, feature_names=feature_names)
        else:
            env_vecs = [e.to_vector() for e in envs]
            self.backend.fit(crops, env_vecs, labels, **fit_kwargs)
        return self

    def get_feature_importances(self) -> Optional[list]:
        """Tabular backends only — see TabularFluidClassifier.get_feature_importances."""
        if not self.is_tabular:
            return None
        return self.backend.get_feature_importances()

    def evaluate(self, crops: list, labels: list, envs: Optional[list] = None) -> str:
        if not self.is_tabular:
            raise NotImplementedError("evaluate() is currently implemented for "
                                       "tabular backends; use a manual val_loader "
                                       "pass for CNN/ViT (see fit()'s val_split).")
        envs = envs or [EnvironmentalConditions() for _ in crops]
        X = np.stack([extract_features(c, e)[0] for c, e in zip(crops, envs)])
        return self.backend.evaluate(X, labels)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        self.backend.save(path)

    def load(self, path: str):
        if self.is_tabular:
            self.backend = TabularFluidClassifier.load(path)
        else:
            self.backend.load(path)
        return self


# ── Dataset-Building Utility ──────────────────────────────────────────────────

def build_dataset_from_directory(
    root_dir: str,
    env_csv: Optional[str] = None,
) -> tuple:
    """
    Expects:
        root_dir/
            water/            *.jpg | *.png
            light_hydrocarbon/
            heavy_oil/
            drilling_mud/
    Each image is treated as an already-cropped detection region (e.g.
    saved from `SpillImageSaver`, then sorted into class folders by
    `label_fluid_images.py`).

    env_csv (optional): a CSV with columns
        filename,wind_speed_mps,current_velocity_mps,temperature_c
    where `filename` matches the image's basename (not full path).
    Images without a matching row fall back to calm-conditions defaults.
    Omit this if you don't have logged environmental readings yet — the
    classifier still works, it just won't learn from wind/current/temp.

    Returns (crops: list[np.ndarray], labels: list[str], envs: list[EnvironmentalConditions]).
    """
    import csv as _csv

    env_lookup = {}
    if env_csv:
        with open(env_csv, newline="") as f:
            for row in _csv.DictReader(f):
                env_lookup[row["filename"]] = EnvironmentalConditions(
                    wind_speed_mps=_to_float(row.get("wind_speed_mps")),
                    current_velocity_mps=_to_float(row.get("current_velocity_mps")),
                    temperature_c=_to_float(row.get("temperature_c")),
                )

    root = Path(root_dir)
    crops, labels, envs = [], [], []
    for class_name in FLUID_CLASSES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = _safe_imread(img_path)
            if img is None:
                continue
            crops.append(img)
            labels.append(class_name)
            envs.append(env_lookup.get(img_path.name, EnvironmentalConditions()))

    if not crops:
        warnings.warn(f"No labelled images found under {root_dir}. "
                       f"Expected subfolders named: {FLUID_CLASSES}")
    return crops, labels, envs


def _to_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
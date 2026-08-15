"""
PINN-Based Residual Validation Layer
Implements physics residual checks for detected spill regions.

Two governing-equation modes are supported (PINNValidator.pde_mode):

  "thin_film" (original, thesis Section 3.4 — default, backward compatible)
      ∂h/∂t + ∇·(h³/3μ · ∇p) = 0, simplified to
      ∂h/∂t + ∇·q(h) = 0   where q(h) ∝ h³

  "advection_diffusion" (new)
      ∂h/∂t + u·∂h/∂x + v·∂h/∂y = D·∇²h

      This is the same underlying physics — the thin-film equation is
      itself a lubrication-approximation reduction of Navier–Stokes —
      but written in advection-diffusion form so bulk surface flow (u, v)
      and drainage/diffusion (D) enter explicitly. u and v are the sum of
      a gravity/floor-slope component and a wind-drift component:
          u = u_gravity(floor_slope) + u_wind(wind_speed, wind_direction)
      This lets the residual account for sloped floors and wind-driven
      drift without adding sensors — both flagged as future work in the
      thesis (Section 4.5).

The PINN itself is unchanged: a small fully-connected network trained on
collocation points to learn a physically consistent thin-film height
field h(x,y,t). Residual = MSE of whichever governing equation is
active, evaluated at sampled points from the detection region.

Architecture: [3 inputs: x,y,t] → 4×64 tanh layers → [1 output: h]
Trained offline on synthetic thin-film simulations.

IMPORTANT CALIBRATION NOTE: the gravity/wind coupling coefficients and
per-fluid diffusion coefficients below (GRAVITY_COUPLING, WIND_COUPLING_MAP,
DIFFUSION_MAP) are placeholder physical estimates, not values fitted or
validated against real spill data — unlike RESIDUAL_THRESHOLD, which the
thesis validated via 5-fold CV. Treat "advection_diffusion" mode as an
untested mechanism until these constants are calibrated the same way.
"""

import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Callable


# ── PINN Network Architecture ────────────────────────────────────────────────

class ThinFilmPINN(nn.Module):
    """
    Fully connected PINN for thin-film spreading.
    Input:  (x, y, t) normalised to [-1, 1]
    Output: h (film thickness), normalised

    dropout_p > 0 enables MC-Dropout: keeping dropout active at
    inference time (model.train()) turns repeated forward passes
    into samples from an approximate posterior over h, which is
    what PINNValidator uses to estimate residual uncertainty.
    """
    def __init__(self, hidden_dim: int = 64, n_layers: int = 4, dropout_p: float = 0.1,
                 input_dim: int = 3):
        super().__init__()
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        # NOTE: dropout is applied functionally in forward() rather than as
        # nn.Dropout submodules. Inserting Dropout modules into nn.Sequential
        # shifts every subsequent layer's index in state_dict (e.g. Linear
        # layers move from net.0/2/4/6/8 to net.0/3/6/9/12), which silently
        # breaks loading of any checkpoint trained without dropout. Keeping
        # dropout functional preserves the original Linear/Tanh indices
        # regardless of dropout_p, so old checkpoints still load.
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout_p <= 0:
            return self.net(x)
        # Manually walk the Sequential so dropout can be applied after each
        # Tanh without existing as its own indexed submodule. F.dropout
        # respects self.training exactly like nn.Dropout would (on during
        # model.train(), off during model.eval()), so MC-Dropout behaves
        # identically to before — only the state_dict layout changed.
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                x = nn.functional.dropout(x, p=self.dropout_p, training=self.training)
        return x


def _build_input(model: ThinFilmPINN, x_pts: torch.Tensor, y_pts: torch.Tensor,
                  t_pts: torch.Tensor, device: str) -> torch.Tensor:
    """
    Stack (x, y, t) into the network's input tensor. If the loaded model
    expects more than 3 inputs (model.input_dim > 3 — e.g. a checkpoint
    trained with an extra feature such as viscosity), the extra slot(s)
    are padded with 0.0 as a neutral placeholder so the shapes line up
    and the checkpoint LOADS — but 0-padding is almost certainly not the
    right value for whatever that 4th feature actually represents.
    Confirm what it is and wire it in properly before trusting residuals
    from a checkpoint with input_dim > 3 in production.
    """
    base = torch.stack([x_pts, y_pts, t_pts], dim=1)
    extra = model.input_dim - 3
    if extra > 0:
        pad = torch.zeros(base.shape[0], extra, device=device)
        return torch.cat([base, pad], dim=1)
    return base


# ── Residual Computation ─────────────────────────────────────────────────────

def _pinn_residual_pass(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    mu: float,
    device: str,
) -> float:
    """
    A single forward pass through the thin-film PDE-residual computation.
    Whether this is deterministic or stochastic depends entirely on the
    model's current mode (model.eval() vs model.train()) set by the
    caller — this function itself does not touch that mode, so it can
    be reused for both the plain and MC-Dropout residual paths.
    """
    x_pts = x_pts.to(device).requires_grad_(True)
    y_pts = y_pts.to(device).requires_grad_(True)
    t_pts = t_pts.to(device).requires_grad_(True)

    inp = _build_input(model, x_pts, y_pts, t_pts, device)
    h = model(inp)  # (N, 1)

    # Gradients via autograd
    dh_dt = torch.autograd.grad(h, t_pts, grad_outputs=torch.ones_like(h),
                                 create_graph=False, retain_graph=True)[0]
    dh_dx = torch.autograd.grad(h, x_pts, grad_outputs=torch.ones_like(h),
                                 create_graph=False, retain_graph=True)[0]
    dh_dy = torch.autograd.grad(h, y_pts, grad_outputs=torch.ones_like(h),
                                 create_graph=False, retain_graph=False)[0]

    # Flux q(h) = h³ / (3μ)
    h_val = h.squeeze()
    flux_x = (h_val ** 3) / (3.0 * mu)
    flux_y = (h_val ** 3) / (3.0 * mu)

    # Simplified residual (flat surface assumption)
    residual = dh_dt + dh_dx * flux_x + dh_dy * flux_y
    return float(residual.pow(2).mean().detach().cpu())


def _advection_diffusion_pass(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    u: float,
    v: float,
    D: float,
    device: str,
) -> float:
    """
    A single forward pass through the advection-diffusion residual:
        R = ∂h/∂t + u·∂h/∂x + v·∂h/∂y − D·∇²h

    u, v are the effective surface-flow velocity components (gravity +
    wind drift, see compute_flow_velocity), D is the fluid's diffusion/
    spreading coefficient. Needs second derivatives for the Laplacian,
    hence create_graph=True on the first-order x/y gradients.
    """
    x_pts = x_pts.to(device).requires_grad_(True)
    y_pts = y_pts.to(device).requires_grad_(True)
    t_pts = t_pts.to(device).requires_grad_(True)

    inp = _build_input(model, x_pts, y_pts, t_pts, device)
    h = model(inp)  # (N, 1)
    ones = torch.ones_like(h)

    dh_dx = torch.autograd.grad(h, x_pts, grad_outputs=ones,
                                 create_graph=True, retain_graph=True)[0]
    dh_dy = torch.autograd.grad(h, y_pts, grad_outputs=ones,
                                 create_graph=True, retain_graph=True)[0]
    dh_dt = torch.autograd.grad(h, t_pts, grad_outputs=ones,
                                 create_graph=False, retain_graph=True)[0]

    d2h_dx2 = torch.autograd.grad(dh_dx, x_pts, grad_outputs=torch.ones_like(dh_dx),
                                   create_graph=False, retain_graph=True)[0]
    d2h_dy2 = torch.autograd.grad(dh_dy, y_pts, grad_outputs=torch.ones_like(dh_dy),
                                   create_graph=False, retain_graph=False)[0]

    laplacian = d2h_dx2 + d2h_dy2
    residual = dh_dt + u * dh_dx + v * dh_dy - D * laplacian
    return float(residual.pow(2).mean().detach().cpu())


def compute_pinn_residual(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    mu: float = 1.0,
    device: str = "cpu"
) -> float:
    """
    Evaluates the thin-film PDE residual at collocation points using a
    single deterministic forward pass (dropout disabled).
    Returns mean squared residual (MSE_res).

    R(x,y,t) = ∂h/∂t + ∂(h³/3μ)/∂x + ∂(h³/3μ)/∂y

    Lower residual → detection behaves like a real fluid spill.
    """
    model.eval()
    return _pinn_residual_pass(model, x_pts, y_pts, t_pts, mu, device)


def compute_advection_diffusion_residual(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    u: float,
    v: float,
    D: float,
    device: str = "cpu",
) -> float:
    """
    Evaluates the advection-diffusion PDE residual at collocation points
    using a single deterministic forward pass (dropout disabled).
    """
    model.eval()
    return _advection_diffusion_pass(model, x_pts, y_pts, t_pts, u, v, D, device)


def _mc_dropout_loop(
    model: ThinFilmPINN,
    single_pass_fn: Callable[[], float],
    n_samples: int,
) -> tuple[float, float]:
    """
    Shared MC-Dropout driver (thesis future-work item: Bayesian /
    uncertainty-aware PINN). Runs `single_pass_fn` `n_samples` times with
    dropout kept active (model.train()), so the only thing varying
    between calls is the dropout mask, then returns (mean, std) across
    the samples. Works for either PDE residual mode — the caller supplies
    a closure over the collocation points and physical parameters for
    whichever mode is active.
    """
    model.train()  # keep dropout active; no optimizer step happens here
    samples = [single_pass_fn() for _ in range(n_samples)]
    model.eval()
    arr = np.array(samples, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def mc_dropout_residual(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    mu: float = 1.0,
    device: str = "cpu",
    n_samples: int = 20,
) -> tuple[float, float]:
    """
    MC-Dropout uncertainty for the thin-film residual.
    Returns (mean_residual, std_residual). If the network was built with
    dropout_p == 0, all samples are identical and std will be ~0 — the
    caller should treat that as "uncertainty not available" rather than
    "uncertainty confirmed low".
    """
    return _mc_dropout_loop(
        model,
        lambda: _pinn_residual_pass(model, x_pts, y_pts, t_pts, mu, device),
        n_samples,
    )


def mc_dropout_advection_diffusion_residual(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    u: float,
    v: float,
    D: float,
    device: str = "cpu",
    n_samples: int = 20,
) -> tuple[float, float]:
    """MC-Dropout uncertainty for the advection-diffusion residual."""
    return _mc_dropout_loop(
        model,
        lambda: _advection_diffusion_pass(model, x_pts, y_pts, t_pts, u, v, D, device),
        n_samples,
    )


# ── Surface Flow Velocity (gravity + wind drift) ─────────────────────────────

# Placeholder coupling coefficients — see module-level calibration note.
GRAVITY_COUPLING: float = 0.05   # floor-slope → surface velocity coupling

WIND_COUPLING_MAP: dict = {
    # fluid-class → fraction of wind speed transferred to surface drift.
    # Lighter, less viscous fluids are assumed more wind-affected.
    "water": 0.15,
    "light_hydrocarbon": 0.10,
    "heavy_oil": 0.02,
    "drilling_mud": 0.01,
    "unknown": 0.05,
}

DIFFUSION_MAP: dict = {
    # fluid-class → spreading/diffusion coefficient D (normalised units)
    "water": 0.30,
    "light_hydrocarbon": 0.18,
    "heavy_oil": 0.03,
    "drilling_mud": 0.015,
    "unknown": 0.08,
}


def compute_flow_velocity(
    fluid_class: str,
    floor_slope: tuple = (0.0, 0.0),
    wind_speed: float = 0.0,
    wind_direction_deg: float = 0.0,
) -> tuple:
    """
    Effective surface advection velocity (u, v) = gravity component +
    wind-drift component:
        u = u_gravity(floor_slope) + u_wind(wind_speed, wind_direction, fluid)

    floor_slope: (slope_x, slope_y), a fixed per-site descriptor of the
        floor's downhill direction/steepness (0,0) = flat floor.
    wind_speed: m/s (or any consistent unit — only relative magnitude
        matters given WIND_COUPLING_MAP is uncalibrated placeholder data).
    wind_direction_deg: compass-style angle wind is blowing TOWARD,
        0° = +x axis, 90° = +y axis.
    """
    sx, sy = floor_slope
    u_gravity = GRAVITY_COUPLING * sx
    v_gravity = GRAVITY_COUPLING * sy

    coupling = WIND_COUPLING_MAP.get(fluid_class, WIND_COUPLING_MAP["unknown"])
    theta = math.radians(wind_direction_deg)
    u_wind = coupling * wind_speed * math.cos(theta)
    v_wind = coupling * wind_speed * math.sin(theta)

    return (u_gravity + u_wind, v_gravity + v_wind)


# ── Collocation Point Sampler ────────────────────────────────────────────────

def sample_collocation_from_bbox(
    bbox: tuple,
    frame_time: float,
    n_points: int = 64,
    t_window: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (x, y, t) collocation points from within a detection bounding box.
    Coordinates normalised to [-1, 1].

    Args:
        bbox: (x1, y1, x2, y2) pixel coordinates
        frame_time: current timestamp in seconds
        n_points: number of collocation points to sample
        t_window: temporal window size in seconds
    """
    x1, y1, x2, y2 = bbox
    x_raw = np.random.uniform(x1, x2, n_points).astype(np.float32)
    y_raw = np.random.uniform(y1, y2, n_points).astype(np.float32)
    t_raw = np.random.uniform(
        max(0.0, frame_time - t_window), frame_time, n_points
    ).astype(np.float32)

    # Normalise to [-1, 1]
    x_norm = 2.0 * (x_raw - x1) / max(x2 - x1, 1e-6) - 1.0
    y_norm = 2.0 * (y_raw - y1) / max(y2 - y1, 1e-6) - 1.0
    t_norm = 2.0 * (t_raw / max(frame_time + 1e-6, 1e-6)) - 1.0

    return (
        torch.tensor(x_norm),
        torch.tensor(y_norm),
        torch.tensor(t_norm)
    )


# ── Learned Decision Threshold ────────────────────────────────────────────────

class ResidualThresholdNet(nn.Module):
    """
    Small calibrator network replacing the hand-tuned ε = 0.05 cutoff
    (thesis future-work item: adaptive residual thresholding).

    Maps (residual, uncertainty) → P(physically plausible) in [0, 1].
    Deliberately tiny (single hidden layer, 8 units) because the amount
    of labelled calibration data available (confirmed alerts + reviewed
    false positives) is small — a large network would just overfit it.
    """
    def __init__(self, in_dim: int = 2, hidden: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class ThresholdCalibrator:
    """
    Wraps ResidualThresholdNet with fit/predict/save/load, and a graceful
    "unfitted" state so PINNValidator can fall back to the fixed ε
    threshold when no calibration data has been supplied yet.

    Input features are standardised (z-scored) before hitting the
    network; the scaler stats are fitted alongside the weights and
    saved together, since a threshold network is useless without the
    normalisation it was trained on.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = ResidualThresholdNet(in_dim=2).to(device)
        self.is_fitted = False
        self._feat_mean = np.zeros(2, dtype=np.float64)
        self._feat_std = np.ones(2, dtype=np.float64)

    def _featurize(self, residual: float, uncertainty: float) -> torch.Tensor:
        raw = np.array([residual, uncertainty], dtype=np.float64)
        z = (raw - self._feat_mean) / np.where(self._feat_std < 1e-8, 1.0, self._feat_std)
        return torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)

    def predict_proba(self, residual: float, uncertainty: float = 0.0) -> float:
        """Returns P(physically plausible). Only meaningful once fitted."""
        self.model.eval()
        with torch.no_grad():
            x = self._featurize(residual, uncertainty)
            return float(self.model(x).item())

    def fit(
        self,
        residuals: list,
        uncertainties: list,
        labels: list,
        epochs: int = 500,
        lr: float = 0.02,
        weight_decay: float = 1e-3,
    ) -> dict:
        """
        Fit the calibrator on labelled (residual, uncertainty, label) data.
        label = 1 → confirmed real spill, 0 → confirmed false positive.

        Intended to be called offline on data logged from real operation
        (or from `scripts/calibrate_threshold.py`), not per-frame in the
        live pipeline. Returns a small training report dict.
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        uncertainties = np.asarray(uncertainties, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.float64)

        if len(residuals) < 8:
            raise ValueError(
                f"Need at least 8 labelled examples to fit a threshold "
                f"calibrator reliably; got {len(residuals)}."
            )

        feats = np.stack([residuals, uncertainties], axis=1)
        self._feat_mean = feats.mean(axis=0)
        self._feat_std = feats.std(axis=0)
        z = (feats - self._feat_mean) / np.where(self._feat_std < 1e-8, 1.0, self._feat_std)

        x = torch.tensor(z, dtype=torch.float32, device=self.device)
        y = torch.tensor(labels_arr, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()

        history = []
        for epoch in range(epochs):
            opt.zero_grad()
            pred = self.model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            if epoch % max(1, epochs // 10) == 0:
                history.append(float(loss.item()))

        self.model.eval()
        with torch.no_grad():
            final_pred = (self.model(x) >= 0.5).float()
            accuracy = float((final_pred == y).float().mean().item())

        self.is_fitted = True
        return {
            "n_examples": len(residuals),
            "final_loss": history[-1] if history else None,
            "train_accuracy": accuracy,
        }

    def save(self, path: str):
        # Store scaler stats as plain lists (not numpy arrays) so the
        # checkpoint loads cleanly under torch's weights_only=True default.
        torch.save({
            "state_dict": self.model.state_dict(),
            "feat_mean": self._feat_mean.tolist(),
            "feat_std": self._feat_std.tolist(),
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path: str) -> bool:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["state_dict"])
            self._feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float64)
            self._feat_std = np.asarray(ckpt["feat_std"], dtype=np.float64)
            self.is_fitted = bool(ckpt.get("is_fitted", True))
            self.model.eval()
            return True
        except Exception as e:
            print(f"[ThresholdCalibrator] Could not load {path}: {e}")
            return False


# ── Online Adaptation (differentiable losses + threshold/replay state) ──────
#
# Everything below supports PINNValidator(adaptive=True): instead of a fixed
# ε = 0.05, the residual threshold tracks the model's own recent confirmed-
# spill residuals, and the PINN's weights get a few small gradient steps at
# confirmed spill sites so the network keeps fitting the physics at THIS
# camera's actual conditions over time.
#
# IMPORTANT — READ BEFORE ENABLING: fine-tuning purely against the PDE
# residual, with no paired ground-truth height data, has a well-known
# degenerate solution: a network that outputs h≈0 (or any constant)
# everywhere trivially satisfies both the thin-film and advection-diffusion
# equations. That would make every residual→0 and defeat the entire point of
# physics validation. The anchor term in _finetune_step() penalises drift
# away from the originally trained weights specifically to resist this, but
# the mechanism has NOT been validated against held-out true/false spill
# examples — before trusting this in production, watch
# GET /api/adaptation over time and confirm confirmed-vs-rejected residual
# separation isn't collapsing (both trending toward the same small value is
# the collapse signature).

def _pinn_residual_loss(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    mu: float,
    device: str,
) -> torch.Tensor:
    """Same computation as _pinn_residual_pass, but returns the live tensor
    (no .detach()/float()) so it can be backpropagated through for
    fine-tuning. create_graph=True throughout since we differentiate
    through these ops a second time via the fine-tune backward pass."""
    x_pts = x_pts.to(device).requires_grad_(True)
    y_pts = y_pts.to(device).requires_grad_(True)
    t_pts = t_pts.to(device).requires_grad_(True)

    inp = _build_input(model, x_pts, y_pts, t_pts, device)
    h = model(inp)
    ones = torch.ones_like(h)

    dh_dt = torch.autograd.grad(h, t_pts, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dh_dx = torch.autograd.grad(h, x_pts, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dh_dy = torch.autograd.grad(h, y_pts, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    h_val = h.squeeze()
    flux_x = (h_val ** 3) / (3.0 * mu)
    flux_y = (h_val ** 3) / (3.0 * mu)

    residual = dh_dt + dh_dx * flux_x + dh_dy * flux_y
    return residual.pow(2).mean()


def _advection_diffusion_loss(
    model: ThinFilmPINN,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    u: float,
    v: float,
    D: float,
    device: str,
) -> torch.Tensor:
    """Tensor-returning advection-diffusion residual for fine-tuning
    (see _pinn_residual_loss docstring)."""
    x_pts = x_pts.to(device).requires_grad_(True)
    y_pts = y_pts.to(device).requires_grad_(True)
    t_pts = t_pts.to(device).requires_grad_(True)

    inp = _build_input(model, x_pts, y_pts, t_pts, device)
    h = model(inp)
    ones = torch.ones_like(h)

    dh_dx = torch.autograd.grad(h, x_pts, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dh_dy = torch.autograd.grad(h, y_pts, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dh_dt = torch.autograd.grad(h, t_pts, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    d2h_dx2 = torch.autograd.grad(dh_dx, x_pts, grad_outputs=torch.ones_like(dh_dx),
                                   create_graph=True, retain_graph=True)[0]
    d2h_dy2 = torch.autograd.grad(dh_dy, y_pts, grad_outputs=torch.ones_like(dh_dy),
                                   create_graph=True, retain_graph=True)[0]

    laplacian = d2h_dx2 + d2h_dy2
    residual = dh_dt + u * dh_dx + v * dh_dy - D * laplacian
    return residual.pow(2).mean()


class ReplayBuffer:
    """
    Bounded history of temporally-confirmed detections, used both to
    report adaptation state and to sample fine-tuning batches. Only
    confirmed detections are added (see PINNValidator.record_confirmation) —
    this is a weak positive-only signal (multiple independent frames
    agreeing something is a real, persisting spill), not a substitute for
    the labelled data ThresholdCalibrator needs, which also sees confirmed
    false positives.
    """
    def __init__(self, maxlen: int = 500):
        self.maxlen = maxlen
        self._buf: deque = deque(maxlen=maxlen)

    def add(self, bbox, frame_time, fluid_class, residual, uncertainty,
            wind_speed=0.0, wind_direction_deg=0.0):
        self._buf.append({
            "bbox": bbox, "frame_time": frame_time, "fluid_class": fluid_class,
            "residual": residual, "uncertainty": uncertainty,
            "wind_speed": wind_speed, "wind_direction_deg": wind_direction_deg,
        })

    def sample(self, n: int = 8) -> list:
        buf = list(self._buf)
        if len(buf) <= n:
            return buf
        idx = np.random.choice(len(buf), size=n, replace=False)
        return [buf[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)


class AdaptiveThreshold:
    """
    Online-adapting residual threshold, in place of the fixed ε = 0.05.

    Tracks an EMA of residuals from temporally-confirmed detections and
    sets epsilon = PERCENTILE_MARGIN × EMA, clamped to [EPS_MIN, EPS_MAX]
    so it can't drift to something unusably strict or permissive. This
    adapts to a specific camera's real operating residual range without
    needing operator-labelled data — but unlike ThresholdCalibrator, it
    never sees false positives, so it can only shift where the "plausible"
    boundary sits, not learn to separate the two classes.
    """
    EPS_MIN: float = 0.02
    EPS_MAX: float = 0.15
    EMA_ALPHA: float = 0.1
    PERCENTILE_MARGIN: float = 1.5

    def __init__(self):
        self.epsilon: float = 0.05
        self._ema_confirmed_residual: Optional[float] = None
        self.history: deque = deque(maxlen=200)   # [(timestamp, epsilon), ...]

    def update(self, residual: float) -> float:
        if self._ema_confirmed_residual is None:
            self._ema_confirmed_residual = residual
        else:
            self._ema_confirmed_residual = (
                (1 - self.EMA_ALPHA) * self._ema_confirmed_residual
                + self.EMA_ALPHA * residual
            )
        new_eps = self.PERCENTILE_MARGIN * self._ema_confirmed_residual
        self.epsilon = float(np.clip(new_eps, self.EPS_MIN, self.EPS_MAX))
        self.history.append((time.time(), self.epsilon))
        return self.epsilon


@dataclass
class AdaptationState:
    total_confirmations: int = 0
    total_finetune_steps: int = 0
    last_finetune_time: Optional[float] = None
    last_save_time: Optional[float] = None


# ── PINN Validator ────────────────────────────────────────────────────────────

@dataclass
class PINNResult:
    residual: float
    passed: bool
    threshold: float
    reason: str = ""
    uncertainty: float = 0.0            # MC-Dropout std of the residual
    probability: Optional[float] = None  # P(physically plausible), if calibrator fitted
    decision_method: str = "fixed_threshold"  # "fixed_threshold" | "learned_threshold"
    detected_fluid_class: Optional[str] = None   # best-fit class when fluid_class="auto"
    candidate_residuals: Optional[dict] = None   # {fluid_class: residual}, auto mode only


class PINNValidator:
    """
    Loads (or initialises) the PINN model and validates detections against
    physics: either the original thin-film residual, or the newer
    advection-diffusion residual (see pde_mode).

    Decision logic, in order of preference:
      1. If a fitted ThresholdCalibrator is available, use
         P(physically plausible) >= DECISION_PROB_THRESHOLD.
      2. Else if adaptive=True, use the self-adapting AdaptiveThreshold
         (starts at ε=0.05, tracks confirmed-detection residuals).
      3. Otherwise fall back to the original fixed residual threshold
         ε = 0.05 (thesis Section 3.4.4).

    Uncertainty (MC-Dropout std across N stochastic passes) is always
    computed when use_mc_dropout=True and reported on PINNResult, since
    it's cheap and useful even before a calibrator has been fitted.

    pde_mode:
      "thin_film"           — original governing equation (default; matches
                               published thesis results).
      "advection_diffusion" — ∂h/∂t + u·∂h/∂x + v·∂h/∂y = D·∇²h, with u/v
                               from compute_flow_velocity(floor_slope, wind).
                               Uncalibrated — see module docstring.

    adaptive=True enables the online adaptation system (see the module
    docstring above ReplayBuffer for the important trivial-solution
    caveat before relying on this in production): confirmed detections
    feed a replay buffer, nudge AdaptiveThreshold, and every
    `finetune_every`-th confirmation runs a few small gradient steps
    on the PINN itself via record_confirmation().
    """

    RESIDUAL_THRESHOLD: float = 0.05   # ε from thesis (5-fold CV: 0.047–0.053)
    DECISION_PROB_THRESHOLD: float = 0.5
    N_COLLOCATION: int = 64
    MC_SAMPLES: int = 20
    AUTOSAVE_EVERY: int = 5             # autosave weights every N fine-tune steps
    VISCOSITY_MAP: dict = None         # fluid-class → μ (mPa·s, normalised)

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        dropout_p: float = 0.1,
        use_mc_dropout: bool = True,
        threshold_calibrator_path: Optional[str] = None,
        pde_mode: str = "thin_film",
        floor_slope: tuple = (0.0, 0.0),
        adaptive: bool = False,
        weights_save_path: Optional[str] = None,
        finetune_every: int = 5,
        finetune_steps: int = 3,
        finetune_lr: float = 1e-4,
        anchor_weight: float = 0.01,
        replay_buffer_size: int = 500,
        input_dim: int = 3,
    ):
        if pde_mode not in ("thin_film", "advection_diffusion"):
            raise ValueError(f"Unknown pde_mode: {pde_mode!r}")
        self.device = device
        self.use_mc_dropout = use_mc_dropout
        self.pde_mode = pde_mode
        self.floor_slope = floor_slope
        self.model = ThinFilmPINN(hidden_dim=64, n_layers=4, dropout_p=dropout_p,
                                   input_dim=input_dim).to(device)
        self.VISCOSITY_MAP = {
            "water": 1.0,
            "light_hydrocarbon": 2.5,
            "heavy_oil": 100.0,
            "drilling_mud": 50.0,
            "unknown": 10.0,
        }

        self.learned_mu: Optional[float] = None  # from checkpoint's log_mu, if present

        if model_path:
            try:
                state = torch.load(model_path, map_location=device)
                ckpt_input_dim = state["net.0.weight"].shape[1]
                if ckpt_input_dim != input_dim:
                    print(f"[PINN] Checkpoint's input layer expects "
                          f"{ckpt_input_dim} inputs, not {input_dim} — rebuilding "
                          f"the network with input_dim={ckpt_input_dim} so these "
                          f"weights actually load, instead of silently falling "
                          f"back to random init.")
                    input_dim = ckpt_input_dim
                    self.model = ThinFilmPINN(hidden_dim=64, n_layers=4,
                                               dropout_p=dropout_p,
                                               input_dim=input_dim).to(device)

                # strict=False: tolerate extra top-level scalars (e.g. log_mu/
                # log_gamma) that aren't part of the net.* backbone rather than
                # failing the entire load over them — those get inspected
                # separately below instead of silently discarding real
                # trained weights.
                result = self.model.load_state_dict(state, strict=False)
                if result.missing_keys:
                    raise RuntimeError(
                        f"backbone keys missing from checkpoint: {result.missing_keys} "
                        f"— this is a real architecture mismatch, not just extra params."
                    )
                print(f"[PINN] Loaded weights from {model_path} (input_dim={input_dim})")

                if result.unexpected_keys:
                    print(f"[PINN] Checkpoint has {len(result.unexpected_keys)} extra "
                          f"top-level parameter(s) not in ThinFilmPINN: "
                          f"{result.unexpected_keys} (backbone still loaded fine).")
                    if "log_mu" in result.unexpected_keys:
                        self.learned_mu = float(torch.exp(state["log_mu"]).item())
                        print(f"[PINN] Found log_mu in checkpoint — using this "
                              f"network's own learned viscosity μ={self.learned_mu:.4f} "
                              f"for thin_film residuals instead of VISCOSITY_MAP lookups. "
                              f"This checkpoint was trained with ONE global viscosity, "
                              f"not conditioned per fluid class — auto fluid-class "
                              f"detection via residual comparison does not apply to it "
                              f"(see PINNValidator.validate).")
            except Exception as e:
                print(f"[PINN] Could not load weights ({e}). Using random init.")
        else:
            print("[PINN] No weights path provided — using untrained network.")
            print("       Set PINN_WEIGHTS_PATH in app.py to load trained weights.")

        if pde_mode == "advection_diffusion":
            print("[PINN] pde_mode=advection_diffusion — NOTE: gravity/wind/diffusion "
                  "coefficients are uncalibrated placeholders (see module docstring).")

        # Learned threshold calibrator (starts unfitted → fixed-ε fallback)
        self.calibrator = ThresholdCalibrator(device=device)
        if threshold_calibrator_path:
            loaded = self.calibrator.load(threshold_calibrator_path)
            if loaded:
                print(f"[PINN] Loaded threshold calibrator from {threshold_calibrator_path}")
            else:
                print("[PINN] Calibrator load failed — using fixed ε threshold.")
        else:
            print("[PINN] No threshold calibrator provided — using fixed ε threshold.")

        # Online adaptation (opt-in; see module docstring above ReplayBuffer)
        self.adaptive = adaptive
        self.weights_save_path = weights_save_path
        self.finetune_every = finetune_every
        self.finetune_steps = finetune_steps
        self.finetune_lr = finetune_lr
        self.anchor_weight = anchor_weight
        self.adaptive_thresh = AdaptiveThreshold()
        self.replay_buffer = ReplayBuffer(maxlen=replay_buffer_size)
        self.adapt_state = AdaptationState()
        # Snapshot of the weights at load time, used to regularise fine-tuning
        # back toward the originally trained network (see module docstring).
        self._initial_state_dict = {
            k: v.clone().detach() for k, v in self.model.state_dict().items()
        }
        if self.adaptive:
            print(f"[PINN] Adaptive threshold + online fine-tuning ENABLED "
                  f"(finetune every {finetune_every} confirmations, "
                  f"save path={weights_save_path or '(not set — adaptation will not persist)'})")


    def _residual_and_uncertainty(
        self,
        fluid_class: str,
        x_pts: torch.Tensor,
        y_pts: torch.Tensor,
        t_pts: torch.Tensor,
        wind_speed: float,
        wind_direction_deg: float,
        use_mc: bool,
    ) -> tuple:
        """Residual (+ uncertainty if use_mc) for one fluid-class hypothesis,
        under whichever pde_mode is active. Shared by the auto-detect
        ranking pass (use_mc=False, cheap) and the final decision
        (use_mc=self.use_mc_dropout)."""
        if self.pde_mode == "advection_diffusion":
            u, v = compute_flow_velocity(
                fluid_class, self.floor_slope, wind_speed, wind_direction_deg
            )
            D = DIFFUSION_MAP.get(fluid_class, DIFFUSION_MAP["unknown"])
            if use_mc:
                return mc_dropout_advection_diffusion_residual(
                    self.model, x_pts, y_pts, t_pts, u, v, D,
                    device=self.device, n_samples=self.MC_SAMPLES,
                )
            return compute_advection_diffusion_residual(
                self.model, x_pts, y_pts, t_pts, u, v, D, device=self.device
            ), 0.0
        else:
            # If the loaded checkpoint carried its own learned μ (log_mu),
            # that's what this specific network actually represents — use
            # it regardless of fluid_class, rather than substituting a
            # per-fluid guess the network was never trained to respond to.
            mu = self.learned_mu if self.learned_mu is not None \
                else self.VISCOSITY_MAP.get(fluid_class, 10.0)
            if use_mc:
                return mc_dropout_residual(
                    self.model, x_pts, y_pts, t_pts, mu=mu,
                    device=self.device, n_samples=self.MC_SAMPLES,
                )
            return compute_pinn_residual(
                self.model, x_pts, y_pts, t_pts, mu=mu, device=self.device
            ), 0.0

    def validate(
        self,
        bbox: tuple,
        frame_time: float,
        fluid_class: str = "auto",
        wind_speed: float = 0.0,
        wind_direction_deg: float = 0.0,
    ) -> PINNResult:
        """
        Validate a detection bounding box against the active physics model.
        Returns PINNResult with residual, uncertainty, and pass/fail decision.

        fluid_class="auto" (default): the vision model has no fluid-type
        classifier — it only detects "spill vs not spill" — so committing
        every detection to one hand-picked fluid's viscosity/diffusion
        profile silently rejects anything that isn't that fluid (this is
        why "only heavy oil" was passing: the operator UI's fluid dropdown
        was pinning ALL detections to heavy_oil's physics regardless of
        what was actually spilled). In "auto" mode, the residual is
        computed under every known fluid hypothesis and the best-fitting
        one is used and reported back as detected_fluid_class — a real
        spill of any known fluid type can then pass, not just whichever
        class happened to be selected.

        Pass an explicit fluid_class only when you positively know the
        substance (e.g. a single-fluid tank you're monitoring).

        wind_speed / wind_direction_deg are only used when
        pde_mode == "advection_diffusion"; harmless no-ops in "thin_film"
        mode. floor_slope is a per-deployment constant set at construction
        (a camera's floor doesn't change slope frame to frame).
        """
        x_pts, y_pts, t_pts = sample_collocation_from_bbox(
            bbox, frame_time, n_points=self.N_COLLOCATION
        )

        uncertainty = 0.0
        detected_fluid_class = fluid_class
        candidate_residuals = None
        fluid_autodetect_applicable = not (
            self.pde_mode == "thin_film" and self.learned_mu is not None
        )
        try:
            if fluid_class == "auto" and fluid_autodetect_applicable:
                # Cheap deterministic ranking pass across every known fluid
                # hypothesis, then full precision (+ MC-Dropout) only on
                # the best-fitting one.
                candidate_residuals = {}
                for fc in self.VISCOSITY_MAP.keys():
                    r, _ = self._residual_and_uncertainty(
                        fc, x_pts, y_pts, t_pts, wind_speed, wind_direction_deg,
                        use_mc=False,
                    )
                    candidate_residuals[fc] = round(r, 5)
                detected_fluid_class = min(candidate_residuals, key=candidate_residuals.get)
            elif fluid_class == "auto":
                # This checkpoint has its own single learned μ — varying μ
                # per fluid hypothesis doesn't correspond to anything this
                # particular network was trained to respond to (see
                # PINNValidator.__init__ log_mu handling), so report
                # honestly rather than attributing a specific fluid label
                # based on a comparison that isn't meaningful here.
                detected_fluid_class = "unknown (single-viscosity model)"

            residual, uncertainty = self._residual_and_uncertainty(
                detected_fluid_class, x_pts, y_pts, t_pts, wind_speed, wind_direction_deg,
                use_mc=self.use_mc_dropout,
            )
        except Exception as e:
            # Fallback: residual unknown, pass through
            print(f"[PINN] Residual computation failed: {e}")
            residual = 0.0

        if self.calibrator.is_fitted:
            probability = self.calibrator.predict_proba(residual, uncertainty)
            passed = probability >= self.DECISION_PROB_THRESHOLD
            reason = "" if passed else (
                f"P(plausible)={probability:.3f} < {self.DECISION_PROB_THRESHOLD}"
            )
            return PINNResult(
                residual=residual,
                passed=passed,
                threshold=self.DECISION_PROB_THRESHOLD,
                reason=reason,
                uncertainty=uncertainty,
                probability=probability,
                decision_method="learned_threshold",
                detected_fluid_class=detected_fluid_class,
                candidate_residuals=candidate_residuals,
            )

        if self.adaptive:
            eps = self.adaptive_thresh.epsilon
            passed = residual <= eps
            reason = "" if passed else f"residual {residual:.4f} > adaptive ε={eps:.4f}"
            return PINNResult(
                residual=residual,
                passed=passed,
                threshold=eps,
                reason=reason,
                uncertainty=uncertainty,
                probability=None,
                decision_method="adaptive_threshold",
                detected_fluid_class=detected_fluid_class,
                candidate_residuals=candidate_residuals,
            )

        passed = residual <= self.RESIDUAL_THRESHOLD
        reason = "" if passed else f"residual {residual:.4f} > ε={self.RESIDUAL_THRESHOLD}"
        return PINNResult(
            residual=residual,
            passed=passed,
            threshold=self.RESIDUAL_THRESHOLD,
            reason=reason,
            uncertainty=uncertainty,
            probability=None,
            decision_method="fixed_threshold",
            detected_fluid_class=detected_fluid_class,
            candidate_residuals=candidate_residuals,
        )

    def record_confirmation(
        self,
        bbox: tuple,
        frame_time: float,
        fluid_class: str,
        residual: float,
        uncertainty: float,
        wind_speed: float = 0.0,
        wind_direction_deg: float = 0.0,
    ):
        """
        Call this once a detection has been TEMPORALLY CONFIRMED (multiple
        independent frames agreeing it's a real, persisting spill) — not
        on every raw detection. Feeds the replay buffer, updates the
        adaptive threshold, and every `finetune_every`-th confirmation
        runs a short online fine-tune step. No-ops entirely if
        adaptive=False, so it's safe to call unconditionally from the
        pipeline's confirmation path.
        """
        if not self.adaptive:
            return
        if fluid_class in (None, "auto"):
            fluid_class = "unknown"

        self.replay_buffer.add(
            bbox, frame_time, fluid_class, residual, uncertainty,
            wind_speed=wind_speed, wind_direction_deg=wind_direction_deg,
        )
        self.adaptive_thresh.update(residual)
        self.adapt_state.total_confirmations += 1

        if self.adapt_state.total_confirmations % self.finetune_every == 0:
            self._finetune_step()

    def _finetune_step(self):
        """
        A handful of gradient steps nudging the PINN to better satisfy the
        governing equation at recently confirmed spill sites, regularised
        by an anchor term back toward the originally loaded weights (see
        module docstring for why the anchor matters).
        """
        if len(self.replay_buffer) < 2:
            return

        samples = self.replay_buffer.sample(n=min(8, len(self.replay_buffer)))
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.finetune_lr)

        for _ in range(self.finetune_steps):
            opt.zero_grad()
            physics_loss = 0.0
            for s in samples:
                x_pts, y_pts, t_pts = sample_collocation_from_bbox(
                    s["bbox"], s["frame_time"], n_points=32
                )
                if self.pde_mode == "advection_diffusion":
                    u, v = compute_flow_velocity(
                        s["fluid_class"], self.floor_slope,
                        s["wind_speed"], s["wind_direction_deg"],
                    )
                    D = DIFFUSION_MAP.get(s["fluid_class"], DIFFUSION_MAP["unknown"])
                    loss = _advection_diffusion_loss(
                        self.model, x_pts, y_pts, t_pts, u, v, D, self.device
                    )
                else:
                    mu = self.VISCOSITY_MAP.get(s["fluid_class"], 10.0)
                    loss = _pinn_residual_loss(
                        self.model, x_pts, y_pts, t_pts, mu, self.device
                    )
                physics_loss = physics_loss + loss
            physics_loss = physics_loss / len(samples)

            # Anchor regularisation — see module docstring on the trivial
            # h≈0 solution risk. Pulls weights back toward the originally
            # trained network rather than letting the physics-only loss
            # drag them toward whatever trivially minimises the residual.
            anchor_loss = 0.0
            for name, param in self.model.named_parameters():
                anchor_loss = anchor_loss + (
                    (param - self._initial_state_dict[name].to(self.device)) ** 2
                ).sum()

            total_loss = physics_loss + self.anchor_weight * anchor_loss
            total_loss.backward()
            opt.step()

        self.model.eval()
        self.adapt_state.total_finetune_steps += 1
        self.adapt_state.last_finetune_time = time.time()

        if self.weights_save_path and (
            self.adapt_state.total_finetune_steps % self.AUTOSAVE_EVERY == 0
        ):
            self._save_weights()

    def _save_weights(self):
        if not self.weights_save_path:
            print("[PINN] No weights_save_path set — cannot save adapted weights.")
            return
        torch.save(self.model.state_dict(), self.weights_save_path)
        self.adapt_state.last_save_time = time.time()
        print(f"[PINN] Adapted weights saved to {self.weights_save_path}")

    def get_adaptation_state(self) -> dict:
        """Full adaptation snapshot for the /api/adaptation dashboard route."""
        return {
            "adaptive_enabled": self.adaptive,
            "calibrator_fitted": self.calibrator.is_fitted,
            "active_decision_method": (
                "learned_threshold" if self.calibrator.is_fitted
                else "adaptive_threshold" if self.adaptive
                else "fixed_threshold"
            ),
            "epsilon": self.adaptive_thresh.epsilon,
            "epsilon_bounds": [self.adaptive_thresh.EPS_MIN, self.adaptive_thresh.EPS_MAX],
            "epsilon_history": [
                {"t": t, "epsilon": e} for t, e in list(self.adaptive_thresh.history)[-30:]
            ],
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_maxlen": self.replay_buffer.maxlen,
            "total_confirmations": self.adapt_state.total_confirmations,
            "total_finetune_steps": self.adapt_state.total_finetune_steps,
            "finetune_every": self.finetune_every,
            "last_finetune_time": self.adapt_state.last_finetune_time,
            "last_save_time": self.adapt_state.last_save_time,
            "weights_save_path": self.weights_save_path,
        }

    def stop(self):
        """Called on app shutdown (see app.py atexit) — persists adapted weights."""
        if self.adaptive and self.weights_save_path:
            self._save_weights()

    def calibrate_threshold(
        self,
        residuals: list,
        uncertainties: list,
        labels: list,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Fit the learned-threshold calibrator from labelled field data and,
        from this call onward, switch validate() over to using it.

        residuals/uncertainties: from PINNResult.residual / .uncertainty,
            logged for past detections.
        labels: 1 = operator-confirmed real spill, 0 = confirmed false positive.
        """
        report = self.calibrator.fit(residuals, uncertainties, labels)
        if save_path:
            self.calibrator.save(save_path)
            print(f"[PINN] Threshold calibrator saved to {save_path}")
        return report
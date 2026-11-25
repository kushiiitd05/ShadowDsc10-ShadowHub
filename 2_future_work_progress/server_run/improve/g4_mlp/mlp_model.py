"""Entity-embedding MLP for log1p(volume): model, training loop, batched prediction."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from mlp_prep import Preprocessor


@dataclass(frozen=True)
class MlpConfig:
    hidden: tuple[int, ...] = (512, 256, 128)
    hub_dim: int = 16
    cat_dim_cap: int = 8
    dropout: float = 0.1
    norm: str = "batch"            # batch | layer
    lr: float = 2e-3               # OneCycle peak
    weight_decay: float = 1e-2
    batch_size: int = 4096
    epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0
    bf16: bool = True              # autocast only takes effect on CUDA
    hub_col: str = "HubID"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    model: nn.Module
    best_rmse: float
    best_epoch: int
    history: list[dict]
    fit_seconds: float


def embedding_dim(name: str, cardinality: int, cfg: MlpConfig) -> int:
    if name == cfg.hub_col:
        return cfg.hub_dim
    return int(min(cfg.cat_dim_cap, max(2, (cardinality + 1) // 2)))


class EntityMlp(nn.Module):
    def __init__(self, n_features: int, cardinalities: list[int], emb_dims: list[int], cfg: MlpConfig,
                 target_mean: float, target_std: float) -> None:
        super().__init__()
        # index 0 = category unseen in training: padding_idx keeps it a zero vector with no gradient
        self.embeddings = nn.ModuleList(nn.Embedding(card, dim, padding_idx=0) for card, dim in zip(cardinalities, emb_dims))
        norm = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}[cfg.norm]
        layers: list[nn.Module] = []
        width = n_features + sum(emb_dims)
        for h in cfg.hidden:
            layers += [nn.Linear(width, h), norm(h), nn.SiLU(), nn.Dropout(cfg.dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        out = self.net(torch.cat(parts, dim=1)).squeeze(1).float()
        return out * self.target_std + self.target_mean


def build_model(pre: Preprocessor, cfg: MlpConfig, target_mean: float, target_std: float) -> EntityMlp:
    cards = pre.cat_cardinalities()
    dims = [embedding_dim(c, k, cfg) for c, k in zip(pre.cat_cols, cards)]
    return EntityMlp(pre.n_features, cards, dims, cfg, target_mean, max(target_std, 1e-6))


def predict_log(model: nn.Module, x_num: np.ndarray, x_cat: np.ndarray, device: str, batch: int = 65536) -> np.ndarray:
    """Predicted log1p(volume) as float64, evaluated in fp32 eval mode."""
    model = model.to(device).eval()
    out = np.empty(len(x_num), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(x_num), batch):
            xn = torch.from_numpy(x_num[i:i + batch]).to(device)
            xc = torch.from_numpy(x_cat[i:i + batch]).to(device)
            out[i:i + batch] = model(xn, xc).double().cpu().numpy()
    return out


def _holdout_rmse(model: nn.Module, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray, device: str) -> float:
    return float(np.sqrt(np.mean((predict_log(model, x_num, x_cat, device) - y) ** 2)))


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_mlp(pre: Preprocessor, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray,
              h_num: np.ndarray, h_cat: np.ndarray, h_y: np.ndarray, cfg: MlpConfig, seed: int, device: str,
              log: bool = False) -> TrainResult:
    """Train on (x, y = log1p volume); early stop on the chronological holdout RMSE (log space); return best-epoch weights.

    The whole training block is moved to `device` once (about 1.6 GB for 1.1M x 360 float32); batches are index-gathered there.
    """
    seed_everything(seed)
    use_bf16 = cfg.bf16 and device.startswith("cuda")
    model = build_model(pre, cfg, float(np.mean(y)), float(np.std(y))).to(device)
    xn, xc = torch.from_numpy(x_num).to(device), torch.from_numpy(x_cat).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    n = len(yt)
    batch = min(cfg.batch_size, n)
    steps = max(n // batch, 1)  # drop the ragged last batch (BatchNorm needs >1 row)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps, pct_start=0.2)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, history = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for s in range(steps):
            idx = order[s * batch:(s + 1) * batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(xn[idx], xc[idx])
            loss = torch.mean((pred - yt[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            total += float(loss.detach())
        score = _holdout_rmse(model, h_num, h_cat, h_y, device)
        history.append({"epoch": epoch, "train_rmse": float(np.sqrt(total / steps)), "hold_rmse": score})
        if log:
            print(f"  epoch {epoch} train {history[-1]['train_rmse']:.4f} hold {score:.4f}", flush=True)
        if score < best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    del xn, xc, yt
    return TrainResult(model, best, best_epoch, history, time.perf_counter() - start)

"""Save / load TilingGoNet checkpoints (the codebase trained nets in-memory only)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..nn.model import NetConfig, TilingGoNet


def save_checkpoint(net: TilingGoNet, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "cfg": asdict(net.cfg), "meta": meta or {}}, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[TilingGoNet, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = TilingGoNet(NetConfig(**ckpt["cfg"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt.get("meta", {})

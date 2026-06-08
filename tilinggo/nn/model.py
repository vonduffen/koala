"""The geometry-blind graph neural network (ARCHITECTURE.md §5.3).

One set of weights plays every tiling: message passing runs over the padded neighbour-index
tensor from ``encoding``, so only the (non-trainable) adjacency changes with the board. The net
is board-size independent, which is what makes an N=5-trained net transfer to other sizes and
tilings (the prior Delta-Go engine demonstrated this for size transfer).

Architecture: input MLP → ``H`` hidden; ``L`` pre-norm residual blocks
    h ← h + MLP(LayerNorm(h ‖ mean_j h_j ‖ max_j h_j ‖ g))
with ``g`` a masked global mean-pool (KataGo-style global context — matters for ko/komi). Heads:
per-node policy (+ pass logit from g), scalar value (tanh), per-node ownership (3-way), scalar
score, and an auxiliary per-node opponent-reply policy.

Masking is exact: padded nodes are excluded from the global pool, and real nodes' neighbour
lists only ever reference real nodes, so padded entries provably cannot affect real outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import FEATURE_DIM, EncodedBatch

_NEG_INF = -1e9


@dataclass(frozen=True)
class NetConfig:
    in_dim: int = FEATURE_DIM
    hidden: int = 96      # H (TUNE)
    blocks: int = 8       # L (TUNE)


def _gather_neighbours(h: torch.Tensor, nbr_index: torch.Tensor) -> torch.Tensor:
    """h: [B, N, H], nbr_index: [B, N, D] → [B, N, D, H] of neighbour hiddens."""
    b, n, hdim = h.shape
    d = nbr_index.shape[2]
    idx = nbr_index.reshape(b, n * d, 1).expand(b, n * d, hdim)
    return torch.gather(h, 1, idx).reshape(b, n, d, hdim)


def _masked_global(h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over nodes → [B, H] (padded nodes excluded)."""
    m = node_mask.unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class MessageBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * hidden)
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.ReLU(),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(self, h, nbr_index, nbr_mask, node_mask):
        nbr = _gather_neighbours(h, nbr_index)            # [B, N, D, H]
        mask = nbr_mask.unsqueeze(-1)                     # [B, N, D, 1]
        deg = nbr_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean_j = (nbr * mask).sum(dim=2) / deg            # [B, N, H]
        max_j = torch.where(mask.bool(), nbr, torch.full_like(nbr, _NEG_INF)).max(dim=2).values
        max_j = torch.where(node_mask.unsqueeze(-1).bool(), max_j, torch.zeros_like(max_j))
        g = _masked_global(h, node_mask).unsqueeze(1).expand_as(h)  # [B, N, H]
        z = torch.cat([h, mean_j, max_j, g], dim=-1)
        return h + self.mlp(self.norm(z))


class TilingGoNet(nn.Module):
    """Policy/value/aux GNN. Forward takes an :class:`EncodedBatch`, returns a dict of outputs."""

    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.input = nn.Sequential(nn.Linear(cfg.in_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList(MessageBlock(h) for _ in range(cfg.blocks))
        self.final_norm = nn.LayerNorm(h)

        self.policy_node = nn.Linear(h, 1)
        self.pass_head = nn.Linear(h, 1)
        self.value_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.score_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.ownership_head = nn.Linear(h, 3)
        self.reply_node = nn.Linear(h, 1)

    def forward(self, batch: EncodedBatch) -> dict[str, torch.Tensor]:
        node_mask = batch.node_mask
        h = self.input(batch.x)
        for block in self.blocks:
            h = block(h, batch.nbr_index, batch.nbr_mask, node_mask)
        h = self.final_norm(h)
        g = _masked_global(h, node_mask)  # [B, H]

        node_logits = self.policy_node(h).squeeze(-1)          # [B, N]
        node_logits = node_logits.masked_fill(node_mask == 0, _NEG_INF)
        pass_logit = self.pass_head(g)                          # [B, 1]
        policy_logits = torch.cat([node_logits, pass_logit], dim=1)  # [B, N+1]

        return {
            "policy_logits": policy_logits,                    # over N+1 (last = pass)
            "value": torch.tanh(self.value_head(g)).squeeze(-1),       # [B] in (-1, 1)
            "score": self.score_head(g).squeeze(-1),                   # [B]
            "ownership": self.ownership_head(h),                       # [B, N, 3]
            "reply_logits": self.reply_node(h).squeeze(-1).masked_fill(node_mask == 0, _NEG_INF),
        }

    @torch.no_grad()
    def policy_value(self, batch: EncodedBatch):
        """Convenience for search: legal-masked policy probabilities and value."""
        out = self.forward(batch)
        logits = out["policy_logits"].masked_fill(~batch.legal, _NEG_INF)
        return F.softmax(logits, dim=1), out["value"]

    @torch.no_grad()
    def node_activations(self, batch: EncodedBatch) -> dict[str, torch.Tensor]:
        """Per-layer node activations [B, N, H] for the Substrate Invariance Test.

        Returns the input embedding, the output of each message-passing block, and the
        final-norm representation (which is exactly the per-node policy/ownership pre-logits).
        """
        node_mask = batch.node_mask
        acts: dict[str, torch.Tensor] = {}
        h = self.input(batch.x)
        acts["input"] = h
        for i, block in enumerate(self.blocks):
            h = block(h, batch.nbr_index, batch.nbr_mask, node_mask)
            acts[f"block{i}"] = h
        acts["final"] = self.final_norm(h)
        return acts

    @property
    def layer_names(self) -> list[str]:
        return ["input"] + [f"block{i}" for i in range(self.cfg.blocks)] + ["final"]


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

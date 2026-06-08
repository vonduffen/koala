"""Export a board graph, static features, and net weights to the binary formats the native
C++ engine (cpp/server.cpp) reads. This is the only glue the play/build path needs — the
self-play training bridge is not part of this (engine + play) release.

Binary formats mirror cpp/tgnet.hpp (weights) and cpp/server.cpp (graph/static).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .nn.encoding import PE_DIM, static_board
from .rules.gostate import Board

_MAGIC = 0x54474E31


def export_graph(board: Board, path: Path) -> None:
    edges = np.asarray(board.graph.edges, dtype="<i4")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<i", board.n))
        fh.write(struct.pack("<f", board.komi))
        fh.write(struct.pack("<i", len(edges)))
        fh.write(edges.tobytes())


def export_static(board: Board, path: Path) -> None:
    sb = static_board(board.graph)
    block = np.concatenate([sb.degree_onehot, sb.dist, sb.pe], axis=1).astype("<f4")
    assert block.shape[1] == 7 + 1 + PE_DIM
    with open(path, "wb") as fh:
        fh.write(struct.pack("<2i", block.shape[0], block.shape[1]))
        fh.write(block.tobytes())


def _tensor_order(net):
    sd = net.state_dict()
    order = ["input.0.weight", "input.0.bias", "input.2.weight", "input.2.bias"]
    for b in range(net.cfg.blocks):
        order += [f"blocks.{b}.norm.weight", f"blocks.{b}.norm.bias",
                  f"blocks.{b}.mlp.0.weight", f"blocks.{b}.mlp.0.bias",
                  f"blocks.{b}.mlp.2.weight", f"blocks.{b}.mlp.2.bias"]
    order += ["final_norm.weight", "final_norm.bias",
              "policy_node.weight", "policy_node.bias",
              "pass_head.weight", "pass_head.bias",
              "value_head.0.weight", "value_head.0.bias",
              "value_head.2.weight", "value_head.2.bias"]
    return [sd[k] for k in order]


def export_weights(net, path: Path) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4i", _MAGIC, net.cfg.in_dim, net.cfg.hidden, net.cfg.blocks))
        for t in _tensor_order(net):
            arr = t.detach().cpu().numpy().astype("<f4").ravel()
            fh.write(struct.pack("<i", arr.size))
            fh.write(arr.tobytes())

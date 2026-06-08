"""Tests for the encoding + GNN (ARCHITECTURE.md §5, Milestone 4 acceptance).

The headline acceptance test is masking invariance: padding a graph with extra nodes must not
change the real nodes' outputs (so padded nodes provably never leak into predictions).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from tilinggo.nn import encoding, model
from tilinggo.nn.encoding import FEATURE_DIM
from tilinggo.rules import Board
from tilinggo.tilings import periodic, uniform


def _state(graph, moves=(0, 1)):
    s = Board(graph).new_game()
    for m in moves:
        s = s.play(m)
    return s


def test_feature_dim_is_consistent():
    s = _state(periodic.rectangular(7, 7))
    batch = encoding.encode_states([s])
    assert batch.x.shape == (1, 49, FEATURE_DIM)
    assert FEATURE_DIM == 42


def test_laplacian_pe_drops_trivial_vector():
    bg = periodic.generate("hex", cells=60)
    pe = encoding.laplacian_pe(bg, dim=16)
    assert pe.shape == (bg.num_nodes, 16)
    # the kept eigenvectors are non-constant (the trivial constant vector was dropped)
    assert np.ptp(pe[:, 0]) > 1e-3
    # eigenvectors are (near) orthonormal columns
    gram = pe.T @ pe
    assert np.allclose(np.diag(gram), 1.0, atol=1e-3)


def test_encode_mixed_tilings_and_sizes():
    states = [_state(periodic.rectangular(9, 9)),
              _state(uniform.generate("snub_square", radius=4)),
              _state(periodic.triangular_hex(5))]
    batch = encoding.encode_states(states)
    n_max = max(s.board.n for s in states)
    assert batch.x.shape == (3, n_max, FEATURE_DIM)
    # node_mask marks exactly the real nodes of each board
    for bi, s in enumerate(states):
        assert int(batch.node_mask[bi].sum()) == s.board.n


def test_forward_shapes_and_legal_masking():
    states = [_state(periodic.rectangular(9, 9)), _state(periodic.generate("tri", cells=60))]
    batch = encoding.encode_states(states)
    net = model.TilingGoNet().eval()
    out = net(batch)
    n_max = batch.x.shape[1]
    assert out["policy_logits"].shape == (2, n_max + 1)
    assert out["value"].shape == (2,)
    assert out["ownership"].shape == (2, n_max, 3)
    probs, value = net.policy_value(batch)
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-5)
    assert probs[~batch.legal].max() < 1e-6      # illegal moves get ~0 probability
    assert value.abs().max() <= 1.0


def test_masking_invariance_padding_does_not_leak():
    # The SAME small board, encoded alone vs. padded inside a batch with a bigger board, must
    # produce identical outputs on its real nodes.
    torch.manual_seed(0)
    net = model.TilingGoNet().eval()
    small = _state(periodic.generate("hex", cells=50), moves=(0, 1, 2))
    big = _state(periodic.rectangular(13, 13))
    n = small.board.n

    out_alone = net(encoding.encode_states([small]))
    out_padded = net(encoding.encode_states([small, big]))

    # node policy logits, value, score, ownership on the real nodes must match
    assert torch.allclose(out_alone["policy_logits"][0, :n],
                          out_padded["policy_logits"][0, :n], atol=1e-4)
    assert torch.allclose(out_alone["value"][0], out_padded["value"][0], atol=1e-4)
    assert torch.allclose(out_alone["ownership"][0, :n],
                          out_padded["ownership"][0, :n], atol=1e-4)
    # the pass logit (from the global pool) lives at each batch's own N_max index
    n_alone = out_alone["policy_logits"].shape[1] - 1
    n_padded = out_padded["policy_logits"].shape[1] - 1
    assert torch.allclose(out_alone["policy_logits"][0, n_alone],
                          out_padded["policy_logits"][0, n_padded], atol=1e-4)


def test_forward_is_deterministic_in_eval():
    torch.manual_seed(1)
    net = model.TilingGoNet().eval()
    batch = encoding.encode_states([_state(periodic.rectangular(9, 9))])
    a = net(batch)["policy_logits"]
    b = net(batch)["policy_logits"]
    assert torch.equal(a, b)


def test_param_count_reasonable():
    # sub-million for H=96/L=8 — modest by AlphaZero standards, ample for these board sizes
    # (the prior Delta-Go champion was 188K). Sizes are TUNE in the spec.
    n = model.count_parameters(model.TilingGoNet())
    assert 300_000 < n < 5_000_000


def test_static_board_cache_survives_id_reuse():
    # Regression: static_board must key on graph *content*, not id() — else a reloaded graph can
    # collide with a GC'd graph's stale cache entry and mis-broadcast (shape mismatch).
    import gc
    from tilinggo.rules import Board
    net = model.TilingGoNet().eval()
    for rep in range(15):
        for cells in (40 + rep % 7, 55 + rep % 7):   # alternating, distinct sizes
            g = periodic.generate("square", cells=cells)
            s = Board(g).new_game().play(0).play(1)
            out = net.node_activations(encoding.encode_states([s]))
            assert out["final"].shape[1] == g.num_nodes
            del g, s, out
            gc.collect()


def test_forward_pass_benchmark(capsys):
    # §5.3 acceptance: a 64-batch on a ~150-node board should be fast (<25 ms on MPS).
    graph = periodic.rectangular(12, 13)  # 156 nodes
    state = _state(graph, moves=(0, 1, 2, 3))
    batch = encoding.encode_states([state] * 64)
    net = model.TilingGoNet().eval()

    def bench(device):
        b = batch.to(device)
        m = net.to(device)
        with torch.no_grad():
            for _ in range(3):  # warmup
                m(b)
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                m(b)
            if device == "mps":
                torch.mps.synchronize()
            return (time.perf_counter() - t0) / 10 * 1000  # ms/call

    cpu_ms = bench("cpu")
    msg = f"forward 64×156: CPU {cpu_ms:.1f} ms"
    if torch.backends.mps.is_available():
        msg += f" | MPS {bench('mps'):.1f} ms"
    with capsys.disabled():
        print("  " + msg)
    assert cpu_ms < 1000  # loose guard against accidental quadratic blowups

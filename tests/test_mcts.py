"""Tests for PUCT MCTS (ARCHITECTURE.md §6, Milestone 5 acceptance).

Acceptance (§9): the engine completes legal games on all tilings, and a 160-visit search beats
the raw (uniform) policy clearly — demonstrated here with a cheap score-heuristic evaluator so
the test is fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilinggo.rules import BLACK, Board
from tilinggo.search.evaluators import ScoreHeuristicEvaluator, UniformEvaluator
from tilinggo.search.mcts import MCTS, MCTSConfig
from tilinggo.tilings import periodic, uniform


def _play_full_game(board, mcts, temperature=0.0, cap=400):
    s = board.new_game()
    moves = 0
    while not s.is_terminal and moves < cap:
        root = mcts.run(s)
        s = s.play(mcts.select_move(root, temperature=temperature))  # play() re-checks legality
        moves += 1
    return s, moves


@pytest.mark.parametrize("graph", [
    periodic.generate("tri", cells=40, seed=1),
    periodic.generate("hex", cells=45, seed=1),
    uniform.generate("snub_square", radius=3),
    periodic.triangular_hex(4),
])
def test_completes_legal_games_on_tilings(graph):
    board = Board(graph, komi=0.5)
    mcts = MCTS(ScoreHeuristicEvaluator(), MCTSConfig(num_simulations=32, eval_batch=8),
                rng=np.random.default_rng(0))
    s, moves = _play_full_game(board, mcts)
    assert s.is_terminal           # reached a terminal position (two passes or move cap)
    assert moves <= board.max_moves


def test_visit_policy_is_valid_distribution():
    board = Board(periodic.generate("tri", cells=40), komi=0.5)
    mcts = MCTS(ScoreHeuristicEvaluator(), MCTSConfig(num_simulations=64),
                rng=np.random.default_rng(0))
    root = mcts.run(board.new_game())
    pi = mcts.visit_policy(root)
    assert abs(pi.sum() - 1.0) < 1e-9
    legal = board.new_game().legal_moves()
    assert np.all(pi[~legal] == 0.0)   # no probability on illegal moves


def test_search_beats_raw_policy():
    # §9 acceptance: 160-visit search must beat the raw policy. Raw policy here = uniform, i.e.
    # random legal play; search uses the same uniform priors but a score-aware value head.
    board = Board(periodic.generate("tri", cells=40), komi=0.5)

    def random_move(state, rng):
        return int(rng.choice(np.flatnonzero(state.legal_moves())))

    def one_game(mcts_is_black, seed):
        rng = np.random.default_rng(seed)
        mcts = MCTS(ScoreHeuristicEvaluator(),
                    MCTSConfig(num_simulations=160, eval_batch=16), rng=rng)
        s = board.new_game()
        moves = 0
        while not s.is_terminal and moves < 400:
            if (s.to_move == BLACK) == mcts_is_black:
                root = mcts.run(s)
                mv = mcts.select_move(root, temperature=0.3 if moves < 6 else 0.0)
            else:
                mv = random_move(s, rng)
            s = s.play(mv)
            moves += 1
        mcts_won = (s.winner() == BLACK) == mcts_is_black
        return mcts_won

    games = 16
    wins = sum(one_game(g % 2 == 0, 100 + g) for g in range(games))  # alternate colours
    assert wins / games >= 0.70, f"search won only {wins}/{games}"


def test_uniform_evaluator_search_runs():
    board = Board(periodic.generate("hex", cells=40), komi=0.5)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(num_simulations=48), rng=np.random.default_rng(0))
    root = mcts.run(board.new_game())
    assert root.N.sum() > 0
    assert board.new_game().legal_moves()[mcts.select_move(root, temperature=0.0)]


def test_net_evaluator_drives_search():
    # The untrained net is a valid evaluator; search should run and return a legal move.
    torch = pytest.importorskip("torch")
    from tilinggo.nn.model import TilingGoNet
    from tilinggo.search.evaluators import NetEvaluator

    torch.manual_seed(0)
    board = Board(periodic.generate("tri", cells=40), komi=0.5)
    mcts = MCTS(NetEvaluator(TilingGoNet()), MCTSConfig(num_simulations=32, eval_batch=16),
                rng=np.random.default_rng(0))
    root = mcts.run(board.new_game())
    move = mcts.select_move(root, temperature=0.0)
    assert board.new_game().legal_moves()[move]


def test_tree_reuse_carries_statistics():
    board = Board(periodic.generate("tri", cells=40), komi=0.5)
    mcts = MCTS(ScoreHeuristicEvaluator(), MCTSConfig(num_simulations=64),
                rng=np.random.default_rng(0))
    s0 = board.new_game()
    root = mcts.run(s0)
    move = mcts.select_move(root, temperature=0.0)
    child = root.children.get(move)
    s1 = s0.play(move)
    # reusing the child subtree should not error and should keep its prior visits
    visits_before = child.N.sum() if (child and child.expanded) else 0
    root2 = mcts.run(s1, reuse=child)
    assert root2.N.sum() >= visits_before

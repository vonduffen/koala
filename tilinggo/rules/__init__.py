"""Rules engine: exact, well-tested Go on an arbitrary BoardGraph (geometry-blind)."""

from .gostate import (
    BLACK,
    EMPTY,
    WHITE,
    Board,
    GoState,
    IllegalMove,
    opponent,
)

__all__ = ["EMPTY", "BLACK", "WHITE", "opponent", "Board", "GoState", "IllegalMove"]

"""Network input encoding (framework-independent)."""

import numpy as np

_SHIFTS = np.arange(64, dtype=np.uint64)


def board_to_planes(
    board_black: int, board_white: int, is_black: bool, legal: int
) -> np.ndarray:
    """Convert bitboards to 4×8×8 planes: my, opp, legal, phase."""
    my_bits = board_black if is_black else board_white
    opp_bits = board_white if is_black else board_black
    bits = np.array([my_bits, opp_bits, legal], dtype=np.uint64)
    occupied = (int(board_black) | int(board_white)).bit_count() / 64.0
    spatial = (
        ((bits[:, None] >> _SHIFTS) & np.uint64(1)).astype(np.float32).reshape(3, 8, 8)
    )
    phase_plane = np.full((1, 8, 8), occupied, dtype=np.float32)
    return np.concatenate((spatial, phase_plane), axis=0)

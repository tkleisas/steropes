"""Deck model: physical dimensions, fiducial markers, overhead camera.

The deck is the flat work surface the device under test (DUT) sits on. Four
ArUco markers at known deck coordinates let the vision pipeline register any
overhead camera frame to deck millimetres via a planar homography.

Deck coordinate convention: origin at the deck's lower-left corner, +X right,
+Y up, units millimetres. Image convention: pixel (0, 0) is the top-left of
the frame and deck +Y points image-up (pixel row 0 = deck +Y max) — the
convention the host stack's cameras use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- deck geometry -----------------------------------------------------------
DECK_WIDTH_MM = 400.0
DECK_DEPTH_MM = 300.0

# ArUco fiducials: marker id -> (x, y) of the marker centre in deck mm.
# DICT_4X4_50, ids 1-4, one marker-width in from each corner of the
# 400x300 mm deck. This mirrors the host stack's defaults exactly
# (androidtester HarnessConfig.deck_marker_map(): DICT_4X4_50, marker_size_mm
# 20, inset = marker_size_mm from each corner of the 400x300 deck) so an
# unmodified client calibrates against rendered frames with its stock
# configuration.
ARUCO_DICTIONARY = "DICT_4X4_50"
DECK_MARKERS: dict[int, tuple[float, float]] = {
    1: (20.0, 20.0),
    2: (380.0, 20.0),
    3: (380.0, 280.0),
    4: (20.0, 280.0),
}

# Marker tile as printed: the ArUco code plus a white quiet zone on each side.
MARKER_CODE_MM = 20.0
MARKER_QUIET_MM = 4.0
MARKER_TILE_MM = MARKER_CODE_MM + 2.0 * MARKER_QUIET_MM  # 28 mm
MARKER_TEX_PX = 280  # texture resolution per marker tile (10 px/mm)


# --- overhead camera ---------------------------------------------------------
@dataclass(frozen=True)
class OverheadCamera:
    """Straight-down camera above the deck centre.

    ``px_per_mm`` is the ground-sample scale at the deck plane; the vertical
    field of view is derived from it and the frame height.
    """

    width_px: int = 1800
    height_px: int = 1400
    height_mm: float = 500.0
    px_per_mm: float = 4.0

    @property
    def fovy_deg(self) -> float:
        """Vertical half-fov such that the frame spans height_px / px_per_mm mm."""
        half_span_mm = (self.height_px / 2.0) / self.px_per_mm
        return 2.0 * math.degrees(math.atan(half_span_mm / self.height_mm))


OVERHEAD_CAMERA = OverheadCamera()

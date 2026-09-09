"""Kinematic CoreXY-style gantry: toolhead geometry, motion, tool camera.

M2 fidelity: the gantry is kinematic — moves are interpolated in deck XY at
fixed dt with no dynamics, and collision checking reduces to contact
detection between the toolhead geoms and the scene (MuJoCo computes the
contacts; nothing is asserted by the check itself).

The toolhead carries a compliant finger tool (taps the DUT screen; Z
actuation and contact physics are M3, see :mod:`steropes.touch`) and a
downward camera mounted beside the finger, so toolcam frames render
from the toolhead's real pose. At cruise height the finger tip clears the
flat terminal body; the terminal's raised back strip (printer hump, see
:mod:`steropes.scene`) is the collision hazard the guard exists to catch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import deck as deck_const
from . import touch

# --- toolhead geometry (mm; body origin at the carriage centre) ---------------
CARRIAGE_HALF_MM = (20.0, 20.0, 10.0)
CARRIAGE_Z_MM = 50.0                       # carriage centre above the deck
FINGER_OFFSET_MM = (12.0, 0.0, -25.0)      # tip 10 mm above deck at cruise
FINGER_RADIUS_MM = 4.0
FINGER_HALF_LEN_MM = 15.0
TOOLCAM_OFFSET_MM = (-12.0, 0.0, -11.0)    # beside the finger, below carriage

#: Height of the terminal's raised back strip — reaches the toolhead envelope.
TERMINAL_RISER_HEIGHT_MM = 60.0

# --- motion ---------------------------------------------------------------------
DEFAULT_SPEED_MM_S = 150.0
DEFAULT_DT_S = 0.02


# --- toolhead camera -------------------------------------------------------------
@dataclass(frozen=True)
class ToolCamera:
    """Straight-down camera bolted to the toolhead.

    ``height_mm`` is its height above the deck plane (the toolhead only moves
    in XY, so it is constant); the pose itself follows the toolhead body.
    """

    width_px: int = 400
    height_px: int = 300
    height_mm: float = CARRIAGE_Z_MM + TOOLCAM_OFFSET_MM[2]  # 39 mm
    px_per_mm: float = 4.0

    @property
    def fovy_deg(self) -> float:
        """Vertical half-fov such that the frame spans height_px / px_per_mm mm."""
        half_span_mm = (self.height_px / 2.0) / self.px_per_mm
        return 2.0 * math.degrees(math.atan(half_span_mm / self.height_mm))


TOOL_CAMERA = ToolCamera()


# --- helpers ------------------------------------------------------------------------

def clamp_to_deck(x: float, y: float) -> tuple[float, float]:
    """Clamp a deck-XY target to the gantry travel limits (mm)."""
    return (min(max(x, 0.0), deck_const.DECK_WIDTH_MM),
            min(max(y, 0.0), deck_const.DECK_DEPTH_MM))


def check_deck_limits(x: float, y: float) -> None:
    """Raise ValueError when (x, y) lies outside the gantry travel."""
    if not (0.0 <= x <= deck_const.DECK_WIDTH_MM
            and 0.0 <= y <= deck_const.DECK_DEPTH_MM):
        raise ValueError(
            f"toolhead target ({x}, {y}) mm outside travel limits "
            f"(0..{deck_const.DECK_WIDTH_MM:.0f}, 0..{deck_const.DECK_DEPTH_MM:.0f}) mm")


def interpolate(start: tuple[float, float], goal: tuple[float, float],
                speed_mm_s: float, dt_s: float) -> list[tuple[float, float]]:
    """Straight-line waypoints from ``start`` to ``goal`` at fixed dt.

    Deterministic; the first point is one step along and the last is exactly
    ``goal`` (one point minimum, so a zero-length move still re-evaluates the
    current pose).
    """
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    n = max(1, math.ceil(math.hypot(dx, dy) / (speed_mm_s * dt_s)))
    return [(start[0] + dx * i / n, start[1] + dy * i / n)
            for i in range(1, n + 1)]


# --- MJCF ----------------------------------------------------------------------------

def toolhead_xml() -> str:
    """MJCF fragment: toolhead body with X/Y slide joints, finger, toolcam.

    The finger is a separate plunger body on a Z slide joint (``tool_z``,
    positive = down) with spring-damper compliance — a pogo-pin stand-in.
    Its joint spring reference is what a tap actuates; see
    :meth:`steropes.scene.DeckScene.tap_finger`.
    """
    chx, chy, chz = CARRIAGE_HALF_MM
    fx, fy, fz = FINGER_OFFSET_MM
    cx, cy, cz = TOOLCAM_OFFSET_MM
    return f"""
    <body name="toolhead" pos="0 0 {CARRIAGE_Z_MM / 1000:.4f}">
      <joint name="tool_x" type="slide" axis="1 0 0"
             range="0 {deck_const.DECK_WIDTH_MM / 1000:.3f}"/>
      <joint name="tool_y" type="slide" axis="0 1 0"
             range="0 {deck_const.DECK_DEPTH_MM / 1000:.3f}"/>
      <geom name="tool_carriage" type="box"
            size="{chx / 1000:.4f} {chy / 1000:.4f} {chz / 1000:.4f}"
            rgba="0.75 0.45 0.15 1"/>
      <camera name="toolcam"
              pos="{cx / 1000:.4f} {cy / 1000:.4f} {cz / 1000:.4f}"
              xyaxes="1 0 0 0 1 0" fovy="{TOOL_CAMERA.fovy_deg:.4f}"/>
      <body name="tool_finger_body" pos="{fx / 1000:.4f} {fy / 1000:.4f} {fz / 1000:.4f}">
        <joint name="tool_z" type="slide" axis="0 0 -1" range="0 0.03"
               springref="0" stiffness="{touch.FINGER_SPRING_N_PER_M}"
               damping="{touch.FINGER_DAMPING_NS_PER_M}"/>
        <geom name="tool_finger" type="cylinder"
              size="{FINGER_RADIUS_MM / 1000:.4f} {FINGER_HALF_LEN_MM / 1000:.4f}"
              density="{touch.FINGER_DENSITY_KG_M3}" rgba="0.85 0.85 0.88 1"/>
      </body>
    </body>
"""

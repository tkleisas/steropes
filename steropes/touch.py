"""M3: physical screen taps — compliant finger, contact-derived registration.

The M1 tier fed taps to the DUT model directly. Here the finger is a real
body on a Z slide joint with spring-damper compliance (a pogo-pin style
plunger): a tap ramps the joint's spring reference downward at a fixed speed,
the spring presses the tip into the screen, and the contact point reported by
the physics engine — not the commanded coordinate — decides which keypad cell
the DUT receives. Peak contact force per tap is read back with
``mj_contactForce``.

Force budget for a capacitive screen tap: 1.5-6 N. With the shipped spring
(1 N/mm) and ~3.5 mm of overtravel, a tap settles at ~3.5 N.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- finger compliance (pogo-pin plunger) -------------------------------------
FINGER_MASS_KG = 0.03              # plunger assembly
FINGER_SPRING_N_PER_M = 1000.0     # 1 N/mm: 3.5 mm overtravel -> ~3.5 N
FINGER_DAMPING_NS_PER_M = 8.0      # ~0.7x critical for the 30 g plunger
FINGER_DENSITY_KG_M3 = 20000.0     # geom density yielding FINGER_MASS_KG

# --- tap motion ---------------------------------------------------------------
DEFAULT_DESCEND_MM = 12.0          # cruise->screen is 8.5 mm; ~3.5 mm overtravel
TAP_SPEED_MM_S = 100.0             # springref ramp speed (deterministic, fixed dt)
TAP_HOLD_S = 0.15                  # settle time at full overtravel
TAP_SETTLE_S = 0.10                # settle time after retract

# --- force budget --------------------------------------------------------------
FORCE_MIN_N = 1.5
FORCE_MAX_N = 6.0


@dataclass
class TapOutcome:
    """What one physical tap actually did (all derived from contacts)."""

    target_mm: tuple[float, float]             # commanded contact point
    contact_mm: tuple[float, float] | None     # contact point at peak force
    geom: str | None                           # geom hit at peak force
    cell: int | None                           # keypad cell from contact_mm
    peak_force_n: float


@dataclass
class SwipeOutcome:
    """What one physical swipe actually did (all derived from contacts)."""

    start_mm: tuple[float, float]              # commanded drag start
    end_mm: tuple[float, float]                # commanded drag end
    contacts_mm: list[tuple[float, float]]     # screen contacts along the drag
    geom: str | None                           # geom hit at peak force
    peak_force_n: float


# --- screen grid geometry (deck mm) ---------------------------------------------
# Screen polygon order is TL TR BR BL in the deck image convention (+Y up),
# so keypad row 0 is the top (max Y) edge.

def cell_at(polygon_mm: list[tuple[float, float]], cols: int, rows: int,
            x: float, y: float) -> int | None:
    """Keypad cell containing deck point (x, y), or None outside the screen."""
    xs = [p[0] for p in polygon_mm]
    ys = [p[1] for p in polygon_mm]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return None
    c = min(int((x - x0) / (x1 - x0) * cols), cols - 1)
    r = min(int((y1 - y) / (y1 - y0) * rows), rows - 1)
    return r * cols + c


def cell_center(polygon_mm: list[tuple[float, float]], cols: int, rows: int,
                cell: int) -> tuple[float, float]:
    """Deck-mm centre of a keypad cell (row-major index)."""
    xs = [p[0] for p in polygon_mm]
    ys = [p[1] for p in polygon_mm]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    r, c = divmod(cell, cols)
    return (x0 + (c + 0.5) * (x1 - x0) / cols,
            y1 - (r + 0.5) * (y1 - y0) / rows)

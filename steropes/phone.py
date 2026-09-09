"""Device-under-test model: an Android phone (lock screen, PIN pad, launcher).

The model mirrors what the host stack's ``devices/*.yaml`` phone profiles
assume: a lock screen with a clock, a swipe-up gesture to a fixed-order 3x4
PIN pad at the standard phone positions, and a launcher with text-labeled app
icons that open named app screens.

The canvas renders at 720x1280 — the exact size the host's ``rectify_screen``
warps to by default — with ``cv2.putText`` glyphs at the scales its
template-matching OCR renders, so rectified screens stay legible to the
unmodified client.

Interactions reach the model as screen-relative fractions (origin top-left,
matching the device profile's ``launcher`` points): taps and swipes are
routed from physical finger contacts (:meth:`steropes.scene.DeckScene.
tap_finger` / ``swipe_finger``) via :func:`mm_to_screen_fraction`, and side
buttons via the modeled button pads (see :class:`steropes.scene.ButtonPad`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SCREEN_W_PX = 720
SCREEN_H_PX = 1280

BG_COLOR = (14, 14, 18)        # BGR, near-black
TEXT_COLOR = (235, 235, 235)   # near-white text
ACCENT_COLOR = (120, 190, 255)
ICON_COLOR = (52, 58, 68)

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Fixed-order 3x4 PIN pad, standard phone layout, as screen-relative
#: fractions (origin top-left). These match the ``pin_*`` launcher points of
#: the host stack's example phone profile, so its blind PIN entry lands on
#: the rendered digits.
PIN_PAD_FRACTIONS: dict[str, tuple[float, float]] = {
    "1": (0.25, 0.45), "2": (0.50, 0.45), "3": (0.75, 0.45),
    "4": (0.25, 0.58), "5": (0.50, 0.58), "6": (0.75, 0.58),
    "7": (0.25, 0.71), "8": (0.50, 0.71), "9": (0.75, 0.71),
    "0": (0.50, 0.84),
}

#: Hit half-extents (screen fractions) around a PIN key / launcher icon.
_KEY_HX, _KEY_HY = 0.125, 0.065
_ICON_HX, _ICON_HY = 0.16, 0.09

#: Minimum upward travel (screen fractions) that counts as a swipe-up unlock.
_SWIPE_UP_MIN = 0.3

#: Default launcher when the profile declares no apps.
DEFAULT_APPS: list[tuple[str, str, float, float]] = [
    ("settings", "Settings", 0.25, 0.40),
    ("camera", "Camera", 0.75, 0.40),
]


# --- screen-fraction <-> deck-mm mapping (screen is an axis-aligned box) -----

def screen_fraction_to_mm(polygon_mm: list[tuple[float, float]],
                          fx: float, fy: float) -> tuple[float, float]:
    """Map screen fractions (origin top-left) to deck mm over the polygon box.

    The phone profile's polygon orders corners TL TR BR BL with TL at the
    (min x, min y) deck corner, so +fy runs toward +Y on the deck.
    """
    xs = [p[0] for p in polygon_mm]
    ys = [p[1] for p in polygon_mm]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))


def mm_to_screen_fraction(polygon_mm: list[tuple[float, float]],
                          x: float, y: float) -> tuple[float, float]:
    """Inverse of :func:`screen_fraction_to_mm`; not clamped to 0..1."""
    xs = [p[0] for p in polygon_mm]
    ys = [p[1] for p in polygon_mm]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return ((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))


@dataclass
class PhoneTap:
    """Outcome of one screen tap routed to the phone."""

    digit: str | None = None    # PIN key hit, if any
    app: str | None = None      # launcher icon opened, if any


class PhoneDUT:
    """Stateful phone model: sleep -> locked -> pin -> launcher -> app.

    ``apps`` is a list of (name, label, x, y) launcher entries in screen
    fractions; the default mirrors the host's example phone (settings at
    (0.25, 0.40), camera at (0.75, 0.40)).
    """

    def __init__(self, pin: str,
                 apps: list[tuple[str, str, float, float]] | None = None
                 ) -> None:
        if not pin or not pin.isdigit():
            raise ValueError(f"PIN must be a non-empty digit string, got {pin!r}")
        self.pin = pin
        self.apps = apps if apps is not None else list(DEFAULT_APPS)
        self.state = "sleep"  # sleep | locked | pin | launcher | app
        self.current_app: str | None = None
        self.failed_attempts = 0
        self._entered = ""

    # -- interaction ----------------------------------------------------------

    def press_button(self, name: str) -> None:
        """Press a physical button by profile name (power/volume_*)."""
        if name == "power":
            if self.state == "sleep":
                self.state = "locked"
            elif self.state in ("locked", "pin"):
                self.state = "sleep"
                self._entered = ""
            # launcher/app: power toggling the screen off mid-session is out
            # of scope for the twin.
        elif not name.startswith("volume"):
            raise ValueError(f"unknown button {name!r}")

    def swipe(self, fx1: float, fy1: float, fx2: float, fy2: float) -> None:
        """Gesture from fraction (fx1, fy1) to (fx2, fy2); swipe-up unlocks."""
        if self.state == "locked" and fy1 - fy2 >= _SWIPE_UP_MIN:
            self.state = "pin"
            self._entered = ""

    def tap(self, fx: float, fy: float) -> PhoneTap:
        """Tap at screen fraction (fx, fy). Off-target taps are ignored."""
        if self.state == "pin":
            for digit, (dx, dy) in PIN_PAD_FRACTIONS.items():
                if abs(fx - dx) <= _KEY_HX and abs(fy - dy) <= _KEY_HY:
                    self._entered += digit
                    if len(self._entered) == len(self.pin):
                        self._submit()
                    return PhoneTap(digit=digit)
        elif self.state == "launcher":
            for name, _label, ax, ay in self.apps:
                if abs(fx - ax) <= _ICON_HX and abs(fy - ay) <= _ICON_HY:
                    self.state = "app"
                    self.current_app = name
                    return PhoneTap(app=name)
        return PhoneTap()

    def _submit(self) -> None:
        if self._entered == self.pin:
            self.state = "launcher"
        else:
            self.failed_attempts += 1  # wrong PIN: stay on the pad, cleared
        self._entered = ""

    # -- rendering --------------------------------------------------------------

    def render_screen(self) -> np.ndarray:
        """Current screen as an upright BGR canvas (SCREEN_H_PX x SCREEN_W_PX)."""
        img = np.full((SCREEN_H_PX, SCREEN_W_PX, 3), BG_COLOR, np.uint8)
        if self.state == "sleep":
            pass  # display off
        elif self.state == "locked":
            self._draw_centered(img, "12:00", TEXT_COLOR, 0.28,
                                scale=3.0, thickness=4)
            self._draw_centered(img, "Locked", ACCENT_COLOR, 0.55,
                                scale=2.0, thickness=3)
            self._draw_centered(img, "Swipe up", TEXT_COLOR, 0.80,
                                scale=1.2, thickness=2)
        elif self.state == "pin":
            self._draw_centered(img, "Enter PIN", TEXT_COLOR, 0.12,
                                scale=2.0, thickness=3)
            self._draw_centered(img, "*" * len(self._entered) or "-",
                                ACCENT_COLOR, 0.28, scale=2.0, thickness=3)
            for digit, (fx, fy) in PIN_PAD_FRACTIONS.items():
                self._draw_centered(img, digit, TEXT_COLOR, fy, fx=fx,
                                    scale=2.5, thickness=4)
        elif self.state == "launcher":
            self._draw_centered(img, "Home", TEXT_COLOR, 0.10,
                                scale=2.0, thickness=3)
            for _name, label, fx, fy in self.apps:
                cx, cy = int(fx * SCREEN_W_PX), int(fy * SCREEN_H_PX)
                cv2.rectangle(img, (cx - 70, cy - 70), (cx + 70, cy + 70),
                              ICON_COLOR, -1)
                cv2.rectangle(img, (cx - 70, cy - 70), (cx + 70, cy + 70),
                              TEXT_COLOR, 2)
                self._draw_centered(img, label, TEXT_COLOR, fy + 0.10, fx=fx,
                                    scale=2.0, thickness=3)
        elif self.state == "app":
            label = self.current_app or ""
            self._draw_centered(img, label.capitalize(), TEXT_COLOR, 0.10,
                                scale=2.0, thickness=3)
            self._draw_centered(img, label.capitalize(), ACCENT_COLOR, 0.50,
                                scale=3.0, thickness=4)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unknown state {self.state!r}")
        return img

    def render_for_deck(self) -> np.ndarray:
        """Screen canvas as the deck texture needs it (vertically flipped).

        The deck texture maps canvas row 0 to the screen's +Y edge, but the
        shared device profile puts the UI's top-left at the (min x, min y)
        deck corner — so the UI is flipped vertically for the texture and
        comes back upright through the client's rectification.
        """
        return np.flipud(self.render_screen())

    @staticmethod
    def _draw_centered(img: np.ndarray, text: str, color: tuple[int, int, int],
                       fy: float, scale: float, thickness: int,
                       fx: float = 0.5) -> None:
        """Draw text centered at screen fraction (fx, fy)."""
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        cx, cy = int(fx * img.shape[1]), int(fy * img.shape[0])
        org = (cx - tw // 2, cy + th // 2 - baseline // 2)
        cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

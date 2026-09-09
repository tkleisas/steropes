"""Device-under-test model: a countertop payment terminal with a scrambled
on-screen PIN keypad.

The model is deliberately small: a seeded RNG lays out a 3x4 keypad with the
digits 0-9 shuffled plus two blank keys, and a tiny screen state machine
(idle -> keypad -> approved/declined) renders a framebuffer the way a real
POS screen would look to a camera — light glyphs on a dark background.

Taps reach the model two ways: as direct model calls
(:meth:`DeviceUnderTest.tap`) for the fast M1 tier, or as physical finger
contacts routed through :meth:`steropes.scene.DeckScene.tap_finger` (M3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

SCREEN_W_PX = 320
SCREEN_H_PX = 240

BG_COLOR = (24, 24, 28)        # BGR, near-black
TEXT_COLOR = (225, 225, 225)   # light grey digits
APPROVED_COLOR = (90, 220, 90)
DECLINED_COLOR = (110, 110, 235)
KEY_LINE_COLOR = (70, 70, 78)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def scrambled_layout(seed: int, cols: int = 3, rows: int = 4) -> list[str]:
    """Seeded keypad layout: digits 0-9 shuffled, padded with blanks.

    Returns ``cols * rows`` entries in row-major order (index ``r * cols + c``);
    exactly two entries are empty strings.
    """
    n = cols * rows
    rng = random.Random(seed)
    cells = list("0123456789") + [""] * (n - 10)
    rng.shuffle(cells)
    return cells


@dataclass
class TapResult:
    """Outcome of tapping one key."""

    digit: str
    accepted: bool


class DeviceUnderTest:
    """Stateful terminal model: keypad layout, PIN check, screen rendering."""

    def __init__(self, pin: str, seed: int, cols: int = 3, rows: int = 4) -> None:
        if not pin or not pin.isdigit():
            raise ValueError(f"PIN must be a non-empty digit string, got {pin!r}")
        self.pin = pin
        self.cols = cols
        self.rows = rows
        self.layout = scrambled_layout(seed, cols, rows)
        self.state = "idle"  # idle | keypad | approved | declined
        self._entered = ""

    # -- interaction ----------------------------------------------------------

    def show_keypad(self) -> None:
        """Transition idle -> keypad (what a 'pay with PIN' amount entry does)."""
        if self.state != "idle":
            raise RuntimeError(f"cannot show keypad from state {self.state!r}")
        self.state = "keypad"
        self._entered = ""

    def tap(self, cell: int) -> TapResult:
        """Tap one keypad cell (model-level stand-in for a physical key press).

        Blank cells are ignored, mirroring a real terminal's dead keys.
        """
        if self.state != "keypad":
            raise RuntimeError(f"cannot tap in state {self.state!r}")
        digit = self.layout[cell]
        if digit:
            self._entered += digit
        return TapResult(digit=digit, accepted=bool(digit))

    def submit(self) -> str:
        """Submit the entered PIN. Returns 'approved' or 'declined'."""
        if self.state != "keypad":
            raise RuntimeError(f"cannot submit in state {self.state!r}")
        self.state = "approved" if self._entered == self.pin else "declined"
        return self.state

    def enter_pin(self, cells: list[int]) -> str:
        """Convenience: tap a sequence of cells then submit."""
        for cell in cells:
            self.tap(cell)
        return self.submit()

    # -- rendering --------------------------------------------------------------

    def render_screen(self) -> np.ndarray:
        """Current screen as a BGR framebuffer (SCREEN_H_PX x SCREEN_W_PX)."""
        img = np.full((SCREEN_H_PX, SCREEN_W_PX, 3), BG_COLOR, np.uint8)
        if self.state == "idle":
            self._draw_centered(img, "READY", TEXT_COLOR, scale=1.2, thickness=2)
        elif self.state == "keypad":
            self._draw_keypad(img)
        elif self.state == "approved":
            self._draw_centered(img, "APPROVED", APPROVED_COLOR, scale=1.1, thickness=3)
        elif self.state == "declined":
            self._draw_centered(img, "DECLINED", DECLINED_COLOR, scale=1.1, thickness=3)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unknown state {self.state!r}")
        return img

    def _draw_keypad(self, img: np.ndarray) -> None:
        cw, ch = SCREEN_W_PX // self.cols, SCREEN_H_PX // self.rows
        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = c * cw, r * ch
                cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch), KEY_LINE_COLOR, 1)
                digit = self.layout[r * self.cols + c]
                if digit:
                    self._draw_centered_in(img, digit, TEXT_COLOR,
                                           (x0, y0, cw, ch), scale=1.6, thickness=3)

    @staticmethod
    def _draw_centered(img: np.ndarray, text: str, color: tuple[int, int, int],
                       scale: float, thickness: int) -> None:
        DeviceUnderTest._draw_centered_in(
            img, text, color, (0, 0, img.shape[1], img.shape[0]), scale, thickness)

    @staticmethod
    def _draw_centered_in(img: np.ndarray, text: str, color: tuple[int, int, int],
                          box: tuple[int, int, int, int],
                          scale: float, thickness: int) -> None:
        x0, y0, w, h = box
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        org = (x0 + (w - tw) // 2, y0 + (h + th) // 2 - baseline // 2)
        cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

"""Unit tests for the vision primitives (no MuJoCo required)."""
import numpy as np
import pytest

from steropes import vision
from steropes.vision import KeypadReading


def test_solve_homography_roundtrip() -> None:
    src = np.array([[40.0, 40.0], [360.0, 40.0], [360.0, 260.0], [40.0, 260.0]])
    dst = src * 2.0 + 40.0  # affine map the DLT must recover exactly
    h = vision.solve_homography(src, dst)
    for p, q in zip(src, dst):
        u, v = vision._apply(h, *p)
        assert u == pytest.approx(q[0], abs=1e-6)
        assert v == pytest.approx(q[1], abs=1e-6)


def test_solve_homography_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        vision.solve_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_plan_pin_taps_maps_digits_to_cells() -> None:
    readings = [KeypadReading(cell=0, digit="5", confidence=0.9),
                KeypadReading(cell=1, digit="8", confidence=0.9),
                KeypadReading(cell=2, digit="2", confidence=0.9),
                KeypadReading(cell=3, digit="0", confidence=0.9),
                KeypadReading(cell=4, digit="", confidence=0.1)]
    assert vision.plan_pin_taps(readings, "5820") == [0, 1, 2, 3]


def test_plan_pin_taps_rejects_missing_digit() -> None:
    readings = [KeypadReading(cell=0, digit="5", confidence=0.9)]
    with pytest.raises(ValueError, match="not found"):
        vision.plan_pin_taps(readings, "51")


def test_plan_pin_taps_rejects_duplicate_digit() -> None:
    readings = [KeypadReading(cell=0, digit="5", confidence=0.9),
                KeypadReading(cell=1, digit="5", confidence=0.8)]
    with pytest.raises(ValueError, match="more than once"):
        vision.plan_pin_taps(readings, "5")

"""Unit tests for the Android-phone DUT model (no MuJoCo required)."""
import numpy as np
import pytest

from steropes import vision
from steropes.phone import (PIN_PAD_FRACTIONS, PhoneDUT,
                            mm_to_screen_fraction, screen_fraction_to_mm)

POLY = [(63.0, 84.0), (128.5, 84.0), (128.5, 221.0), (63.0, 221.0)]


def test_starts_asleep_with_dark_screen() -> None:
    dut = PhoneDUT(pin="1337")
    assert dut.state == "sleep"
    assert dut.render_screen().max() <= 30  # display off: near-black


def test_power_wakes_and_sleeps() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    assert dut.state == "locked"
    dut.press_button("power")
    assert dut.state == "sleep"


def test_unknown_button_rejected() -> None:
    with pytest.raises(ValueError, match="unknown button"):
        PhoneDUT(pin="1337").press_button("mute")


def test_volume_buttons_are_inert_noops() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("volume_up")
    assert dut.state == "sleep"


def test_swipe_up_unlocks_to_pin_pad() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    assert dut.state == "pin"


def test_short_or_downward_swipe_does_not_unlock() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    dut.swipe(0.5, 0.5, 0.5, 0.4)   # too short
    assert dut.state == "locked"
    dut.swipe(0.5, 0.2, 0.5, 0.9)   # downward
    assert dut.state == "locked"


def test_swipe_ignored_while_sleeping() -> None:
    dut = PhoneDUT(pin="1337")
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    assert dut.state == "sleep"


def _enter_pin(dut: PhoneDUT, pin: str) -> None:
    for digit in pin:
        fx, fy = PIN_PAD_FRACTIONS[digit]
        assert dut.tap(fx, fy).digit == digit


def test_correct_pin_reaches_launcher() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    _enter_pin(dut, "1337")
    assert dut.state == "launcher"


def test_wrong_pin_stays_on_pad_and_counts() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    _enter_pin(dut, "0000")
    assert dut.state == "pin"
    assert dut.failed_attempts == 1
    _enter_pin(dut, "1337")  # pad was cleared; retry works
    assert dut.state == "launcher"


def test_tap_between_keys_is_ignored() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    assert dut.tap(0.5, 0.35).digit is None  # above the top key row
    assert dut.state == "pin"


def test_launcher_icon_opens_app() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    _enter_pin(dut, "1337")
    assert dut.tap(0.25, 0.40).app == "settings"
    assert dut.state == "app" and dut.current_app == "settings"


def test_rendered_screens_are_template_legible() -> None:
    # Same template recipe as the host's assert_text fallback (scale 2.0).
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    assert vision.text_score(dut.render_screen(), "Locked") > 0.8
    dut.swipe(0.5, 0.9, 0.5, 0.2)
    assert vision.text_score(dut.render_screen(), "Enter PIN") > 0.8
    _enter_pin(dut, "1337")
    launcher = dut.render_screen()
    assert vision.text_score(launcher, "Home") > 0.8
    assert vision.text_score(launcher, "Settings") > 0.8
    dut.tap(0.25, 0.40)
    assert vision.text_score(dut.render_screen(), "Settings") > 0.8


def test_render_for_deck_is_vertical_flip() -> None:
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    assert np.array_equal(dut.render_for_deck(), np.flipud(dut.render_screen()))


def test_fraction_mm_mapping_round_trips() -> None:
    for fx, fy in [(0.0, 0.0), (1.0, 1.0), (0.25, 0.45), (0.5, 0.84)]:
        x, y = screen_fraction_to_mm(POLY, fx, fy)
        back = mm_to_screen_fraction(POLY, x, y)
        assert back == pytest.approx((fx, fy), abs=1e-9)
    # fraction (0, 0) is the (min x, min y) deck corner, as the host's
    # example phone profile declares it
    assert screen_fraction_to_mm(POLY, 0.0, 0.0) == (63.0, 84.0)


def test_pin_pad_layout_matches_host_profile() -> None:
    # devices/example_phone.yaml launcher pin_* points, verbatim
    expected = {
        "1": (0.25, 0.45), "2": (0.50, 0.45), "3": (0.75, 0.45),
        "4": (0.25, 0.58), "5": (0.50, 0.58), "6": (0.75, 0.58),
        "7": (0.25, 0.71), "8": (0.50, 0.71), "9": (0.75, 0.71),
        "0": (0.50, 0.84),
    }
    assert PIN_PAD_FRACTIONS == expected

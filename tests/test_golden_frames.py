"""Golden-frame regression: pipeline renders must stay byte-identical.

The screen texture was made physical in 3D (upright, unmirrored for a human
at the deck); the offscreen overhead/toolcam renderers are fed the
compensating deck-convention flip (``DeckScene._pipeline_canvas``), so every
frame the host stack's vision consumes must be EXACTLY the bytes it always
was. These tests render fixed DUT states at fixed toolhead poses and compare
against golden PNGs captured with the deck-convention texture
(``render_for_deck()``), which ``_pipeline_canvas`` reproduces — that
equivalence is asserted directly as well.

Goldens live in ``tests/golden/``; regenerate them with
``python tests/test_golden_frames.py --regen`` (only ever after verifying the
new frames are correct by eye — the goldens ARE the pipeline contract).
Note: frames are GPU/driver-rendered; goldens are exact for the reference
machine this repo is developed on.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from steropes.dut import DeviceUnderTest
from steropes.phone import PIN_PAD_FRACTIONS, PhoneDUT
from steropes.scene import DeckScene, load_profile

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
PROFILES = Path(__file__).resolve().parent.parent / "profiles"

#: Toolhead poses for the toolcam goldens (carriage centred over the screen;
#: the toolcam sits at gantry.TOOLCAM_OFFSET_MM = (-12, 0) from the carriage).
PHONE_TOOL_XY = (95.75 + 12.0, 152.5)
POS_TOOL_XY = (200.0 + 12.0, 108.0)


def _phone_states() -> list[tuple[str, PhoneDUT]]:
    """(state_name, DUT in that state) for sleep/locked/pin/launcher."""
    states = [("sleep", PhoneDUT(pin="1337"))]
    locked = PhoneDUT(pin="1337")
    locked.press_button("power")
    states.append(("locked", locked))
    pin = PhoneDUT(pin="1337")
    pin.press_button("power")
    pin.swipe(0.5, 0.9, 0.5, 0.2)
    states.append(("pin", pin))
    launcher = PhoneDUT(pin="1337")
    launcher.press_button("power")
    launcher.swipe(0.5, 0.9, 0.5, 0.2)
    for digit in "1337":
        fx, fy = PIN_PAD_FRACTIONS[digit]
        launcher.tap(fx, fy)
    states.append(("launcher", launcher))
    return states


def _render_frames(tmp_path: Path) -> dict[str, np.ndarray]:
    """Render every golden frame (overhead + toolcam per state)."""
    frames: dict[str, np.ndarray] = {}

    profile = load_profile(PROFILES / "android_phone_v1.yaml")
    scene = DeckScene(profile, screen_shape=(1280, 720),
                      workdir=tmp_path / "phone")
    for name, dut in _phone_states():
        scene.set_screen(dut.render_screen())
        # interleave the two renderers: the texture upload must stay correct
        # with both GL contexts live (the pre-fix bug dropped uploads to the
        # non-current context).
        frames[f"phone_{name}_overhead"] = scene.render_overhead()
        frames[f"phone_{name}_toolcam"] = scene.render_toolcam()
    # a lit toolcam frame with the toolhead over the screen
    locked = PhoneDUT(pin="1337")
    locked.press_button("power")
    scene.set_screen(locked.render_screen())
    scene.move_toolhead(*PHONE_TOOL_XY)
    frames["phone_locked_over_overhead"] = scene.render_overhead()
    frames["phone_locked_over_toolcam"] = scene.render_toolcam()
    scene.close()

    profile = load_profile(PROFILES / "countertop_pos_v1.yaml")
    dut = DeviceUnderTest(pin=profile.pin, seed=42, cols=3, rows=4)
    dut.show_keypad()
    scene = DeckScene(profile, screen_shape=(240, 320), workdir=tmp_path / "pos")
    scene.set_screen(dut.render_screen())
    scene.move_toolhead(*POS_TOOL_XY)
    frames["pos_keypad_overhead"] = scene.render_overhead()
    frames["pos_keypad_toolcam"] = scene.render_toolcam()
    scene.close()
    return frames


def _decode_golden(name: str) -> np.ndarray:
    img = cv2.imread(str(GOLDEN_DIR / f"{name}.png"))
    assert img is not None, f"missing golden {name}.png (regen with --regen)"
    return img


def test_pipeline_frames_match_golden_byte_exact(tmp_path: Path) -> None:
    frames = _render_frames(tmp_path)
    for name, frame in sorted(frames.items()):
        golden = _decode_golden(name)
        assert frame.shape == golden.shape, name
        if not np.array_equal(frame, golden):
            diff = np.abs(frame.astype(int) - golden.astype(int)).max(axis=2)
            pytest.fail(f"{name}: {(diff > 0).sum()} pixels differ from the "
                        f"golden frame (max delta {diff.max()})")


def test_pipeline_canvas_matches_render_for_deck(tmp_path: Path) -> None:
    """The scene's compensating flip must reproduce render_for_deck() exactly."""
    profile = load_profile(PROFILES / "android_phone_v1.yaml")
    scene = DeckScene(profile, screen_shape=(1280, 720), workdir=tmp_path)
    dut = PhoneDUT(pin="1337")
    dut.press_button("power")
    assert np.array_equal(scene._pipeline_canvas(dut.render_screen()),
                          dut.render_for_deck())
    scene.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--regen"]:
        raise SystemExit("usage: python tests/test_golden_frames.py --regen")
    import tempfile
    GOLDEN_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, frame in _render_frames(Path(tmp)).items():
            cv2.imwrite(str(GOLDEN_DIR / f"{name}.png"), frame)
            print(f"wrote {name}.png")

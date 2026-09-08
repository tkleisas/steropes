"""Minimal YAML scenario runner.

A scenario drives the physics-rendered deck scene through a list of steps and
asserts expectations along the way. One YAML file = one scenario:

    name: keypad_ocr
    profile: ../profiles/countertop_pos_v1.yaml   # relative to this file
    seed: 42
    steps:
      - action: show_screen
        screen: keypad
      - action: render
        save: frame.png
      - action: decode_keypad
      - action: expect_keypad_matches_ground_truth

Run from the command line:

    python -m steropes.scenario scenarios/keypad_ocr.yaml [--out out]

Prints PASS/FAIL per step, saves artifacts under ``<out>/<scenario-name>/``,
and exits non-zero if any step fails.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from . import deck as deck_const
from . import vision
from .dut import DeviceUnderTest
from .scene import DeckScene, TerminalProfile, load_profile

#: Candidate strings for status-line classification on the result screen.
STATUS_CANDIDATES = ["READY", "APPROVED", "DECLINED"]


@dataclass
class StepResult:
    label: str
    ok: bool
    detail: str = ""


class StepFailure(Exception):
    """Raised when a scenario step's expectation is not met."""


class _Context:
    """Mutable state shared across steps of one scenario run."""

    def __init__(self, profile: TerminalProfile, seed: int,
                 workdir: Path) -> None:
        self.dut = DeviceUnderTest(pin=profile.pin, seed=seed,
                                   cols=profile.keypad_cols,
                                   rows=profile.keypad_rows)
        self.scene = DeckScene(
            profile,
            screen_shape=(self.dut.render_screen().shape[0],
                          self.dut.render_screen().shape[1]),
            workdir=workdir)
        self.frame = None
        self.homography = None
        self.rms_mm = None
        self.detected = None
        self.rectified = None
        self.readings = None
        self.plan = None


def _need(value, what: str):
    if value is None:
        raise StepFailure(f"'{what}' requires an earlier step to produce it")
    return value


class ScenarioRunner:
    """Loads and executes one scenario YAML file."""

    def __init__(self, path: str | Path, out_root: str | Path = "out") -> None:
        self.path = Path(path)
        spec = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.name: str = spec.get("name", self.path.stem)
        profile_path = (self.path.parent / spec["profile"]).resolve()
        self.profile = load_profile(profile_path)
        self.seed = int(spec.get("seed", 0))
        self.steps: list[dict] = spec["steps"]
        self.out_dir = Path(out_root) / self.name

    # -- public API ---------------------------------------------------------

    def run(self) -> bool:
        """Execute all steps; returns True when every step passed."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ctx = _Context(self.profile, self.seed, self.out_dir)
        print(f"scenario: {self.name}  (profile={self.profile.name}, "
              f"seed={self.seed})")
        results: list[StepResult] = []
        for i, step in enumerate(self.steps):
            label = self._label(i, step)
            try:
                detail = self._dispatch(step, ctx) or ""
                results.append(StepResult(label, True, detail))
            except (StepFailure, ValueError, RuntimeError) as exc:
                results.append(StepResult(label, False, str(exc)))
                self._report(results)
                ctx.scene.close()
                return False
        self._report(results)
        ctx.scene.close()
        return all(r.ok for r in results)

    @staticmethod
    def _report(results: list[StepResult]) -> None:
        for r in results:
            mark = "PASS" if r.ok else "FAIL"
            suffix = f"  ({r.detail})" if r.detail else ""
            print(f"  [{mark}] {r.label}{suffix}")

    @staticmethod
    def _label(index: int, step: dict) -> str:
        extras = {k: v for k, v in step.items() if k != "action"}
        suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        return f"step {index + 1}: {step['action']}{suffix}"

    # -- step dispatch ----------------------------------------------------------

    def _dispatch(self, step: dict, ctx: _Context) -> str:
        action = step["action"]
        handler = getattr(self, f"_step_{action}", None)
        if handler is None:
            raise StepFailure(f"unknown action {action!r}")
        params = {k: v for k, v in step.items() if k != "action"}
        return handler(ctx, **params)

    def _save(self, name: str, image) -> None:
        cv2.imwrite(str(self.out_dir / name), image)

    # -- steps -----------------------------------------------------------------

    def _step_show_screen(self, ctx: _Context, screen: str) -> str:
        if screen == "keypad":
            ctx.dut.show_keypad()
        elif screen != "idle":
            raise StepFailure(f"unknown screen {screen!r} (idle|keypad)")
        ctx.rectified = None  # stale; must be re-rendered + rectified
        return f"dut state: {ctx.dut.state}"

    def _step_render(self, ctx: _Context, save: str | None = None,
                     camera: str = "overhead") -> str:
        ctx.scene.set_screen(ctx.dut.render_screen())
        if camera == "overhead":
            ctx.frame = ctx.scene.render_overhead()
            ctx.rectified = None
            frame = ctx.frame
        elif camera == "toolcam":
            # Toolcam frames are artifacts only; ctx.frame stays the overhead
            # frame that the vision pipeline (calibrate/rectify) consumes.
            frame = ctx.scene.render_toolcam()
        else:
            raise StepFailure(f"unknown camera {camera!r} (overhead|toolcam)")
        if save:
            self._save(save, frame)
        return f"{camera} frame {frame.shape[1]}x{frame.shape[0]}"

    def _step_calibrate(self, ctx: _Context) -> str:
        frame = _need(ctx.frame, "calibrate (render first)")
        ctx.homography, ctx.detected, ctx.rms_mm = vision.calibrate(
            frame, deck_const.DECK_MARKERS, deck_const.ARUCO_DICTIONARY)
        return f"rms={ctx.rms_mm:.4f} mm"

    def _step_expect_markers(self, ctx: _Context, count: int) -> str:
        frame = _need(ctx.frame, "expect_markers (render first)")
        ctx.detected = vision.detect_markers(frame, deck_const.ARUCO_DICTIONARY)
        n = len(ctx.detected)
        if n != count:
            raise StepFailure(f"expected {count} markers, detected {n} "
                              f"(ids {sorted(ctx.detected)})")
        return f"ids {sorted(ctx.detected)}"

    def _step_expect_homography_rms_below(self, ctx: _Context, value: float) -> str:
        rms = _need(ctx.rms_mm, "expect_homography_rms_below (calibrate first)")
        if rms >= value:
            raise StepFailure(f"homography RMS {rms:.4f} mm >= {value} mm")
        return f"{rms:.4f} mm < {value} mm"

    def _rectify_now(self, ctx: _Context):
        if ctx.homography is None:
            self._step_calibrate(ctx)
        frame = _need(ctx.frame, "rectify (render first)")
        polygon_px = [ctx.homography.deck_to_pixel(x, y)
                      for x, y in self.profile.screen_polygon_mm]
        ctx.rectified = vision.rectify(
            frame, polygon_px, self.profile.screen_canonical_px)
        return ctx.rectified

    def _step_decode_keypad(self, ctx: _Context,
                            save: str | None = None) -> str:
        rectified = self._rectify_now(ctx)
        if save:
            self._save(save, rectified)
        cells = vision.split_grid(rectified, self.profile.keypad_cols,
                                  self.profile.keypad_rows)
        ctx.readings = vision.read_keypad(cells)
        decoded = [r.digit for r in ctx.readings]
        return f"decoded {decoded}"

    def _step_expect_keypad_matches_ground_truth(self, ctx: _Context) -> str:
        readings = _need(ctx.readings,
                         "expect_keypad_matches_ground_truth (decode first)")
        decoded = [r.digit for r in readings]
        if decoded != list(ctx.dut.layout):
            raise StepFailure(f"decoded {decoded} != ground truth "
                              f"{list(ctx.dut.layout)}")
        low = min(r.confidence for r in readings if r.digit)
        return f"all {len(decoded)} cells match (min digit conf {low:.3f})"

    def _step_plan_taps(self, ctx: _Context, pin: str | None = None) -> str:
        readings = _need(ctx.readings, "plan_taps (decode_keypad first)")
        pin = str(pin) if pin is not None else self.profile.pin
        ctx.plan = vision.plan_pin_taps(readings, pin)
        return f"PIN {pin} -> cells {ctx.plan}"

    def _step_expect_tap_count(self, ctx: _Context, value: int) -> str:
        plan = _need(ctx.plan, "expect_tap_count (plan_taps first)")
        if len(plan) != value:
            raise StepFailure(f"expected {value} taps, planned {len(plan)}")
        return f"{len(plan)} taps"

    def _step_enter_pin(self, ctx: _Context) -> str:
        # M1 fidelity: the tap is a direct model call. Physical contact
        # (toolhead pressing the on-screen key) is a later milestone (M3).
        plan = _need(ctx.plan, "enter_pin (plan_taps first)")
        state = ctx.dut.enter_pin(list(plan))
        return f"dut state: {state}"

    def _step_expect_dut_state(self, ctx: _Context, value: str) -> str:
        if ctx.dut.state != value:
            raise StepFailure(f"dut state {ctx.dut.state!r} != {value!r}")
        return ctx.dut.state

    def _step_expect_screen_text(self, ctx: _Context, value: str) -> str:
        rectified = self._rectify_now(ctx)
        best, score = vision.match_text_line(rectified, STATUS_CANDIDATES)
        if best != value:
            raise StepFailure(f"screen reads {best!r} (score {score:.3f}), "
                              f"expected {value!r}")
        return f"{best!r} (score {score:.3f})"

    # -- gantry steps (M2) ----------------------------------------------------

    def _step_move_toolhead(self, ctx: _Context, x: float, y: float) -> str:
        ctx.scene.move_toolhead(float(x), float(y))
        px, py = ctx.scene.toolhead_position_mm
        return f"toolhead at ({px:.2f}, {py:.2f}) mm"

    def _step_expect_toolhead_at(self, ctx: _Context, x: float, y: float,
                                 tol_mm: float = 0.5) -> str:
        px, py = ctx.scene.toolhead_position_mm
        err = math.hypot(px - float(x), py - float(y))
        if err > tol_mm:
            raise StepFailure(f"toolhead at ({px:.2f}, {py:.2f}) mm, "
                              f"{err:.3f} mm from ({x}, {y}) > tol {tol_mm}")
        return f"err {err:.3f} mm <= {tol_mm} mm"

    def _step_expect_no_collision(self, ctx: _Context) -> str:
        pairs = ctx.scene.collision_pairs
        if pairs:
            raise StepFailure(f"unexpected toolhead contact: {pairs}")
        return "no toolhead contacts"

    def _step_expect_collision(self, ctx: _Context) -> str:
        pairs = ctx.scene.collision_pairs
        if not pairs:
            raise StepFailure("expected toolhead contact, none detected "
                              "(collision guard may be vacuous)")
        return f"contact: {pairs}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m steropes.scenario",
        description="Run steropes YAML scenarios against the rendered deck scene.")
    parser.add_argument("scenarios", nargs="+", help="scenario YAML file(s)")
    parser.add_argument("--out", default="out",
                        help="artifact output root (default: out/)")
    args = parser.parse_args(argv)

    ok = True
    for path in args.scenarios:
        ok = ScenarioRunner(path, out_root=args.out).run() and ok
    print("ALL SCENARIOS PASS" if ok else "SCENARIO FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

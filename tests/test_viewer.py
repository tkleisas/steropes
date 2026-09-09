"""Viewer glue tests — strictly headless: no window is ever opened here.

Covered: CLI parsing, the lock-guarded scene facade, the viewer loop against
a fake viewer object, the display preflight's failure path, and the
serve-mode wiring (machine + HTTP server constructed, never launched).
"""
import threading
import time
from pathlib import Path

import pytest

from steropes.scenario import ScenarioRunner
from steropes.scene import DeckScene
from steropes.viewer import (LockedScene, ViewerUnavailable, build_parser,
                             build_serve, check_display, main, overview_camera,
                             run_viewer_loop)

PROFILE = (Path(__file__).resolve().parent.parent
           / "profiles" / "android_phone_v1.yaml")
SMOKE_SCENARIO = (Path(__file__).resolve().parent.parent
                  / "scenarios" / "smoke_deck.yaml")


# -- CLI ------------------------------------------------------------------------

def test_cli_defaults() -> None:
    args = build_parser().parse_args(["--profile", "p.yaml"])
    assert not args.serve
    assert args.scenario is None
    assert args.smoke is None
    assert args.port == 7125


def test_cli_full_surface() -> None:
    args = build_parser().parse_args(
        ["--profile", "p.yaml", "--serve", "--port", "7000",
         "--smoke", "3.5"])
    assert args.serve and args.port == 7000 and args.smoke == 3.5


def test_cli_serve_and_scenario_are_mutually_exclusive(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--profile", "p.yaml", "--serve", "--scenario", "s.yaml"])
    assert excinfo.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


# -- display preflight -----------------------------------------------------------

def test_check_display_fails_cleanly_without_display(monkeypatch) -> None:
    import glfw
    monkeypatch.setattr(glfw, "init", lambda: False)
    with pytest.raises(ViewerUnavailable, match="no display"):
        check_display()


def test_check_display_fails_cleanly_without_glfw(monkeypatch) -> None:
    import builtins
    real_import = builtins.__import__

    def no_glfw(name, *args, **kwargs):
        if name == "glfw":
            raise ImportError("No module named 'glfw'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_glfw)
    with pytest.raises(ViewerUnavailable, match="glfw"):
        check_display()


# -- LockedScene -----------------------------------------------------------------

class _FakeScene:
    """Records calls and asserts the shared lock is held throughout."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.calls: list = []

    def tap_finger(self, x, y):
        assert self._lock.locked(), "scene call ran without the lock"
        self.calls.append(("tap_finger", x, y))
        return "outcome"

    def set_screen(self, img):
        assert self._lock.locked(), "set_screen ran without the lock"
        self.calls.append(("set_screen", img))

    @property
    def toolhead_position_mm(self):
        assert self._lock.locked(), "attribute read without the lock"
        return (1.0, 2.0)


def test_locked_scene_guards_calls_and_reads() -> None:
    lock = threading.Lock()
    scene = LockedScene(_FakeScene(lock), lock)
    assert scene.tap_finger(1, 2) == "outcome"
    assert scene.toolhead_position_mm == (1.0, 2.0)
    assert scene.wrapped.calls == [("tap_finger", 1, 2)]


def test_locked_scene_signals_screen_changes() -> None:
    lock = threading.Lock()
    changed = threading.Event()
    scene = LockedScene(_FakeScene(lock), lock, on_set_screen=changed.set)
    scene.tap_finger(1, 2)
    assert not changed.is_set()
    scene.set_screen("img")
    assert changed.is_set()


# -- viewer loop (fake viewer, no window) ----------------------------------------

class _FakeViewer:
    """The mujoco.viewer.Handle surface the loop uses, lock-checked."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.syncs = 0
        self.textures: list[int] = []
        self.closed = False

    def is_running(self) -> bool:
        return not self.closed

    def sync(self) -> None:
        assert self._lock.locked(), "viewer.sync ran without the lock"
        self.syncs += 1

    def update_texture(self, tex_id: int) -> None:
        self.textures.append(tex_id)

    def close(self) -> None:
        self.closed = True


def test_viewer_loop_syncs_under_lock_and_closes_on_smoke() -> None:
    lock = threading.Lock()
    fake = _FakeViewer(lock)
    steps = []
    run_viewer_loop(fake, lock, step=lambda: steps.append(1),
                    smoke_s=0.15, poll_s=0.005)
    assert fake.closed
    assert fake.syncs > 0
    assert len(steps) == fake.syncs  # physics stepped once per frame


def test_viewer_loop_reuploads_screen_texture_on_change() -> None:
    lock = threading.Lock()
    fake = _FakeViewer(lock)
    changed = threading.Event()

    def step() -> None:
        if fake.syncs == 3:
            changed.set()          # a screen redraw mid-run
        if fake.syncs >= 8:
            fake.close()           # window closed after the upload

    run_viewer_loop(fake, lock, step=step, smoke_s=None,
                    screen_changed=changed, screen_tex_id=7, poll_s=0.001)
    assert fake.textures == [7]


def test_viewer_loop_stops_when_window_closed() -> None:
    lock = threading.Lock()
    fake = _FakeViewer(lock)

    def step() -> None:
        if fake.syncs >= 3:
            fake.close()  # user closed the window

    run_viewer_loop(fake, lock, step=step, poll_s=0.001)
    assert fake.closed


# -- serve wiring (constructed, never launched) -----------------------------------

def test_build_serve_constructs_machine_and_httpd(tmp_path) -> None:
    httpd = build_serve(PROFILE, 0, "127.0.0.1", tmp_path)
    try:
        machine = httpd.machine
        assert isinstance(machine.data_lock, type(threading.Lock()))
        assert isinstance(machine.screen_changed, threading.Event)
        assert machine.scene.model is not None
    finally:
        httpd.machine.close()
        httpd.server_close()


def test_machine_jobs_wait_for_data_lock(tmp_path) -> None:
    """A queued script cannot touch the scene while the viewer's lock is held."""
    httpd = build_serve(PROFILE, 0, "127.0.0.1", tmp_path)
    machine = httpd.machine
    try:
        with machine.data_lock:  # as the viewer loop would hold it mid-sync
            machine.submit_script("HOME_ALL")
            time.sleep(0.3)
            assert not machine.homed, \
                "machine thread ran a script without the data lock"
        deadline = time.monotonic() + 30.0
        while not machine.homed:
            assert time.monotonic() < deadline, "HOME_ALL never executed"
            time.sleep(0.02)
    finally:
        machine.close()
        httpd.server_close()


def test_screen_change_flagged_after_http_tap(tmp_path) -> None:
    httpd = build_serve(PROFILE, 0, "127.0.0.1", tmp_path)
    machine = httpd.machine
    try:
        machine.screen_changed.clear()
        machine.submit_script("HOME_ALL\nFINGER_TAP X=95 Y=150")
        deadline = time.monotonic() + 60.0
        while True:
            status = machine.status()["print_stats"]["state"]
            assert status != "error"
            if status == "standby" and machine.homed:
                break
            assert time.monotonic() < deadline, "tap script never finished"
            time.sleep(0.02)
        assert machine.screen_changed.is_set()
    finally:
        machine.close()
        httpd.server_close()


# -- scenario wiring (headless; runner drives the wrapped scene) -------------------

def test_scenario_runner_wraps_scene_and_fires_hook(tmp_path) -> None:
    lock = threading.Lock()
    seen: list = []
    wrapped: list = []

    def wrapper(scene):
        proxy = LockedScene(scene, lock)
        wrapped.append(proxy)
        return proxy

    runner = ScenarioRunner(SMOKE_SCENARIO, out_root=tmp_path,
                            on_scene=seen.append, scene_wrapper=wrapper)
    assert runner.run()
    assert len(seen) == 1 and isinstance(seen[0], DeckScene)
    assert len(wrapped) == 1 and wrapped[0].wrapped is seen[0]


# -- camera ------------------------------------------------------------------------

def test_overview_camera_covers_the_deck() -> None:
    class _Cam:
        def __init__(self):
            import numpy as np
            self.lookat = np.zeros(3)
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    cam = _Cam()
    overview_camera(cam)
    assert tuple(cam.lookat[:2]) == (0.2, 0.15)  # deck centre, metres
    assert cam.distance > 0.4  # whole 400x300 mm deck in frame
    assert cam.elevation < -45.0  # looking down, not from the side

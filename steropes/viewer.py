"""Interactive windowed viewing of the live simulation (MuJoCo passive viewer).

Opens the deck scene in MuJoCo's bundled interactive viewer
(``mujoco.viewer.launch_passive``) — orbit/zoom/pause/speed come for free —
instead of a custom renderer. Three modes:

- default (``--profile`` only): the scene in a window, physics free-running
  against the wall clock on the main thread; the user orbits and pauses.
- ``--serve``: the Moonraker-compatible server (:mod:`steropes.server`) runs
  in the same process, so the host stack drives the *visible* sim over HTTP
  from another terminal.
- ``--scenario``: a local scenario runs against the visible scene through
  the existing :class:`steropes.scenario.ScenarioRunner`, in a worker thread.

Threading: whoever *executes* the physics owns the scene (GL contexts are
thread-affine — the server's machine thread, the scenario worker thread, or
in default mode the main thread itself). The passive viewer runs on the main
thread and only ever reads model/data inside ``viewer.sync()``; every such
read is serialised against physics writes by one shared lock — the machine's
``data_lock`` in serve mode, or a lock behind the :class:`LockedScene` facade
in scenario mode. mjdata is never copied; the viewer always shows the live
simulation. Screen-texture changes are signalled through a ``threading.Event``
and re-uploaded to the viewer's own GL context via ``viewer.update_texture``.

Headless safety: GLFW is probed before anything is built; with no display the
CLI exits with a clear message instead of hanging or crashing.

CLI: ``python -m steropes.viewer --profile profiles/android_phone_v1.yaml
[--serve --port 7125] [--scenario scenarios/phone_wake_unlock.yaml]
[--smoke SECONDS]``
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from . import deck as deck_const
from .scenario import ScenarioRunner
from .scene import DeckScene, load_profile
from .server import build_dut, make_server

#: Upper bound on simulated seconds advanced per viewer frame in default mode
#: (keeps the window responsive after a hiccup instead of catching up).
MAX_CATCH_UP_S = 0.1


class ViewerUnavailable(RuntimeError):
    """Raised when no display/GL is available for the interactive viewer."""


def check_display() -> None:
    """Fail fast with a clear message when no windowing system is available."""
    try:
        import glfw
    except ImportError as exc:
        raise ViewerUnavailable(
            "glfw is not importable — it ships with the mujoco wheel; "
            "reinstall mujoco to get the interactive viewer") from exc
    try:
        ok = glfw.init()
    except Exception as exc:
        raise ViewerUnavailable(f"GLFW initialisation failed: {exc}") from exc
    if not ok:
        raise ViewerUnavailable(
            "no display/GL available — the interactive viewer needs a desktop "
            "session (headless: use python -m steropes.server instead)")
    glfw.terminate()


class LockedScene:
    """Lock-guarded facade over a DeckScene shared with the viewer loop.

    Every method call and attribute read is serialised under ``lock``; the
    viewer loop on the main thread takes the same lock around
    ``viewer.sync()``, so mjdata is never read mid-step. ``on_set_screen``
    (if given) is invoked after each ``set_screen`` so the viewer can
    re-upload the screen texture to its own GL context.
    """

    def __init__(self, scene: DeckScene, lock: threading.Lock,
                 on_set_screen=None) -> None:
        self.__dict__["_scene"] = scene
        self.__dict__["_lock"] = lock
        self.__dict__["_on_set_screen"] = on_set_screen

    @property
    def wrapped(self) -> DeckScene:
        """The underlying scene (for the owner thread's own use)."""
        return self.__dict__["_scene"]

    def __getattr__(self, name: str):
        scene = self.__dict__["_scene"]
        lock = self.__dict__["_lock"]
        with lock:
            attr = getattr(scene, name)
        if not callable(attr):
            return attr

        def locked_call(*args, **kwargs):
            with lock:
                result = attr(*args, **kwargs)
            if name == "set_screen" and self.__dict__["_on_set_screen"]:
                self.__dict__["_on_set_screen"]()
            return result

        return locked_call


def overview_camera(cam) -> None:
    """Point the viewer's free camera at a top-down overview of the deck.

    The scene's fixed cameras (``overhead``, ``toolcam``) are listed in the
    viewer's camera drop-down automatically; this is only the initial free
    camera pose.
    """
    cam.lookat[:] = (deck_const.DECK_WIDTH_MM / 2000.0,
                     deck_const.DECK_DEPTH_MM / 2000.0, 0.0)
    cam.distance = 0.55
    cam.azimuth = 90.0    # screen-up is deck +Y, matching the overhead frame
    cam.elevation = -80.0


def run_viewer_loop(viewer, lock: threading.Lock, *, step=None,
                    smoke_s: float | None = None,
                    screen_changed: threading.Event | None = None,
                    screen_tex_id: int | None = None,
                    poll_s: float = 0.01) -> None:
    """Drive a launched passive viewer until the window closes or smoke ends.

    ``viewer`` is anything with the ``mujoco.viewer.Handle`` surface
    (``is_running``/``sync``/``update_texture``/``close``) — tests pass a
    fake, so no window is ever opened outside a real run. ``step`` (default
    mode) advances the physics between frames; it takes ``lock`` itself.
    """
    deadline = None if smoke_s is None else time.monotonic() + smoke_s
    while viewer.is_running():
        if deadline is not None and time.monotonic() >= deadline:
            break
        if step is not None:
            step()
        with lock:
            viewer.sync()
        if (screen_changed is not None and screen_tex_id is not None
                and screen_changed.is_set()):
            screen_changed.clear()
            viewer.update_texture(screen_tex_id)
        time.sleep(poll_s)
    viewer.close()


def _open_viewer(model: mujoco.MjModel, data: mujoco.MjData,
                 lock: threading.Lock, *, step=None, smoke_s: float | None,
                 screen_changed: threading.Event | None = None) -> None:
    """Launch the passive viewer on the main thread and run the loop."""
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "screen")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        overview_camera(viewer.cam)
        run_viewer_loop(viewer, lock, step=step, smoke_s=smoke_s,
                        screen_changed=screen_changed, screen_tex_id=tex_id)


def build_serve(profile: str | Path, port: int, host: str,
                workdir: str | Path, seed: int = 0):
    """Build (machine + HTTP server) for a visible served twin; not launched."""
    return make_server(profile, port, workdir, host=host, seed=seed)


def _run_serve(args: argparse.Namespace) -> int:
    httpd = build_serve(args.profile, args.port, args.host, args.workdir,
                        args.seed)
    machine = httpd.machine  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    print(f"steropes viewer serving {args.profile} on http://{host}:{port}")
    print("the window shows the machine the HTTP API drives; "
          "Ctrl-C or close the window to stop")
    try:
        _open_viewer(machine.scene.model, machine.scene.data,
                     machine.data_lock, smoke_s=args.smoke,
                     screen_changed=machine.screen_changed)
    finally:
        httpd.shutdown()
        machine.close()
        httpd.server_close()
    return 0


def _run_scenario(args: argparse.Namespace) -> int:
    lock = threading.Lock()
    screen_changed = threading.Event()
    ready = threading.Event()
    holder: dict = {}

    def on_scene(scene: DeckScene) -> None:
        holder["scene"] = scene
        ready.set()

    runner = ScenarioRunner(
        args.scenario, out_root=args.out,
        on_scene=on_scene,
        scene_wrapper=lambda scene: LockedScene(
            scene, lock, on_set_screen=screen_changed.set))
    profile = load_profile(args.profile)
    if profile.name != runner.profile.name:
        print(f"note: --profile {profile.name!r} ignored; the window shows "
              f"the scenario's profile {runner.profile.name!r}")

    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("ok", runner.run()),
        name="viewer-scenario", daemon=True)
    thread.start()
    if not ready.wait(timeout=120.0):
        raise ViewerUnavailable("scenario did not build its scene in time")
    scene = holder["scene"]
    print(f"watching scenario {runner.name} live; "
          f"the window stays open after it finishes")
    _open_viewer(scene.model, scene.data, lock, smoke_s=args.smoke,
                 screen_changed=screen_changed)
    thread.join()
    ok = bool(result.get("ok"))
    print(f"scenario {runner.name}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _run_interactive(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    dut = build_dut(profile, args.seed)
    screen = dut.render_for_deck()
    scene = DeckScene(profile, screen_shape=screen.shape[:2],
                      workdir=args.workdir)
    scene.set_screen(screen)
    lock = threading.Lock()  # uncontended here; keeps one loop shape

    model, data = scene.model, scene.data
    dt = model.opt.timestep
    last = time.monotonic()

    def step() -> None:
        nonlocal last
        now = time.monotonic()
        elapsed = min(now - last, MAX_CATCH_UP_S)
        last = now
        n = int(elapsed / dt)
        if n <= 0:
            return
        with lock:
            for _ in range(n):
                mujoco.mj_step(model, data)

    print("interactive twin: drag to orbit, scroll to zoom, "
          "Space pauses; close the window or Ctrl-C to quit")
    try:
        _open_viewer(model, data, lock, step=step, smoke_s=args.smoke)
    finally:
        scene.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m steropes.viewer",
        description="Interactive windowed viewer for the physics twin "
                    "(MuJoCo passive viewer).")
    parser.add_argument("--profile", required=True, help="DUT profile YAML")
    parser.add_argument("--serve", action="store_true",
                        help="run the Moonraker-compatible server in-process "
                             "so the host stack can drive the visible sim")
    parser.add_argument("--port", type=int, default=7125,
                        help="HTTP port with --serve (default: 7125)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--scenario", default=None,
                        help="scenario YAML to run against the visible scene "
                             "(worker thread, existing runner)")
    parser.add_argument("--out", default="out",
                        help="scenario artifact root (default: out/)")
    parser.add_argument("--workdir", default="out/viewer",
                        help="scratch dir for scene assets "
                             "(default: out/viewer)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", type=float, default=None, metavar="SECONDS",
                        help="auto-close the window after SECONDS "
                             "(manual/CI verification)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.serve and args.scenario:
        parser.error("--serve and --scenario are mutually exclusive: in serve "
                     "mode the machine thread owns the scene (GL contexts are "
                     "thread-affine); drive the visible sim over HTTP instead")
    try:
        check_display()
    except ViewerUnavailable as exc:
        print(f"steropes viewer: {exc}", file=sys.stderr)
        return 2
    if args.serve:
        return _run_serve(args)
    if args.scenario:
        return _run_scenario(args)
    return _run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())

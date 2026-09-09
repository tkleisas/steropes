"""Moonraker-compatible HTTP server in front of the physics twin (M4).

Serves the subset of Moonraker's HTTP API the host stack's
``androidtester.motion.MoonrakerClient`` speaks, backed by a live
:class:`steropes.scene.DeckScene` — so the unmodified host stack runs against
the simulated machine exactly as it would against hardware.

Endpoints:

- ``POST /printer/gcode/script`` — ``{"script": "..."}`` -> ``{"result": "ok"}``
  (queued, executed in order against the physics; errors put the machine in
  Klipper's ``error`` state, which strict clients refuse to read as idle).
- ``GET /printer/objects/query`` — ``{"result": {"status": {...}}}"`` with
  ``print_stats.state`` (standby/printing/error) and
  ``motion_report.live_velocity`` (mm/s; exactly 0.0 when idle).
- ``GET /server/info`` — minimal Moonraker flavour.
- ``GET /camera/overhead.png`` / ``/camera/toolcam.png`` — PNG snapshots.
- ``GET /camera/overhead`` / ``/camera/toolcam`` — MJPEG streams
  (``cv2.VideoCapture``-compatible camera sources for the client's config).

Macro contract (what ``androidtester.harness.Harness`` sends, mirrored from
``firmware/klipper/androidtester.cfg``): ``HOME_ALL``, ``TOOLS_UP``,
``FINGER_TAP X= Y=``, ``FINGER_SWIPE X1= Y1= X2= Y2= T=`` (ms),
``FINGER_LONG_PRESS X= Y= T=`` (ms), ``BUTTON_PRESS PLUNGER=<servo name>``,
``PARK``; plus raw ``G28``/``G90``/``G91``/``G0``/``G1``/``G4``/``M400``.
Moves before ``HOME_ALL`` are rejected, as on a real machine.

All scene access (physics steps, texture uploads, renders) happens on one
dedicated machine thread — the offscreen GL context is thread-affine, so HTTP
handler threads only enqueue jobs and wait on results.

CLI: ``python -m steropes.server --profile profiles/android_phone_v1.yaml
--port 7125``
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import gantry, phone as phone_mod, touch
from .dut import DeviceUnderTest
from .phone import PhoneDUT
from .scene import DeckScene, TerminalProfile, load_profile

#: Travel speed reported as live_velocity during XY moves (mm/s; matches the
#: Klipper template's travel_speed).
TRAVEL_SPEED_MM_S = gantry.DEFAULT_SPEED_MM_S
#: Park position (deck mm), as _AT_VARS declares in the Klipper template.
PARK_XY_MM = (10.0, 10.0)
#: Home position (deck mm). The X/Y endstops sit at mid-Y on this rig: the
#: parked carriage (40x40 mm) clears all four corner fiducials there, so the
#: client's post-home calibration frame still sees every marker.
HOME_XY_MM = (0.0, 150.0)


class MachineError(RuntimeError):
    """Raised for bad scripts, unknown macros, and machine faults."""


def _parse_params(tokens: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` macro parameters (keys case-insensitive)."""
    params: dict[str, str] = {}
    for token in tokens:
        if "=" in token:
            key, _, value = token.partition("=")
            params[key.upper()] = value
    return params


def _float(params: dict[str, str], key: str, line: str) -> float:
    try:
        return float(params[key])
    except (KeyError, ValueError):
        raise MachineError(f"missing/invalid parameter {key} in {line!r}")


class TwinMachine:
    """The simulated machine behind the Moonraker facade.

    A single daemon thread owns the scene and processes a FIFO of jobs
    (scripts, renders). Status reads are lock-protected snapshots; ``state``
    is Klipper's vocabulary: ``standby`` idle, ``printing`` while a script
    is queued or executing, ``error`` after a fault (sticky, like Klipper's
    shutdown — a fresh server is the reset).
    """

    def __init__(self, profile_path: str | Path, workdir: str | Path,
                 seed: int = 0) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0          # scripts queued + executing
        self._state = "standby"
        self._message = ""
        self._live_velocity = 0.0
        self._live_position = [0.0, 0.0, 0.0, 0.0]
        self.homed = False
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(Path(profile_path), Path(workdir), seed),
            name="twin-machine", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=120.0):
            raise MachineError("machine thread did not initialise in time")
        if self._init_error is not None:
            raise MachineError(f"machine init failed: {self._init_error}")

    # -- machine thread --------------------------------------------------------

    def _run(self, profile_path: Path, workdir: Path, seed: int) -> None:
        try:
            self.profile: TerminalProfile = load_profile(profile_path)
            if self.profile.dut == "phone":
                self.dut: DeviceUnderTest | PhoneDUT = PhoneDUT(
                    pin=self.profile.pin,
                    apps=[(a.name, a.label, a.x, a.y)
                          for a in self.profile.apps] or None)
            else:
                self.dut = DeviceUnderTest(pin=self.profile.pin, seed=seed,
                                           cols=self.profile.keypad_cols,
                                           rows=self.profile.keypad_rows)
            screen = self.dut.render_for_deck()
            self.scene = DeckScene(self.profile,
                                   screen_shape=screen.shape[:2],
                                   workdir=workdir)
            self.scene.set_screen(screen)
            self._update_position()
        except BaseException as exc:  # surfaced by the constructor
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind = job[0]
            if kind == "script":
                self._run_script(job[1])
            elif kind == "render":
                _, camera, fmt, reply = job
                try:
                    reply.put(self._render(camera, fmt))
                except BaseException as exc:
                    reply.put(exc)

    def _render(self, camera: str, fmt: str) -> bytes:
        frame = (self.scene.render_overhead() if camera == "overhead"
                 else self.scene.render_toolcam())
        ok, buf = cv2.imencode(f".{fmt}", frame)
        if not ok:
            raise MachineError(f"failed to encode {camera} frame as {fmt}")
        return buf.tobytes()

    # -- status -----------------------------------------------------------------

    def _update_position(self) -> None:
        x, y = self.scene.toolhead_position_mm
        z = float(self.scene.data.qpos[self.scene._qz]) * 1000.0
        with self._lock:
            self._live_position = [x, y, z, 0.0]

    def _set_velocity(self, v: float) -> None:
        with self._lock:
            self._live_velocity = v

    def status(self) -> dict:
        """Moonraker-shaped status snapshot for /printer/objects/query."""
        with self._lock:
            state = self._state
            busy = self._pending > 0
            if state != "error":
                state = "printing" if busy else "standby"
            return {
                "print_stats": {
                    "state": state,
                    "filename": "",
                    "total_duration": 0.0,
                    "print_duration": 0.0,
                    "filament_used": 0.0,
                    "message": self._message,
                    "info": {"total_layer": None, "current_layer": None},
                },
                "motion_report": {
                    "live_position": list(self._live_position),
                    "live_velocity": self._live_velocity if busy else 0.0,
                    "live_extruder_velocity": 0.0,
                    "steppers": ["stepper_x", "stepper_y", "stepper_z"],
                    "trapq": [],
                },
            }

    # -- job submission (HTTP handler threads) -----------------------------------

    def submit_script(self, script: str) -> None:
        """Queue a G-code script; the machine reports busy from this instant."""
        if not isinstance(script, str) or not script.strip():
            raise MachineError("empty G-code script")
        with self._lock:
            self._pending += 1
        self._jobs.put(("script", script))

    def snapshot(self, camera: str = "overhead", fmt: str = "png",
                 timeout_s: float = 120.0) -> bytes:
        """Render one frame on the machine thread; blocks until done."""
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._jobs.put(("render", camera, fmt, reply))
        result = reply.get(timeout=timeout_s)
        if isinstance(result, BaseException):
            raise MachineError(f"render failed: {result}")
        return result

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=30.0)
        if self._init_error is None:
            self.scene.close()

    # -- script execution (machine thread) ----------------------------------------

    def _run_script(self, script: str) -> None:
        try:
            if self._state != "error":  # an errored machine stays down
                for raw in script.splitlines():
                    line = raw.split(";", 1)[0].strip()
                    if line:
                        self._exec_line(line)
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._message = str(exc)
        finally:
            with self._lock:
                self._pending -= 1
                if self._pending == 0 and self._state != "error":
                    self._state = "standby"
                    self._live_velocity = 0.0

    def _exec_line(self, line: str) -> None:
        tokens = line.split()
        command = tokens[0].upper()
        params = _parse_params(tokens[1:])
        if command in ("G90", "G91", "M400", "M114"):
            return  # positioning modes / drain / position query: no-ops here
        if command in ("G28", "HOME_ALL"):
            self._move(*HOME_XY_MM)
            self.homed = True
            return
        if command in ("G0", "G1"):
            self._require_homed(line)
            x, y = self.scene.toolhead_position_mm
            self._move(float(params.get("X", x)), float(params.get("Y", y)))
            return
        if command == "G4":  # dwell: P ms
            self._set_velocity(0.0)
            self.scene.settle(float(params.get("P", "0")) / 1000.0)
            return
        if command == "TOOLS_UP":
            # The finger is spring-retracted between contacts; just settle.
            self._set_velocity(0.0)
            self.scene.settle(0.1)
            return
        if command == "PARK":
            self._require_homed(line)
            self.scene.settle(0.1)
            self._move(*PARK_XY_MM)
            return
        if command == "FINGER_TAP":
            self._require_homed(line)
            outcome = self.scene.tap_finger(_float(params, "X", line),
                                            _float(params, "Y", line))
            self._route_tap(outcome)
            return
        if command == "FINGER_LONG_PRESS":
            self._require_homed(line)
            hold_ms = float(params.get("T", "1000"))
            outcome = self.scene.tap_finger(_float(params, "X", line),
                                            _float(params, "Y", line),
                                            hold_s=hold_ms / 1000.0)
            self._route_tap(outcome)
            return
        if command == "FINGER_SWIPE":
            self._require_homed(line)
            outcome = self.scene.swipe_finger(
                _float(params, "X1", line), _float(params, "Y1", line),
                _float(params, "X2", line), _float(params, "Y2", line),
                duration_s=float(params.get("T", "400")) / 1000.0)
            self._route_swipe(outcome)
            return
        if command == "BUTTON_PRESS":
            self._require_homed(line)
            self._button_press(params.get("PLUNGER", ""), line)
            return
        if command == "SET_SERVO":
            return  # raw servo pulses: the pads model their effect already
        raise MachineError(f"unknown G-code command in {line!r}")

    def _require_homed(self, line: str) -> None:
        if not self.homed:
            raise MachineError(f"move before homing: {line!r}")

    def _move(self, x: float, y: float) -> None:
        self._set_velocity(TRAVEL_SPEED_MM_S)
        self.scene.move_toolhead(x, y)
        self._update_position()

    def _button_press(self, plunger: str, line: str) -> None:
        pads = {b.plunger: b for b in self.profile.buttons}
        pad = pads.get(plunger)
        if pad is None:
            raise MachineError(
                f"unknown plunger {plunger!r} in {line!r} "
                f"(known: {sorted(pads)})")
        self._route_tap(self.scene.tap_finger(*pad.pad_mm))

    # -- DUT routing (contact-derived, never from commanded coordinates) ---------

    def _route_tap(self, outcome: touch.TapOutcome) -> None:
        geom = outcome.geom
        if geom == "screen_surface" and outcome.contact_mm is not None:
            if isinstance(self.dut, PhoneDUT):
                fx, fy = phone_mod.mm_to_screen_fraction(
                    self.profile.screen_polygon_mm, *outcome.contact_mm)
                self.dut.tap(fx, fy)
            elif outcome.cell is not None:
                self.dut.tap(outcome.cell)
        elif geom and geom.startswith("button_pad_"):
            name = geom[len("button_pad_"):]
            if hasattr(self.dut, "press_button"):
                self.dut.press_button(name)
        self._refresh_screen()

    def _route_swipe(self, outcome: touch.SwipeOutcome) -> None:
        if isinstance(self.dut, PhoneDUT) and outcome.contacts_mm:
            poly = self.profile.screen_polygon_mm
            fx1, fy1 = phone_mod.mm_to_screen_fraction(
                poly, *outcome.contacts_mm[0])
            fx2, fy2 = phone_mod.mm_to_screen_fraction(
                poly, *outcome.contacts_mm[-1])
            self.dut.swipe(fx1, fy1, fx2, fy2)
        self._refresh_screen()

    def _refresh_screen(self) -> None:
        self.scene.set_screen(self.dut.render_for_deck())


# --- HTTP layer ------------------------------------------------------------------

_MJPEG_BOUNDARY = b"frame"
_MJPEG_INTERVAL_S = 0.25


class _Handler(BaseHTTPRequestHandler):
    """Moonraker-flavoured routes; heavy work defers to the machine thread."""

    server_version = "Steropes/0.1"  # stand-in for Moonraker's version string

    @property
    def machine(self) -> TwinMachine:
        return self.server.machine  # type: ignore[attr-defined]

    def log_message(self, *args) -> None:  # quiet: access logs are noise here
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": {"code": code, "message": message}})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/printer/objects/query":
            self._json(200, {"result": {"status": self.machine.status()}})
        elif path == "/server/info":
            self._json(200, {"result": {
                "klippy_connected": True,
                "klippy_state": "ready",
                "components": ["database", "file_manager"],
                "failed_components": [],
                "registered_directories": ["config", "docs"],
                "warnings": [],
                "websocket_port": self.server.server_port,  # type: ignore[attr-defined]
                "moonraker_version": "steropes-twin",
            }})
        elif path in ("/camera/overhead.png", "/camera/toolcam.png"):
            camera = path.rsplit("/", 1)[1].split(".", 1)[0]
            self._snapshot(camera, "png")
        elif path in ("/camera/overhead", "/camera/toolcam"):
            self._mjpeg(path.rsplit("/", 1)[1])
        else:
            self._error(404, f"no such endpoint: {path}")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/printer/gcode/script":
            self._error(404, f"no such endpoint: {path}")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.machine.submit_script(payload.get("script"))
        except (ValueError, TypeError) as exc:
            self._error(400, f"malformed request: {exc}")
            return
        except MachineError as exc:
            self._error(400, str(exc))
            return
        self._json(200, {"result": "ok"})

    def _snapshot(self, camera: str, fmt: str) -> None:
        try:
            body = self.machine.snapshot(camera, fmt)
        except MachineError as exc:
            self._error(503, str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", f"image/{fmt}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self, camera: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary="
                         f"{_MJPEG_BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True
        while True:
            try:
                jpg = self.machine.snapshot(camera, "jpeg")
                self.wfile.write(b"--" + _MJPEG_BOUNDARY + b"\r\n"
                                 b"Content-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode()
                                 + b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            except MachineError:
                return  # machine shutting down; drop the stream
            time.sleep(_MJPEG_INTERVAL_S)


def make_server(profile: str | Path, port: int, workdir: str | Path,
                host: str = "127.0.0.1", seed: int = 0) -> ThreadingHTTPServer:
    """Build (machine + HTTP server) bound to ``host:port``; not yet serving."""
    machine = TwinMachine(profile, workdir, seed=seed)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    httpd.machine = machine  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m steropes.server",
        description="Moonraker-compatible HTTP front-end for the physics twin.")
    parser.add_argument("--profile", required=True, help="DUT profile YAML")
    parser.add_argument("--port", type=int, default=7125,
                        help="HTTP port (default: 7125, Moonraker's)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--workdir", default="out/server",
                        help="scratch dir for scene assets (default: out/server)")
    args = parser.parse_args(argv)

    httpd = make_server(args.profile, args.port, args.workdir, args.host)
    host, port = httpd.server_address[:2]
    print(f"steropes twin serving {args.profile} on http://{host}:{port}")
    print("endpoints: /printer/gcode/script, /printer/objects/query, "
          "/camera/overhead.png, /camera/overhead (MJPEG)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.machine.close()  # type: ignore[attr-defined]
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

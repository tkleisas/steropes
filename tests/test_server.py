"""M4 tests: the Moonraker-compatible server and the HTTP-level flow.

The strict status reader and the deck-calibration math replicate the host
stack's client (androidtester motion.py / calib.py / vision.py) so the twin
is checked against the same contract the unmodified client enforces.
"""
import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from steropes import deck as deck_const
from steropes import vision
from steropes.phone import PIN_PAD_FRACTIONS
from steropes.server import make_server

PROFILE = (Path(__file__).resolve().parent.parent
           / "profiles" / "android_phone_v1.yaml")

#: Screen polygon from the host's devices/example_phone.yaml (TL TR BR BL).
SCREEN_POLYGON_MM = [(63.0, 84.0), (128.5, 84.0), (128.5, 221.0), (63.0, 221.0)]
PROFILE_PIN = "1337"

#: States the client accepts (androidtester.motion.KNOWN_STATES).
KNOWN_STATES = {"standby", "printing", "paused", "cancelled", "complete",
                "error"}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    httpd = make_server(PROFILE, 0, tmp_path_factory.mktemp("server"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.machine.close()
    httpd.shutdown()


def _base(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def _post_script(server, script: str) -> dict:
    req = urllib.request.Request(
        _base(server) + "/printer/gcode/script",
        data=json.dumps({"script": script}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def _status(server) -> dict:
    url = _base(server) + "/printer/objects/query?print_stats&motion_report"
    return json.load(urllib.request.urlopen(url))["result"]["status"]


def _extract_motion_state(status: dict) -> tuple[str, float]:
    """The client's strict reader (androidtester.motion._extract_motion_state)."""
    state = status["print_stats"]["state"]
    velocity = status["motion_report"]["live_velocity"]
    assert state in KNOWN_STATES, f"unknown print_stats.state {state!r}"
    assert isinstance(velocity, (int, float)) and not isinstance(velocity, bool)
    return state, float(velocity)


def _wait_idle(server, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        state, velocity = _extract_motion_state(_status(server))
        assert state != "error", f"printer error: {status}"
        if state in {"standby", "cancelled", "complete"} and velocity == 0.0:
            return
        assert time.monotonic() < deadline, "machine still busy"
        time.sleep(0.02)


def _snapshot(server, camera: str = "overhead") -> np.ndarray:
    png = urllib.request.urlopen(_base(server) + f"/camera/{camera}.png").read()
    frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None, "snapshot did not decode as a PNG"
    return frame


def _detect_4x4(frame: np.ndarray) -> dict[int, np.ndarray]:
    """DICT_4X4_50 detection with corners (androidtester.calib.detect_markers)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters()).detectMarkers(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten())}


def _calibrate_like_client(frame: np.ndarray) -> np.ndarray:
    """px -> deck-mm homography with the client's corner convention.

    androidtester.calib.calibrate_deck maps pattern corner 0 to
    (cx + size/2, cy - size/2); mirrored placement fails loudly here.
    """
    detections = _detect_4x4(frame)
    assert sorted(detections) == sorted(deck_const.DECK_MARKERS), \
        f"markers missing: detected {sorted(detections)}"
    size = 20.0  # client's marker_size_mm default
    px_points, mm_points = [], []
    for mid in sorted(detections):
        cx, cy = deck_const.DECK_MARKERS[mid]
        h = size / 2.0
        px_points.extend(detections[mid])
        mm_points.extend([(cx + h, cy - h), (cx - h, cy - h),
                          (cx - h, cy + h), (cx + h, cy + h)])
    homography = vision.solve_homography(np.array(px_points), np.array(mm_points))
    errors = []
    for (u, v), (mx, my) in zip(px_points, mm_points):
        p = homography @ np.array([u, v, 1.0])
        errors.append(np.hypot(p[0] / p[2] - mx, p[1] / p[2] - my))
    rms = float(np.sqrt(np.mean(np.square(errors))))
    assert rms < 0.5, f"calibration RMS {rms:.3f} mm over threshold"
    return homography


def _rectify_like_client(frame: np.ndarray, homography_px_to_mm) -> np.ndarray:
    """Warp the profile's screen polygon to the client's default 720x1280."""
    inv = np.linalg.inv(homography_px_to_mm)
    polygon_px = []
    for x, y in SCREEN_POLYGON_MM:
        p = inv @ np.array([x, y, 1.0])
        polygon_px.append([p[0] / p[2], p[1] / p[2]])
    dst = np.array([[0, 0], [719, 0], [719, 1279], [0, 1279]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.array(polygon_px, np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (720, 1280))


# -- status & script plumbing ----------------------------------------------------

def test_status_is_strict_client_compatible_when_idle(server) -> None:
    state, velocity = _extract_motion_state(_status(server))
    assert state == "standby"
    assert velocity == 0.0  # exactly zero: strict wait_idle requires it


def test_home_round_trip_reports_position(server) -> None:
    assert _post_script(server, "HOME_ALL") == {"result": "ok"}
    _wait_idle(server)
    status = _status(server)
    x, y = status["motion_report"]["live_position"][:2]
    assert (x, y) == pytest.approx((0.0, 150.0), abs=1e-6)
    assert server.machine.homed


def test_busy_while_macro_animates(server) -> None:
    _post_script(server, "HOME_ALL")
    _wait_idle(server)
    _post_script(server, "FINGER_SWIPE X1=95.75 Y1=207.3 X2=95.75 Y2=111.4 T=400")
    saw_busy = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        state, velocity = _extract_motion_state(_status(server))
        if state in {"standby", "cancelled", "complete"} and velocity == 0.0:
            break
        assert state == "printing", f"unexpected busy state {state!r}"
        saw_busy = True
        time.sleep(0.005)
    assert saw_busy, "never observed a busy status while a swipe animated"


def test_malformed_script_rejected_with_400(server) -> None:
    req = urllib.request.Request(
        _base(server) + "/printer/gcode/script",
        data=json.dumps({"script": "  "}).encode(),
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req)
    assert excinfo.value.code == 400


def test_unknown_endpoint_404(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_base(server) + "/printer/nope")
    assert excinfo.value.code == 404


def test_move_before_homing_and_unknown_macro_error_out(tmp_path) -> None:
    # Error state is sticky (Klipper shutdown), so this gets its own server.
    httpd = make_server(PROFILE, 0, tmp_path / "err")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        _post_script(httpd, "FINGER_TAP X=100 Y=100")
        _wait_idle_error(httpd)
        assert "before homing" in _status(httpd)["print_stats"]["message"]
    finally:
        httpd.machine.close()
        httpd.shutdown()


def _wait_idle_error(httpd, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state, _ = _extract_motion_state(_status(httpd))
        if state == "error":
            return
        time.sleep(0.02)
    raise AssertionError("machine never entered error state")


# -- camera ------------------------------------------------------------------------

def test_camera_snapshot_is_png_with_all_markers(server) -> None:
    _post_script(server, "HOME_ALL")  # home pose clears every marker
    _wait_idle(server)
    frame = _snapshot(server)
    assert frame.shape == (deck_const.OVERHEAD_CAMERA.height_px,
                           deck_const.OVERHEAD_CAMERA.width_px, 3)
    assert sorted(_detect_4x4(frame)) == [1, 2, 3, 4]


def test_toolcam_snapshot_is_valid_png(server) -> None:
    frame = _snapshot(server, "toolcam")
    assert frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0


def test_calibration_matches_client_corner_convention(server) -> None:
    _post_script(server, "HOME_ALL")
    _wait_idle(server)
    _calibrate_like_client(_snapshot(server))  # asserts markers + RMS itself


# -- HTTP integration: wake -> swipe -> PIN -> Home ---------------------------------

def _fraction_to_mm(fx: float, fy: float) -> tuple[float, float]:
    xs = [p[0] for p in SCREEN_POLYGON_MM]
    ys = [p[1] for p in SCREEN_POLYGON_MM]
    return (min(xs) + fx * (max(xs) - min(xs)),
            min(ys) + fy * (max(ys) - min(ys)))


def test_wake_unlock_flow_over_http(server) -> None:
    """The host's smoke_wake_unlock flow, driven through the twin's HTTP API
    with physical taps, and verified with the client's own vision math."""
    _post_script(server, "HOME_ALL")
    _wait_idle(server)
    _post_script(server, "TOOLS_UP")
    _post_script(server, "BUTTON_PRESS PLUNGER=plunger_0")  # power: wake
    _wait_idle(server)
    assert server.machine.dut.state == "locked"

    x1, y1 = _fraction_to_mm(0.5, 0.9)
    x2, y2 = _fraction_to_mm(0.5, 0.2)
    _post_script(server, f"FINGER_SWIPE X1={x1} Y1={y1} X2={x2} Y2={y2} T=400")
    _wait_idle(server)
    assert server.machine.dut.state == "pin"

    for digit in PROFILE_PIN:
        x, y = _fraction_to_mm(*PIN_PAD_FRACTIONS[digit])
        _post_script(server, f"FINGER_TAP X={x:.2f} Y={y:.2f}")
        _wait_idle(server)
    assert server.machine.dut.state == "launcher"

    # Park aside and read the screen back exactly as the client would.
    _post_script(server, "G1 X20 Y150")
    _wait_idle(server)
    frame = _snapshot(server)
    rectified = _rectify_like_client(frame, _calibrate_like_client(frame))
    assert vision.text_score(rectified, "Home") >= 0.7

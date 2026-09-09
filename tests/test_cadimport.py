"""Tests for the M0 OpenSCAD import pipeline: manifest validation, cached
STL export (binary located via OPENSCAD / default path), scene attachment
of the imported meshes, and the CAD<->collision drift guard.

The integration tests are gated on the OpenSCAD binary being present;
on a machine without it they skip and the rest of the suite still runs.
"""
import os
import re
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest
import yaml

from steropes import cadimport, gantry
from steropes.scene import DeckScene, load_profile

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "cad" / "manifest.yaml"
PROFILE = REPO / "profiles" / "countertop_pos_v1.yaml"
OUT = REPO / "out" / "cad_import"

OPENSCAD = cadimport.find_openscad()
needs_openscad = pytest.mark.skipif(OPENSCAD is None,
                                    reason="OpenSCAD binary not found")


# --- manifest validation -------------------------------------------------------

def _write_manifest(tmp_path: Path, parts: dict) -> Path:
    """Write a manifest YAML into tmp_path with openscad/ alongside."""
    (tmp_path / "openscad").mkdir(exist_ok=True)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"parts": parts}), encoding="utf-8")
    return path


_GOOD_PART = {"scad": "thing.scad", "part": "thing", "role": "visual",
              "body": "toolhead", "offset_mm": [0, 0, 0]}


def test_manifest_loads_reference_manifest() -> None:
    m = cadimport.load_manifest(MANIFEST)
    assert {p.name for p in m.parts} == {"carriage_plate", "finger_body",
                                         "toolcam_bracket"}
    assert m.constants["finger_tip_diameter_mm"] == 8.0
    assert all(p.role == "visual" for p in m.parts)


def test_manifest_missing_scad_file_raises(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, {"thing": _GOOD_PART})
    with pytest.raises(cadimport.ManifestError, match="scad file not found"):
        cadimport.load_manifest(path)


def test_manifest_unknown_part_raises() -> None:
    m = cadimport.load_manifest(MANIFEST)
    with pytest.raises(cadimport.ManifestError, match="unknown part"):
        m.part("flux_capacitor")


def test_manifest_missing_field_raises(tmp_path: Path) -> None:
    (tmp_path / "openscad").mkdir(exist_ok=True)
    (tmp_path / "openscad" / "thing.scad").write_text("cube(1);")
    bad = {k: v for k, v in _GOOD_PART.items() if k != "role"}
    path = _write_manifest(tmp_path, {"thing": bad})
    with pytest.raises(cadimport.ManifestError, match="missing field 'role'"):
        cadimport.load_manifest(path)


def test_manifest_unsupported_role_raises(tmp_path: Path) -> None:
    (tmp_path / "openscad").mkdir(exist_ok=True)
    (tmp_path / "openscad" / "thing.scad").write_text("cube(1);")
    bad = {**_GOOD_PART, "role": "collision"}
    path = _write_manifest(tmp_path, {"thing": bad})
    with pytest.raises(cadimport.ManifestError, match="unsupported role"):
        cadimport.load_manifest(path)


def test_manifest_unknown_body_raises(tmp_path: Path) -> None:
    (tmp_path / "openscad").mkdir(exist_ok=True)
    (tmp_path / "openscad" / "thing.scad").write_text("cube(1);")
    bad = {**_GOOD_PART, "body": "warp_drive"}
    path = _write_manifest(tmp_path, {"thing": bad})
    with pytest.raises(cadimport.ManifestError, match="unknown target body"):
        cadimport.load_manifest(path)


# --- binary location -------------------------------------------------------------

def test_find_openscad_env_override() -> None:
    assert cadimport.find_openscad({"OPENSCAD": str(MANIFEST)}) == str(MANIFEST)
    assert cadimport.find_openscad({"OPENSCAD": "/no/such/binary"}) is None


# --- export (mocked runner) -----------------------------------------------------

def test_export_command_construction() -> None:
    cmd = cadimport.export_command("/bin/openscad", Path("a/x.scad"),
                                   "widget", Path("b/x.stl"))
    assert cmd == ["/bin/openscad", "-o", str(Path("b/x.stl")),
                   "--export-format", "binstl",  # MuJoCo rejects ASCII STL
                   "-D", 'PART="widget"', str(Path("a/x.scad"))]


_FAKE_STL = b"\x00" * 84 + b"\x00" * 50  # 80B header + count + one face


def _fake_runner(calls: list):
    """subprocess.run stand-in: records argv, 'writes' the -o output file."""
    def run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(_FAKE_STL)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


@pytest.fixture
def tmp_manifest(tmp_path: Path) -> cadimport.Manifest:
    (tmp_path / "openscad").mkdir()
    (tmp_path / "openscad" / "thing.scad").write_text("cube(1);")
    return cadimport.load_manifest(
        _write_manifest(tmp_path, {"thing": _GOOD_PART}))


def test_export_runs_binary_and_caches(tmp_manifest: cadimport.Manifest) -> None:
    calls: list = []
    runner = _fake_runner(calls)
    stl = cadimport.export_part(tmp_manifest, "thing", binary="/bin/openscad",
                                runner=runner)
    assert stl.is_file() and len(calls) == 1
    cadimport.export_part(tmp_manifest, "thing", binary="/bin/openscad",
                          runner=runner)
    assert len(calls) == 1  # cached: STL newer than the scad source


def test_export_reexports_stale_cache(tmp_manifest: cadimport.Manifest) -> None:
    calls: list = []
    runner = _fake_runner(calls)
    stl = cadimport.export_part(tmp_manifest, "thing", binary="/bin/openscad",
                                runner=runner)
    # age the STL so the source (and the shared config check) look newer
    stale = stl.stat().st_mtime - 100
    os.utime(stl, (stale, stale))
    cadimport.export_part(tmp_manifest, "thing", binary="/bin/openscad",
                          runner=runner)
    assert len(calls) == 2


def test_export_failure_raises(tmp_manifest: cadimport.Manifest) -> None:
    def bad_runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "syntax error")
    with pytest.raises(cadimport.ManifestError, match="export failed"):
        cadimport.export_part(tmp_manifest, "thing", binary="/bin/openscad",
                              runner=bad_runner)


def test_export_unknown_part_raises(tmp_manifest: cadimport.Manifest) -> None:
    with pytest.raises(cadimport.ManifestError, match="unknown part"):
        cadimport.export_part(tmp_manifest, "nope", binary="/bin/openscad",
                              runner=_fake_runner([]))


# --- export integration (real OpenSCAD) --------------------------------------------

@needs_openscad
def test_export_produces_valid_stl() -> None:
    m = cadimport.load_manifest(MANIFEST)
    stls = cadimport.export_all(m)  # cached across the whole suite
    assert set(stls) == {"carriage_plate", "finger_body", "toolcam_bracket"}
    for name, stl in stls.items():
        info = cadimport.inspect_stl(stl)
        assert info.header.startswith("OpenSCAD")
        assert info.face_count > 50
        spans = tuple(b - a for a, b in zip(info.mins, info.maxs))
        assert all(s > 0.1 for s in spans), f"{name}: degenerate {spans}"


@needs_openscad
def test_export_dimensions_match_design() -> None:
    m = cadimport.load_manifest(MANIFEST)
    stls = cadimport.export_all(m)

    def spans(name):
        info = cadimport.inspect_stl(stls[name])
        return tuple(b - a for a, b in zip(info.mins, info.maxs))

    # carriage: 40 x 40 x 20 plate (gantry.CARRIAGE_HALF_MM * 2)
    assert spans("carriage_plate") == pytest.approx((40, 40, 20), abs=0.01)
    # finger: Ø13 collar in XY, 30 mm shaft in Z (the retaining-pin bore
    # through the collar shaves the extreme X vertices, hence the looser X
    # tolerance)
    fx, fy, fz = spans("finger_body")
    assert fx == pytest.approx(13, abs=0.15)
    assert fy == pytest.approx(13, abs=0.01)
    assert fz == pytest.approx(30, abs=0.01)
    # toolcam bracket: 25 x 24 cradle, 2.5 plate + 6 pocket deep
    assert spans("toolcam_bracket") == pytest.approx((25, 24, 8.5), abs=0.01)


# --- scene integration -------------------------------------------------------------

@needs_openscad
def test_scene_attaches_meshes_as_visual_only(tmp_path: Path) -> None:
    scene = DeckScene(load_profile(PROFILE), screen_shape=(240, 320),
                      workdir=tmp_path, cad_manifest=MANIFEST)
    try:
        for name in ("cad_carriage_plate", "cad_finger_body",
                     "cad_toolcam_bracket"):
            gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"geom {name} missing"
            assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH,
                                     name) >= 0, f"mesh {name} missing"
            assert scene.model.geom_contype[gid] == 0
            assert scene.model.geom_conaffinity[gid] == 0
        # replaced primitives: hidden (alpha 0) but still collidable
        for prim in ("tool_carriage", "tool_finger"):
            gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, prim)
            assert scene.model.geom_rgba[gid, 3] == 0.0
            assert scene.model.geom_contype[gid] == 1
    finally:
        scene.close()


@needs_openscad
def test_rendered_frame_differs_from_primitive_baseline(
        tmp_path_factory) -> None:
    """The imported meshes must actually show up on camera; both frames are
    saved to out/cad_import/ as visual evidence."""
    profile = load_profile(PROFILE)
    base = DeckScene(profile, screen_shape=(240, 320),
                     workdir=tmp_path_factory.mktemp("cad_base"))
    cad = DeckScene(profile, screen_shape=(240, 320),
                    workdir=tmp_path_factory.mktemp("cad_scene"),
                    cad_manifest=MANIFEST)
    try:
        for s in (base, cad):
            s.move_toolhead(200.0, 150.0)
        f_base, f_cad = base.render_overhead(), cad.render_overhead()
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / "baseline_overhead.png"), f_base)
        cv2.imwrite(str(OUT / "cad_overhead.png"), f_cad)
        cv2.imwrite(str(OUT / "cad_toolcam.png"), cad.render_toolcam())
        diff = np.abs(f_cad.astype(np.int16) - f_base.astype(np.int16))
        # the toolhead covers ~0.5% of the frame; require a solid block of
        # clearly-changed pixels, not just noise
        changed = (diff.max(axis=2) > 30).mean()
        assert changed > 0.002, f"CAD meshes barely changed the frame ({changed=})"
    finally:
        base.close()
        cad.close()


# --- drift guard: manifest == cad_config.scad == gantry.py -------------------------

def _scad_const(text: str, name: str):
    """Value of a `NAME = number;` or `NAME = [x, y, z];` assignment."""
    m = re.search(rf"^{name}\s*=\s*(\[[^\]]+\]|[-0-9.eE]+)\s*;", text, re.M)
    assert m, f"{name} not found in cad_config.scad"
    raw = m.group(1)
    if raw.startswith("["):
        return tuple(float(v) for v in raw[1:-1].split(","))
    return float(raw)


def test_manifest_constants_match_gantry() -> None:
    """The collision model's dimensions and the CAD manifest must not drift."""
    c = cadimport.load_manifest(MANIFEST).constants
    chx, chy, _ = gantry.CARRIAGE_HALF_MM
    assert tuple(c["carriage_footprint_mm"]) == (2 * chx, 2 * chy)
    assert c["finger_tip_diameter_mm"] == 2 * gantry.FINGER_RADIUS_MM
    assert tuple(c["finger_offset_mm"]) == gantry.FINGER_OFFSET_MM
    assert tuple(c["toolcam_offset_mm"]) == gantry.TOOLCAM_OFFSET_MM
    # ...and the bracket is actually placed where the collision model's
    # toolcam sits (the scene reads placement from the manifest).
    m = cadimport.load_manifest(MANIFEST)
    assert m.part("toolcam_bracket").offset_mm == tuple(c["toolcam_offset_mm"])


def test_cad_config_matches_manifest() -> None:
    """The OpenSCAD sources must carry the same constants as the manifest."""
    m = cadimport.load_manifest(MANIFEST)
    text = (m.cad_dir / "cad_config.scad").read_text(encoding="utf-8")
    c = m.constants
    assert _scad_const(text, "CARRIAGE_W") == c["carriage_footprint_mm"][0]
    assert _scad_const(text, "CARRIAGE_D") == c["carriage_footprint_mm"][1]
    assert _scad_const(text, "FINGER_DIA") == c["finger_tip_diameter_mm"]
    assert _scad_const(text, "FINGER_OFFSET") == tuple(c["finger_offset_mm"])
    assert _scad_const(text, "TOOLCAM_OFFSET") == tuple(c["toolcam_offset_mm"])

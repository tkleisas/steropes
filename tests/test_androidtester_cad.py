"""Capstone: import the AndroidTester machine's REAL OpenSCAD parts.

``cad/manifest.androidtester.yaml`` sources gantry_carriage/finger_module/
camera_mounts straight from the machine repository (env ``ANDROIDTESTER_ROOT``,
default ``../AndroidTester``), exports them through the same pipeline as the
local parts, and attaches them to the toolhead as visual-only geoms.

Every test here is gated: it skips unless the OpenSCAD binary is present AND
the machine checkout exists (set ANDROIDTESTER_ROOT, or keep a sibling
checkout). With neither, the rest of the suite is unaffected.
"""
from pathlib import Path

import mujoco
import pytest

from steropes import cadimport, gantry, touch
from steropes.scene import DeckScene, load_profile

REPO = Path(__file__).resolve().parent.parent
AT_MANIFEST = REPO / "cad" / "manifest.androidtester.yaml"
PROFILE = REPO / "profiles" / "countertop_pos_v1.yaml"

AT_ROOT = cadimport._expand_openscad_dir("${ANDROIDTESTER_ROOT}", AT_MANIFEST)
OPENSCAD = cadimport.find_openscad()

needs_machine_cad = pytest.mark.skipif(
    OPENSCAD is None or not (AT_ROOT / "cad" / "openscad").is_dir(),
    reason="needs OpenSCAD and the AndroidTester checkout "
           "(set ANDROIDTESTER_ROOT)")


def test_manifest_missing_machine_repo_raises(tmp_path: Path,
                                              monkeypatch) -> None:
    """A clear error when the source root does not resolve (ungated)."""
    monkeypatch.delenv(cadimport.AT_ROOT_ENV, raising=False)
    bad = tmp_path / "m.yaml"
    bad.write_text(
        "openscad_dir: ${ANDROIDTESTER_ROOT}/cad/openscad\nparts: {}\n",
        encoding="utf-8")
    with pytest.raises(cadimport.ManifestError, match="openscad_dir"):
        cadimport.load_manifest(bad)


@needs_machine_cad
def test_manifest_resolves_into_machine_repo() -> None:
    m = cadimport.load_manifest(AT_MANIFEST)
    assert {p.name for p in m.parts} == {
        "at_toolhead_plate", "at_finger_body", "at_toolcam_bracket"}
    # sources read from the machine repo; STLs written inside THIS repo
    expected = cadimport._expand_openscad_dir(
        "${ANDROIDTESTER_ROOT}/cad/openscad", AT_MANIFEST).resolve()
    assert m.cad_dir.resolve() == expected
    assert m.stl_dir == REPO / "cad" / "stl"
    assert m.shared_config == "00_config.scad"
    for p in m.parts:
        assert m.scad_path(p).is_file()


@needs_machine_cad
def test_export_matches_machine_design() -> None:
    """Drift guard against the real CAD: STL spans == manifest constants."""
    m = cadimport.load_manifest(AT_MANIFEST)
    stls = cadimport.export_all(m)  # cached, like the local parts

    def spans(name):
        info = cadimport.inspect_stl(stls[name])
        assert info.header.startswith("OpenSCAD")
        assert info.face_count > 50
        return tuple(b - a for a, b in zip(info.mins, info.maxs))

    c = m.constants
    assert spans("at_toolhead_plate") == pytest.approx(
        tuple(c["toolhead_plate_mm"]), abs=0.01)
    # finger body: the mounting flange adds 4 mm ears beyond BODY_W in X
    fx, fy, fz = spans("at_finger_body")
    bw, bd, bh = (float(v) for v in c["finger_body_mm"])
    assert (fx, fy, fz) == pytest.approx((bw + 8.0, bd, bh), abs=0.01)
    assert spans("at_toolcam_bracket") == pytest.approx(
        tuple(c["toolcam_bracket_mm"]), abs=0.01)


@needs_machine_cad
def test_scene_with_machine_cad_renders_and_taps(tmp_path: Path) -> None:
    """The twin wearing the real toolhead: meshes present, frames render,
    and a physical tap still registers through the imported visuals."""
    profile = load_profile(PROFILE)
    scene = DeckScene(profile, screen_shape=(240, 320), workdir=tmp_path,
                      cad_manifest=AT_MANIFEST)
    try:
        for name in ("cad_at_toolhead_plate", "cad_at_finger_body",
                     "cad_at_toolcam_bracket"):
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

        scene.move_toolhead(200.0, 150.0)
        overhead, toolcam = scene.render_overhead(), scene.render_toolcam()
        assert overhead.shape == (scene.camera.height_px,
                                  scene.camera.width_px, 3)
        assert toolcam.shape == (gantry.TOOL_CAMERA.height_px,
                                 gantry.TOOL_CAMERA.width_px, 3)
        # the real plate (100x60 mm) is a large visual: it must be visible
        # in both frames, not silently mis-placed off-camera
        assert overhead.std() > 1.0 and toolcam.std() > 1.0

        target = touch.cell_center(profile.screen_polygon_mm,
                                   profile.keypad_cols, profile.keypad_rows, 4)
        outcome = scene.tap_finger(*target)
        assert outcome.geom == "screen_surface"
        assert outcome.cell == 4
        assert touch.FORCE_MIN_N < outcome.peak_force_n < touch.FORCE_MAX_N
    finally:
        scene.close()

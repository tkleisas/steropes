"""Unit tests for gantry kinematics, travel limits, and the collision guard."""
from pathlib import Path

import pytest

from steropes import gantry
from steropes.scene import DeckScene, load_profile

PROFILE = (Path(__file__).resolve().parent.parent
           / "profiles" / "countertop_pos_v1.yaml")


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    s = DeckScene(load_profile(PROFILE), screen_shape=(240, 320),
                  workdir=tmp_path_factory.mktemp("gantry"))
    yield s
    s.close()


def test_clamp_to_deck_keeps_inside_points() -> None:
    assert gantry.clamp_to_deck(100.0, 200.0) == (100.0, 200.0)
    assert gantry.clamp_to_deck(0.0, 300.0) == (0.0, 300.0)


def test_clamp_to_deck_clamps_outside_points() -> None:
    assert gantry.clamp_to_deck(-5.0, 500.0) == (0.0, 300.0)
    assert gantry.clamp_to_deck(999.0, -1.0) == (400.0, 0.0)


def test_check_deck_limits_rejects_out_of_range() -> None:
    gantry.check_deck_limits(400.0, 300.0)  # edges are valid
    with pytest.raises(ValueError, match="outside travel limits"):
        gantry.check_deck_limits(-0.1, 100.0)
    with pytest.raises(ValueError, match="outside travel limits"):
        gantry.check_deck_limits(100.0, 300.1)


def test_interpolate_hits_goal_at_fixed_dt() -> None:
    pts = gantry.interpolate((0.0, 0.0), (100.0, 0.0), 100.0, 0.1)
    assert pts[-1] == (100.0, 0.0)
    assert len(pts) == 10  # 100 mm at 10 mm/step
    assert all(pts[i][0] < pts[i + 1][0] for i in range(len(pts) - 1))


def test_interpolate_zero_length_still_yields_one_point() -> None:
    assert gantry.interpolate((50.0, 50.0), (50.0, 50.0), 100.0, 0.1) == [(50.0, 50.0)]


def test_move_toolhead_rejects_out_of_range(scene: DeckScene) -> None:
    with pytest.raises(ValueError, match="outside travel limits"):
        scene.move_toolhead(-10.0, 100.0)
    with pytest.raises(ValueError, match="outside travel limits"):
        scene.move_toolhead(100.0, 400.0)


def test_move_toolhead_reaches_target(scene: DeckScene) -> None:
    scene.move_toolhead(80.0, 100.0)
    px, py = scene.toolhead_position_mm
    assert px == pytest.approx(80.0, abs=1e-6)
    assert py == pytest.approx(100.0, abs=1e-6)


def test_safe_move_reports_no_contacts(scene: DeckScene) -> None:
    scene.move_toolhead(80.0, 250.0)
    scene.move_toolhead(80.0, 100.0)
    scene.move_toolhead(200.0, 100.0)  # over the terminal, clear at cruise
    assert scene.collision_pairs == []


def test_intercept_move_reports_contacts(scene: DeckScene) -> None:
    # Vacuous-guard check: the detector must fire when it should.
    scene.move_toolhead(80.0, 100.0)
    scene.move_toolhead(80.0, 240.0)
    scene.move_toolhead(200.0, 240.0)  # approached above the riser — clean
    assert scene.collision_pairs == []
    scene.move_toolhead(200.0, 100.0)  # crosses the terminal riser
    pairs = scene.collision_pairs
    assert pairs, "expected contact with the terminal riser"
    assert any("terminal_riser" in pair for pair in pairs)
    assert any(any(g.startswith("tool_") for g in pair) for pair in pairs)


def test_contact_log_resets_on_next_move(scene: DeckScene) -> None:
    scene.move_toolhead(200.0, 240.0)
    scene.move_toolhead(200.0, 100.0)
    assert scene.collision_pairs  # intercept, as above
    scene.move_toolhead(80.0, 100.0)  # clean retreat, routed around the riser
    scene.move_toolhead(80.0, 250.0)
    assert scene.collision_pairs == []

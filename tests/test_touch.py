"""M3 tests: physical finger taps, contact-derived registration, force budget."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from steropes import gantry, touch
from steropes.dut import DeviceUnderTest
from steropes.scene import DeckScene, load_profile
from steropes.scenario import ScenarioRunner, StepFailure

PROFILE = (Path(__file__).resolve().parent.parent
           / "profiles" / "countertop_pos_v1.yaml")
PROFILE_OBJ = load_profile(PROFILE)
POLY = PROFILE_OBJ.screen_polygon_mm
COLS, ROWS = PROFILE_OBJ.keypad_cols, PROFILE_OBJ.keypad_rows

CENTRE_CELL = 4  # row 1, col 1 of the 3x4 keypad


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    s = DeckScene(PROFILE_OBJ, screen_shape=(240, 320),
                  workdir=tmp_path_factory.mktemp("touch"))
    yield s
    s.close()


def cell_xy(cell: int) -> tuple[float, float]:
    return touch.cell_center(POLY, COLS, ROWS, cell)


# -- geometry helpers ----------------------------------------------------------


def test_cell_at_round_trips_cell_center() -> None:
    for cell in range(COLS * ROWS):
        assert touch.cell_at(POLY, COLS, ROWS, *cell_xy(cell)) == cell


def test_cell_at_returns_none_outside_screen() -> None:
    assert touch.cell_at(POLY, COLS, ROWS, 10.0, 10.0) is None


# -- physical tap registration ---------------------------------------------------


def test_physical_tap_registers_intended_cell(scene: DeckScene) -> None:
    target = cell_xy(CENTRE_CELL)
    outcome = scene.tap_finger(*target)
    assert outcome.geom == "screen_surface"
    assert outcome.cell == CENTRE_CELL
    # the registered point comes from the contact, within a finger radius
    # of the target — it is not an echo of the command
    assert outcome.contact_mm is not None
    err = ((outcome.contact_mm[0] - target[0]) ** 2
           + (outcome.contact_mm[1] - target[1]) ** 2) ** 0.5
    assert err < gantry.FINGER_RADIUS_MM


def test_off_target_tap_registers_miss(scene: DeckScene) -> None:
    # anti-vacuous: a tap on bare deck must not reach the DUT
    outcome = scene.tap_finger(60.0, 200.0)
    assert outcome.geom == "deck"
    assert outcome.cell is None
    assert outcome.peak_force_n > 0.0  # contact happened; the miss is real


def test_wrong_cell_tap_registers_wrong_cell(scene: DeckScene) -> None:
    # aim one cell left of centre: contact physics, not the intent, decides
    x, y = cell_xy(CENTRE_CELL)
    cell_w = (POLY[1][0] - POLY[0][0]) / COLS
    outcome = scene.tap_finger(x - cell_w, y)
    assert outcome.cell == CENTRE_CELL - 1


# -- force budget -----------------------------------------------------------------


def test_peak_force_within_budget(scene: DeckScene) -> None:
    outcome = scene.tap_finger(*cell_xy(CENTRE_CELL))
    assert touch.FORCE_MIN_N < outcome.peak_force_n < touch.FORCE_MAX_N


def test_stiffer_spring_exceeds_budget(scene: DeckScene) -> None:
    stiff = scene.tap_finger(*cell_xy(CENTRE_CELL), stiffness=4000.0)
    assert stiff.peak_force_n > touch.FORCE_MAX_N
    # compliance override is temporary: model params and behaviour restored
    jnt = scene._jz_joint
    assert scene.model.jnt_stiffness[jnt] == pytest.approx(
        touch.FINGER_SPRING_N_PER_M)
    nominal = scene.tap_finger(*cell_xy(CENTRE_CELL))
    assert touch.FORCE_MIN_N < nominal.peak_force_n < touch.FORCE_MAX_N


def test_tap_is_deterministic(scene: DeckScene) -> None:
    a = scene.tap_finger(*cell_xy(CENTRE_CELL))
    b = scene.tap_finger(*cell_xy(CENTRE_CELL))
    assert a.peak_force_n == pytest.approx(b.peak_force_n, abs=1e-9)
    assert a.cell == b.cell


# -- scenario step handlers --------------------------------------------------------


def _runner() -> ScenarioRunner:
    return ScenarioRunner.__new__(ScenarioRunner)  # handlers need no init state


def _ctx_with_taps(*outcomes: touch.TapOutcome) -> SimpleNamespace:
    return SimpleNamespace(taps=list(outcomes))


def test_expect_tap_registered_pass_and_fail() -> None:
    hit = touch.TapOutcome(target_mm=(0, 0), contact_mm=(1, 1),
                           geom="screen_surface", cell=CENTRE_CELL,
                           peak_force_n=3.0)
    ctx = _ctx_with_taps(hit)
    _runner()._step_expect_tap_registered(ctx, cell=CENTRE_CELL)
    with pytest.raises(StepFailure, match="registered cell"):
        _runner()._step_expect_tap_registered(ctx, cell=CENTRE_CELL + 1)


def test_expect_tap_registered_accepts_explicit_miss() -> None:
    miss = touch.TapOutcome(target_mm=(0, 0), contact_mm=(1, 1),
                            geom="deck", cell=None, peak_force_n=2.0)
    _runner()._step_expect_tap_registered(_ctx_with_taps(miss), cell=None)


def test_expect_tap_registered_needs_a_tap() -> None:
    with pytest.raises(StepFailure, match="no taps recorded"):
        _runner()._step_expect_tap_registered(_ctx_with_taps(), cell=0)


def test_expect_max_tap_force_below() -> None:
    soft = touch.TapOutcome(target_mm=(0, 0), contact_mm=(1, 1),
                            geom="screen_surface", cell=0, peak_force_n=3.0)
    hard = touch.TapOutcome(target_mm=(0, 0), contact_mm=(1, 1),
                            geom="screen_surface", cell=0, peak_force_n=9.0)
    _runner()._step_expect_max_tap_force_below(_ctx_with_taps(soft), n=6.0)
    with pytest.raises(StepFailure, match="peak tap force"):
        _runner()._step_expect_max_tap_force_below(_ctx_with_taps(hard), n=6.0)
    with pytest.raises(StepFailure, match="no taps recorded"):
        _runner()._step_expect_max_tap_force_below(_ctx_with_taps(), n=6.0)


def test_expect_max_tap_force_below_rejects_vacuous_zero() -> None:
    ghost = touch.TapOutcome(target_mm=(0, 0), contact_mm=None,
                             geom=None, cell=None, peak_force_n=0.0)
    with pytest.raises(StepFailure, match="vacuous"):
        _runner()._step_expect_max_tap_force_below(_ctx_with_taps(ghost), n=6.0)


def test_route_tap_feeds_only_real_cells() -> None:
    # seed-42 layout has digit '5' in cell 1; a miss must not reach the DUT
    dut = DeviceUnderTest(pin="5", seed=42)
    dut.show_keypad()
    ctx = SimpleNamespace(dut=dut)
    miss = touch.TapOutcome(target_mm=(0, 0), contact_mm=None,
                            geom="deck", cell=None, peak_force_n=2.0)
    ScenarioRunner._route_tap(None, ctx, miss)
    assert dut.submit() == "declined"  # nothing entered
    dut2 = DeviceUnderTest(pin="5", seed=42)
    dut2.show_keypad()
    hit = touch.TapOutcome(target_mm=(0, 0), contact_mm=(1, 1),
                           geom="screen_surface", cell=1, peak_force_n=3.0)
    ScenarioRunner._route_tap(None, SimpleNamespace(dut=dut2), hit)
    assert dut2.submit() == "approved"

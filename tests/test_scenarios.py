"""Run every YAML scenario in scenarios/ as a headless pytest test."""
from pathlib import Path

import pytest

from steropes.scenario import ScenarioRunner

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
SCENARIOS = sorted(SCENARIO_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario(path: Path, tmp_path: Path) -> None:
    assert ScenarioRunner(path, out_root=tmp_path).run()

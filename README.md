# Steropes

> ταῖς ἔνι μὲν νόος ἐστὶ μετὰ φρεσίν, ἔν δὲ καὶ αὐδὴ
> καὶ σθένος, ἀθανάτων δὲ θεῶν ἄπο ἔργα ἴσασιν.
>
> *"In them is understanding in their hearts, and in them speech and strength,
> and from the immortal gods they know the crafts."*
> — Homer, *Iliad* 18.419–420, on the golden automaton aides of Hephaestus

**A 3D, physics-based digital twin for robotic test hardware** — named for Steropes
(Στερόπης, "lightning-bearer"), one of the three Cyclopes who labored in the forge
of Hephaestus: the crew that did the heavy work before the god's creations walked,
and a creature of a **single eye** — which is what this toolkit lends to machines
that do not yet have one.

Steropes simulates a physical machine in a rigid-body physics engine with emulated
cameras, so that control software, vision pipelines, and mechanical designs can be
exercised **before the hardware exists** — and continuously in CI afterwards.

The reference workload is a robotic harness that physically tests payment
terminals (card insert/eject, magstripe swipe, contactless tap, scrambled
on-screen PIN entry), but the toolkit is designed to be reusable for any small
gantry/manipulator-class machine described by OpenSCAD parts and a Klipper-style
configuration.

## What it does

- **Physics, not fakes.** Rigid-body dynamics (MuJoCo) drive the machine: gantry
  motion, servo tools, key presses against a force budget, card grip/insert/swipe
  with real contact friction. Custody of the card *emerges* from physics instead of
  being asserted by test code.
- **Emulated cameras.** Overhead and toolhead cameras render true-perspective RGB
  frames from any pose. The device-under-test's screen is a dynamic texture driven
  by a terminal state model — so the *real* vision pipeline (ArUco homography,
  perspective rectification, OCR) runs against rendered frames, glare and all.
- **OpenSCAD in, structure once.** Visual geometry comes from the machine's actual
  OpenSCAD parts (per-part STL export); the articulated model (joints, masses,
  simplified collision shapes) is authored once in MJCF/URDF. Collision-checking of
  every motion path against the 3D scene comes free.
- **Standard seams.** Steropes is meant to sit behind the interfaces machine
  control stacks already use — a Moonraker-compatible HTTP API, a motion client
  protocol, and a camera abstraction — so unmodified host software (CLI, scenario
  engine, calibration, vision) can talk to the simulated machine exactly as it
  would to hardware.

## Quick start

Requires Python 3.12+.

```sh
pip install -e ".[dev]"

# run a scenario against the physics-rendered scene
python -m steropes.scenario scenarios/keypad_ocr.yaml

# run the full test suite (all scenarios, headless)
pytest
```

Scenarios print PASS/FAIL per step and save artifacts (rendered frames,
rectified screens) under `out/<scenario-name>/`. See `scenarios/` for the
YAML step vocabulary and `profiles/` for the terminal profile format.

## Roadmap

Each milestone stands alone — if the project stalls, what's built is still useful.

- **M0 — Reference machine CAD.** `$part`-selectable per-part STL export from
  OpenSCAD sources: the geometry import pipeline for the reference gantry machine.
- **M1 — Static scene + cameras. DONE.** Deck, terminal, ArUco markers, animated
  screen texture; overhead camera frames through the real vision pipeline, incl.
  OCR of a rendered scrambled PIN keypad. Proof:
  `python -m steropes.scenario scenarios/pin_purchase.yaml` — a rendered keypad is
  decoded by template OCR, a PIN is planned and entered, and the result screen
  reads back APPROVED from the re-rendered frame.
- **M2 — Kinematic gantry.** Toolhead moves along rails; toolcam renders from real
  poses; motion paths collision-checked against the scene.
- **M3 — Contacts.** Key actuation with spring-damper keys, card grip/insert/eject
  with friction and compliance.
- **M4 — Closed loop.** Moonraker-emulating server in front of the physics backend;
  an unmodified Klipper-style host stack runs scenarios against it; CI tier +
  drift guards against the reference machine's CAD and firmware configuration.

## Reality gap — an honest note

A physics sim is **design validation, not hardware validation**. Friction
coefficients and contact parameters are wrong until measured on the physical
machine; results like "insertion succeeds at 8 N" are evidence, not proof. FEM-level
questions (printed flexures) and signal-level questions (NFC, magstripe) are out of
scope by design.

## Status

Pre-alpha. M1 (static scene + cameras + vision loop) is complete and covered by
scenario tests; the package is importable and `pytest` is green. Everything past
M1 is design scaffold — see the roadmap above.

## License

[MIT](LICENSE) — do what you want, attribution appreciated.

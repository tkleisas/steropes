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

# serve the twin over a Moonraker-compatible HTTP API (phone DUT)
python -m steropes.server --profile profiles/android_phone_v1.yaml --port 7125

# run the full test suite (all scenarios, headless)
pytest
```

Scenarios print PASS/FAIL per step and save artifacts (rendered frames,
rectified screens) under `out/<scenario-name>/`. See `scenarios/` for the
YAML step vocabulary and `profiles/` for the terminal profile format.

## Roadmap

Each milestone stands alone — if the project stalls, what's built is still useful.

- **M0 — Reference machine CAD. DONE.** `$part`-selectable per-part STL
  export from OpenSCAD sources: the geometry import pipeline for the
  reference gantry machine. Parts in `cad/openscad/` (carriage plate,
  finger body, toolcam bracket, driven by a shared `cad_config.scad`)
  are exported via the OpenSCAD CLI to `cad/stl/` (cached, gitignored)
  and attached to the toolhead as visual-only geoms — collision stays on
  the MJCF primitives, so scenarios behave identically with or without
  the CAD meshes. A drift guard pins the manifest constants to the
  gantry collision constants. See [The CAD pipeline](#the-cad-pipeline)
  below.
- **M1 — Static scene + cameras. DONE.** Deck, terminal, ArUco markers, animated
  screen texture; overhead camera frames through the real vision pipeline, incl.
  OCR of a rendered scrambled PIN keypad. Proof:
  `python -m steropes.scenario scenarios/pin_purchase.yaml` — a rendered keypad is
  decoded by template OCR, a PIN is planned and entered, and the result screen
  reads back APPROVED from the re-rendered frame.
- **M2 — Kinematic gantry. DONE.** Toolhead moves along rails; toolcam renders
  from real poses; motion paths collision-checked against the scene. Proof:
  `python -m steropes.scenario scenarios/gantry_moves.yaml scenarios/collision_guard.yaml`
  — the toolhead tracks waypoints (including over the terminal) with toolcam
  frames from each pose and no contact, then a deliberate intercept path trips
  the contact guard.
- **M3 — Contacts. DONE.** The finger physically taps the DUT screen: a
  pogo-pin style compliant plunger (Z slide joint, spring-damper) descends
  onto the rendered keypad, and the contact point and peak force come from
  the physics engine, not from the commanded coordinates — off-target taps
  register as wrong cells or misses, and every tap is checked against a
  1.5–6 N force budget. Proof:
  `python -m steropes.scenario scenarios/pin_purchase_physical.yaml` — the
  M1 flagship done for real: READY → keypad OCR → planned PIN tapped
  physically (~3.3 N peak) → re-rendered screen reads APPROVED. (Card
  grip/insert/eject is out of scope for this DUT.)
- **M4 — Closed loop (first half). DONE.** A Moonraker-emulating HTTP server
  sits in front of the physics backend and an Android-phone DUT (lock screen
  → swipe-up → PIN pad → launcher with tappable, text-labeled apps) lives on
  the deck. The macro contract — `HOME_ALL`, `TOOLS_UP`, `FINGER_TAP X Y`,
  `FINGER_SWIPE X1 Y1 X2 Y2 T`, `FINGER_LONG_PRESS X Y T`,
  `BUTTON_PRESS PLUNGER`, `PARK` — executes against the physics (compliant
  finger, contact-routed); status reports strict Klipper states; overhead and
  toolcam frames are served as PNG snapshots and MJPEG streams. Proof:
  `python -m steropes.server --profile profiles/android_phone_v1.yaml --port 7125`
  in one shell, then the host stack's own `smoke_wake_unlock` scenario,
  UNMODIFIED, passes against it (calibration RMS 0.161 mm, "Home" read off
  the rectified rendered screen). The same flow runs without the server:
  `python -m steropes.scenario scenarios/phone_wake_unlock.yaml`. Remaining
  for M4's second half: CI tier + drift guards against the reference
  machine's CAD and firmware configuration.

## The CAD pipeline

Machine parts are authored as parametric OpenSCAD files in
`cad/openscad/`, with shared dimensions in `cad_config.scad`. Each file
declares a `PART = "preview";` selector near the top and a guarded
export branch at the bottom:

```scad
include <cad_config.scad>
PART = "preview";

module my_part() { /* ... */ }

if (PART == "preview" || PART == "my_part")
    my_part();
```

`cad/manifest.yaml` maps each part to its scad file, `PART` selector,
role, target MJCF body, in-body offset, and colour. To add a part: write
the `.scad` file, add a `parts:` entry to the manifest, and pass
`cad_manifest="cad/manifest.yaml"` to `DeckScene` —
`steropes.cadimport` exports a binary STL per part to `cad/stl/`
(gitignored; re-exported only when the source or `cad_config.scad`
changes) and injects the meshes into the toolhead bodies.

The OpenSCAD binary is found via the `OPENSCAD` environment variable,
falling back to `C:\Program Files\OpenSCAD\openscad.com`.

**Visual vs. collision.** Imported meshes are *visual only*
(`contype="0" conaffinity="0"`). Collision stays on the simplified MJCF
primitives authored in `steropes/gantry.py` — so contact behaviour, the
collision guard, and every scenario are identical whether or not the CAD
meshes are attached. A part's `replaces:` field names the primitive geom
it visually doubles up; the scene hides that primitive (alpha 0) without
touching its collision flags. Only `role: visual` exists; collision
meshes are deliberately unsupported.

**Drift guard.** `constants:` in the manifest mirrors both
`cad_config.scad` and the collision dimensions in `steropes/gantry.py`
(carriage footprint, finger tip diameter, finger/toolcam offsets).
`tests/test_cadimport.py` asserts all three agree, and that the scene
reads mesh placement from the manifest — change a dimension in one place
and the tests tell you about the others.

## Reality gap — an honest note

A physics sim is **design validation, not hardware validation**. Friction
coefficients and contact parameters are wrong until measured on the physical
machine; results like "insertion succeeds at 8 N" are evidence, not proof. FEM-level
questions (printed flexures) and signal-level questions (NFC, magstripe) are out of
scope by design.

## Status

Pre-alpha. M0 (OpenSCAD → STL → visual-only MJCF mesh pipeline, manifest-
driven with a CAD/collision drift guard), M1 (static scene + cameras +
vision loop), M2 (kinematic gantry + collision guard), M3 (physical
screen taps with compliant finger, contact-derived registration, and
per-tap force budget), and the first half of M4 (Moonraker-compatible
HTTP server + Android-phone DUT; an unmodified Klipper-style host stack
runs its wake→unlock scenario against the twin) are complete and covered
by scenario tests; the package is importable and `pytest` is green (80
tests). Everything past that is design scaffold — see the roadmap above.

## License

[MIT](LICENSE) — do what you want, attribution appreciated.

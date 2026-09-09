"""M0: OpenSCAD geometry import pipeline.

Machine parts authored in OpenSCAD (``cad/openscad/``) become visual
geometry in the physics scene, driven by ``cad/manifest.yaml``:

1. The manifest maps part name -> scad file, ``PART`` selector, role,
   target MJCF body, and in-body offset (see the manifest header for the
   format). A manifest may also set ``openscad_dir`` (with a
   ``${ANDROIDTESTER_ROOT}`` placeholder, default ``../AndroidTester``) to
   source parts from the machine repository itself — see
   ``cad/manifest.androidtester.yaml`` — and ``shared_config`` to name the
   shared include the STL cache is invalidated against.
2. :func:`export_part` / :func:`export_all` run the OpenSCAD CLI to
   produce one ASCII STL per part under ``cad/stl/`` (gitignored).
   Export is cached: a part is re-exported only when its scad source (or
   the shared ``cad_config.scad``) is newer than the STL.
3. :func:`scene_fragments` turns the manifest into MJCF snippets — mesh
   assets (STL is mm, so ``scale="0.001 0.001 0.001"``) plus geoms
   grouped by target body — which :class:`steropes.scene.DeckScene`
   injects into the toolhead.

Convention: imported meshes are VISUAL ONLY (``contype=0
conaffinity=0``). Collision stays on the simplified primitives authored
in :mod:`steropes.gantry`; a part's ``replaces`` field names the
primitive geom it visually doubles up, and the scene hides that
primitive (rgba alpha 0) without touching its collision flags. That is
why every existing scenario behaves identically with or without the CAD
meshes attached.

The OpenSCAD binary is located via the ``OPENSCAD`` environment
variable, falling back to the default Windows install path
(:data:`DEFAULT_OPENSCAD`).
"""
from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

DEFAULT_OPENSCAD = r"C:\Program Files\OpenSCAD\openscad.com"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "cad" / "manifest.yaml"

#: MJCF bodies CAD geoms may attach to (the toolhead's fixed structure).
ATTACHABLE_BODIES = ("toolhead", "tool_finger_body")

#: Supported part roles. Collision meshes are deliberately unsupported:
# collision stays on the MJCF primitives (see module docstring).
VALID_ROLES = ("visual",)

#: Shared include every part sources its dimensions from; a change here
#: invalidates the STL cache for all parts. A manifest may override the
#: filename with a top-level ``shared_config:`` field (the AndroidTester
#: sources use ``00_config.scad``).
SHARED_CONFIG = "cad_config.scad"

#: Environment variable pointing at a checkout of the AndroidTester machine
#: repository; manifests may reference it as ``${ANDROIDTESTER_ROOT}`` in
#: their ``openscad_dir``. Defaults to a sibling checkout.
AT_ROOT_ENV = "ANDROIDTESTER_ROOT"
AT_ROOT_DEFAULT = "../AndroidTester"


class ManifestError(ValueError):
    """Raised for any manifest problem: bad fields, missing files,
    unknown parts, unsupported roles or target bodies."""


@dataclass(frozen=True)
class CadPart:
    """One manifest part entry."""

    name: str
    scad: str                            # filename, relative to cad/openscad/
    part: str                            # PART selector passed with -D
    role: str                            # "visual" (see VALID_ROLES)
    body: str                            # target MJCF body
    offset_mm: tuple[float, float, float]
    color: tuple[float, float, float] = (0.55, 0.55, 0.60)
    replaces: str | None = None          # primitive geom hidden on attach


@dataclass(frozen=True)
class Manifest:
    """Validated cad/manifest.yaml."""

    path: Path
    cad_dir: Path                        # OpenSCAD sources (<manifest>/openscad)
    stl_dir: Path                        # exported STLs (<manifest>/stl)
    constants: dict
    parts: tuple[CadPart, ...]
    shared_config: str = SHARED_CONFIG   # shared include, for cache freshness
    _by_name: dict = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {p.name: p for p in self.parts})

    def part(self, name: str) -> CadPart:
        """Return the named part; raises ManifestError for unknown parts."""
        try:
            return self._by_name[name]
        except KeyError:
            raise ManifestError(
                f"unknown part {name!r}; manifest defines: "
                + ", ".join(sorted(self._by_name))) from None

    def scad_path(self, part: CadPart) -> Path:
        return self.cad_dir / part.scad

    def stl_path(self, part: CadPart) -> Path:
        return self.stl_dir / f"{part.name}.stl"

    def replaced_geoms(self) -> list[str]:
        """Primitive geoms visually doubled up by imported meshes."""
        return [p.replaces for p in self.parts if p.replaces]


# --- loading / validation -------------------------------------------------------

def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> Manifest:
    """Load and validate a manifest YAML.

    Validation is eager: every referenced scad file must exist, every
    part must name a supported role and an attachable body, and required
    fields must be present — problems surface here, not mid-scene-build.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cad_dir = path.parent / "openscad"
    if raw.get("openscad_dir"):
        cad_dir = _expand_openscad_dir(str(raw["openscad_dir"]), path)
        if not cad_dir.is_dir():
            raise ManifestError(
                f"openscad_dir not found: {cad_dir} (set the "
                f"{AT_ROOT_ENV} environment variable to the machine "
                "repository checkout)")
    stl_dir = path.parent / "stl"

    parts = []
    for name, spec in (raw.get("parts") or {}).items():
        for key in ("scad", "part", "role", "body", "offset_mm"):
            if key not in spec:
                raise ManifestError(f"part {name!r}: missing field {key!r}")
        if spec["role"] not in VALID_ROLES:
            raise ManifestError(
                f"part {name!r}: unsupported role {spec['role']!r} "
                f"(valid: {', '.join(VALID_ROLES)}); collision stays on "
                "the MJCF primitives")
        if spec["body"] not in ATTACHABLE_BODIES:
            raise ManifestError(
                f"part {name!r}: unknown target body {spec['body']!r} "
                f"(attachable: {', '.join(ATTACHABLE_BODIES)})")
        if not (cad_dir / spec["scad"]).is_file():
            raise ManifestError(
                f"part {name!r}: scad file not found: "
                f"{cad_dir / spec['scad']}")
        parts.append(CadPart(
            name=str(name), scad=str(spec["scad"]), part=str(spec["part"]),
            role=str(spec["role"]), body=str(spec["body"]),
            offset_mm=tuple(float(v) for v in spec["offset_mm"]),
            color=tuple(float(v) for v in spec.get("color",
                                                  (0.55, 0.55, 0.60))),
            replaces=spec.get("replaces")))
    return Manifest(path=path, cad_dir=cad_dir, stl_dir=stl_dir,
                    constants=dict(raw.get("constants") or {}),
                    parts=tuple(parts),
                    shared_config=str(raw.get("shared_config", SHARED_CONFIG)))


def _expand_openscad_dir(value: str, manifest_path: Path) -> Path:
    """Resolve a manifest's ``openscad_dir``.

    ``${ANDROIDTESTER_ROOT}`` expands from the environment (default:
    :data:`AT_ROOT_DEFAULT`); a relative result resolves against the
    manifest's grandparent directory (the project root when the manifest
    lives in ``cad/``), so ``../AndroidTester`` means the sibling checkout.
    """
    root = os.environ.get(AT_ROOT_ENV, AT_ROOT_DEFAULT)
    expanded = value.replace("${" + AT_ROOT_ENV + "}", root)
    p = Path(expanded)
    if not p.is_absolute():
        p = manifest_path.parent.parent / p
    return p


# --- export ------------------------------------------------------------------------

def find_openscad(env: Mapping[str, str] | None = None) -> str | None:
    """Path to the OpenSCAD CLI binary, or None if not installed.

    The ``OPENSCAD`` environment variable overrides the default install
    location (:data:`DEFAULT_OPENSCAD`).
    """
    env = os.environ if env is None else env
    candidate = env.get("OPENSCAD", DEFAULT_OPENSCAD)
    return candidate if Path(candidate).is_file() else None


def export_command(binary: str, scad_path: Path, part: str,
                   stl_path: Path) -> list[str]:
    """OpenSCAD CLI argv exporting one PART selector to a binary STL.

    Binary because MuJoCo's mesh loader does not read ASCII STL.
    """
    return [binary, "-o", str(stl_path), "--export-format", "binstl",
            "-D", f'PART="{part}"', str(scad_path)]


def _is_fresh(manifest: Manifest, part: CadPart) -> bool:
    """True when the cached STL is newer than every source it derives from."""
    stl = manifest.stl_path(part)
    if not stl.is_file():
        return False
    stl_mtime = stl.stat().st_mtime
    sources = [manifest.scad_path(part), manifest.cad_dir / manifest.shared_config]
    return all(stl_mtime >= src.stat().st_mtime
               for src in sources if src.is_file())


def export_part(manifest: Manifest, name: str, binary: str | None = None,
                runner=subprocess.run, force: bool = False) -> Path:
    """Export one part's STL (cached); returns the STL path.

    ``runner`` is the subprocess entry point (injectable for tests) and
    must raise on failure like ``subprocess.run(..., check=True)``.
    """
    part = manifest.part(name)
    stl = manifest.stl_path(part)
    if not force and _is_fresh(manifest, part):
        return stl
    if binary is None:
        binary = find_openscad()
    if binary is None:
        raise ManifestError(
            "OpenSCAD binary not found; set the OPENSCAD environment "
            "variable to its path")
    manifest.stl_dir.mkdir(parents=True, exist_ok=True)
    cmd = export_command(binary, manifest.scad_path(part), part.part, stl)
    result = runner(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ManifestError(
            f"OpenSCAD export failed for part {name!r}:\n{result.stderr}")
    return stl


def export_all(manifest: Manifest, binary: str | None = None,
               runner=subprocess.run) -> dict[str, Path]:
    """Export every manifest part (cached); returns name -> STL path."""
    return {p.name: export_part(manifest, p.name, binary=binary,
                                runner=runner)
            for p in manifest.parts}


# --- MJCF fragments -----------------------------------------------------------------

@dataclass(frozen=True)
class SceneFragments:
    """MJCF snippets attaching a manifest's meshes to the scene."""

    assets: str                          # <mesh> lines for <asset>
    body_geoms: dict[str, str]           # body name -> <geom> lines
    hides: tuple[str, ...]               # primitive geoms to make invisible


def scene_fragments(manifest: Manifest,
                    stls: dict[str, Path] | None = None) -> SceneFragments:
    """Build the MJCF assets/geoms for a manifest's exported meshes.

    Geoms are visual-only (``contype=0 conaffinity=0``); STL units are
    millimetres, so assets carry ``scale="0.001 0.001 0.001"``. Geom
    positions come from the manifest's per-part ``offset_mm`` — the scene
    reads placement from the manifest, and the drift guard
    (tests/test_cadimport.py) pins the manifest to the gantry constants
    the collision model uses.
    """
    if stls is None:
        stls = {p.name: manifest.stl_path(p) for p in manifest.parts}
    assets, geoms = [], {b: [] for b in ATTACHABLE_BODIES}
    for part in manifest.parts:
        mesh = f"cad_{part.name}"
        stl = stls[part.name].as_posix()
        assets.append(
            f'    <mesh name="{mesh}" file="{stl}"'
            f' scale="0.001 0.001 0.001"/>')
        ox, oy, oz = (v / 1000.0 for v in part.offset_mm)
        r, g, b = part.color
        geoms[part.body].append(
            f'      <geom name="{mesh}" type="mesh" mesh="{mesh}"'
            f' pos="{ox:.4f} {oy:.4f} {oz:.4f}"'
            f' rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>')
    return SceneFragments(
        assets="\n".join(assets),
        body_geoms={b: "\n".join(lines) for b, lines in geoms.items()},
        hides=tuple(manifest.replaced_geoms()))


# --- STL inspection (drift guard / tests) ---------------------------------------------

_FACE_DTYPE = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)),
                        ("attr", "<u2")])


@dataclass(frozen=True)
class StlInfo:
    """Facts about a binary STL, read directly (no STL library needed —
    MuJoCo loads the mesh natively in the scene)."""

    header: str                          # 80-byte header, ascii-decoded
    face_count: int
    mins: tuple[float, ...]              # per-axis minima, native units (mm)
    maxs: tuple[float, ...]


def inspect_stl(stl_path: str | Path) -> StlInfo:
    """Parse a binary STL's header, face count, and per-axis bounds.

    Raises ManifestError for malformed or ASCII files (the export uses
    ``--export-format binstl`` precisely because MuJoCo rejects ASCII).
    """
    data = Path(stl_path).read_bytes()
    if len(data) < 84:
        raise ManifestError(f"too small to be a binary STL: {stl_path}")
    (count,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * count or count < 1:
        raise ManifestError(
            f"malformed STL (header says {count} faces, file has room for "
            f"{(len(data) - 84) // 50}): {stl_path}")
    header = data[:80].decode("ascii", errors="replace").rstrip("\x00").strip()
    faces = np.frombuffer(data, dtype=_FACE_DTYPE, count=count, offset=84)
    verts = faces["verts"].reshape(-1, 3)
    return StlInfo(header=header, face_count=int(count),
                   mins=tuple(verts.min(axis=0)),
                   maxs=tuple(verts.max(axis=0)))

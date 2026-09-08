"""MuJoCo scene: deck, fiducial markers, DUT body, dynamic screen texture.

Builds a programmatic MJCF for a static top-down scene — deck plane, four
flat ArUco marker tiles at their deck coordinates, a flush terminal body with
a textured screen quad — and renders it from an overhead pinhole camera.

Frame convention: the physical straight-down camera shows deck +Y as
image-up; rendered frames are flipped vertically on read-out so pixel_y grows
with deck +Y (the deck image convention). Marker textures are pre-flipped so
the ArUco codes appear unmirrored in the final frame. The vision pipeline is
convention-agnostic (registration flows through the solved homography), but a
fixed convention keeps saved frames directly comparable across runs.

Texture updates: the DUT screen is a model texture whose pixels are rewritten
(:meth:`DeckScene.set_screen`) and re-uploaded to the GL context whenever the
screen redraws.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from . import deck as deck_const

# Flat quad mesh (2x2 m, UV-mapped) used for markers and the screen; MuJoCo's
# built-in box texture mapping does not span a face, so UVs are explicit.
# v=1 is texture row 0 (top). Local +Y -> v=1, local -X -> u=0.
QUAD_OBJ = """\
v -1 -1 0
v 1 -1 0
v 1 1 0
v -1 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


# --- terminal profile ------------------------------------------------------------

@dataclass(frozen=True)
class TerminalProfile:
    """DUT geometry on the deck, from a YAML profile.

    ``screen_polygon_mm`` holds the screen corners in deck mm in the order
    top-left, top-right, bottom-right, bottom-left (as seen in the deck image
    convention).
    """

    name: str
    body_center_mm: tuple[float, float]
    body_footprint_mm: tuple[float, float]
    screen_polygon_mm: list[tuple[float, float]]
    screen_canonical_px: tuple[int, int]
    keypad_cols: int
    keypad_rows: int
    pin: str


def load_profile(path: str | Path) -> TerminalProfile:
    """Load a terminal profile YAML (see ``profiles/`` for an example)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TerminalProfile(
        name=raw["name"],
        body_center_mm=tuple(raw["body"]["center_mm"]),
        body_footprint_mm=tuple(raw["body"]["footprint_mm"]),
        screen_polygon_mm=[tuple(p) for p in raw["screen"]["polygon_mm"]],
        screen_canonical_px=tuple(raw["screen"]["canonical_px"]),
        keypad_cols=int(raw["keypad"]["cols"]),
        keypad_rows=int(raw["keypad"]["rows"]),
        pin=str(raw["pin"]),
    )


# --- marker tiles -----------------------------------------------------------------

def marker_tile(marker_id: int) -> np.ndarray:
    """Grayscale marker tile (quiet zone + ArUco code), row 0 = tile top."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, deck_const.ARUCO_DICTIONARY))
    code_px = int(deck_const.MARKER_TEX_PX * deck_const.MARKER_CODE_MM
                  / deck_const.MARKER_TILE_MM)
    code = cv2.aruco.generateImageMarker(dictionary, marker_id, code_px)
    border = (deck_const.MARKER_TEX_PX - code_px) // 2
    tile = np.full((deck_const.MARKER_TEX_PX, deck_const.MARKER_TEX_PX),
                   255, np.uint8)
    tile[border:border + code_px, border:border + code_px] = code
    return tile


# --- scene -------------------------------------------------------------------------

class DeckScene:
    """Static deck scene with an overhead camera and a live screen texture."""

    def __init__(self, profile: TerminalProfile,
                 screen_shape: tuple[int, int],
                 workdir: str | Path,
                 camera: deck_const.OverheadCamera = deck_const.OVERHEAD_CAMERA
                 ) -> None:
        """``screen_shape`` is (height, width) of the screen framebuffer."""
        self.profile = profile
        self.camera = camera
        self._workdir = Path(workdir)
        self._workdir.mkdir(parents=True, exist_ok=True)
        (self._workdir / "flat_quad.obj").write_text(QUAD_OBJ, encoding="utf-8")

        xml = self._build_xml(screen_shape)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._tex_ids = [
            self._write_texture("screen", np.zeros(screen_shape, np.uint8))]
        for mid in deck_const.DECK_MARKERS:
            # vflip: the global frame v-flip must not mirror the codes
            self._tex_ids.append(self._write_texture(
                f"marker_{mid}", marker_tile(mid), vflip=True))

        self._renderer = mujoco.Renderer(
            self.model, height=camera.height_px, width=camera.width_px)
        self._upload_textures()

    # -- MJCF -----------------------------------------------------------------

    def _build_xml(self, screen_shape: tuple[int, int]) -> str:
        ch, cw = screen_shape
        cam = self.camera
        quad_path = (self._workdir / "flat_quad.obj").as_posix()

        # Screen quad from the profile polygon (deck mm, order TL TR BR BL).
        # NOTE: the quad sits in the deck plane (z ~ 1 mm), not at a raised
        # bezel height: the vision path rectifies the screen with the
        # deck-plane homography, and an overhead camera hundreds of mm away
        # would see a raised screen with enough parallax to shift glyphs and
        # break template OCR. Height-aware registration is a later milestone.
        xs = [p[0] for p in self.profile.screen_polygon_mm]
        ys = [p[1] for p in self.profile.screen_polygon_mm]
        scx, scy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        shx, shy = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
        scz = 1.5  # mm; see note above

        bcx, bcy = self.profile.body_center_mm
        bhx, bhy = (self.profile.body_footprint_mm[0] / 2,
                    self.profile.body_footprint_mm[1] / 2)

        deck_w, deck_d = deck_const.DECK_WIDTH_MM, deck_const.DECK_DEPTH_MM
        tile_m = deck_const.MARKER_TILE_MM / 2000.0

        marker_assets = "\n".join(
            f'    <texture name="marker_{mid}" type="2d" builtin="flat"'
            f' rgb1="1 1 1" width="{deck_const.MARKER_TEX_PX}"'
            f' height="{deck_const.MARKER_TEX_PX}"/>\n'
            f'    <material name="marker_{mid}" texture="marker_{mid}"'
            f' emission="1" specular="0" shininess="0"/>'
            for mid in deck_const.DECK_MARKERS)
        marker_geoms = "\n".join(
            f'    <geom name="marker_{mid}" type="mesh" mesh="marker_quad"'
            f' pos="{x / 1000:.4f} {y / 1000:.4f} 0.0005"'
            f' material="marker_{mid}" contype="0" conaffinity="0"/>'
            for mid, (x, y) in deck_const.DECK_MARKERS.items())

        return f"""
<mujoco>
  <visual><global offwidth="{cam.width_px}" offheight="{cam.height_px}"/></visual>
  <asset>
    <texture name="screen" type="2d" builtin="flat" rgb1="0.1 0.1 0.1"
             width="{cw}" height="{ch}"/>
    <material name="screen" texture="screen" emission="1" specular="0" shininess="0"/>
{marker_assets}
    <mesh name="marker_quad" file="{quad_path}" inertia="shell"
          scale="{tile_m:.5f} {tile_m:.5f} 1"/>
    <mesh name="screen_quad" file="{quad_path}" inertia="shell"
          scale="{shx / 1000:.5f} {shy / 1000:.5f} 1"/>
  </asset>
  <worldbody>
    <geom name="deck" type="box"
          size="{deck_w / 2000:.4f} {deck_d / 2000:.4f} 0.005"
          pos="{deck_w / 2000:.4f} {deck_d / 2000:.4f} -0.005"
          rgba="0.216 0.216 0.216 1"/>
    <geom name="terminal" type="box"
          size="{bhx / 1000:.4f} {bhy / 1000:.4f} 0.0005"
          pos="{bcx / 1000:.4f} {bcy / 1000:.4f} 0.0005"
          rgba="0.25 0.25 0.28 1"/>
    <geom name="screen" type="mesh" mesh="screen_quad"
          pos="{scx / 1000:.4f} {scy / 1000:.4f} {scz / 1000:.5f}"
          material="screen" contype="0" conaffinity="0"/>
{marker_geoms}
    <camera name="overhead"
            pos="{deck_w / 2000:.4f} {deck_d / 2000:.4f} {cam.height_mm / 1000:.3f}"
            xyaxes="1 0 0 0 1 0" fovy="{cam.fovy_deg:.4f}"/>
  </worldbody>
</mujoco>
"""

    # -- textures --------------------------------------------------------------

    def _write_texture(self, name: str, img: np.ndarray,
                       vflip: bool = False) -> int:
        """Copy a grayscale or BGR image into a model texture's RGB data.

        tex_data layout is row-major, row 0 = texture top (v=1). Returns the
        texture id; the GL upload happens in _upload_textures().
        """
        tex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, name)
        w, h = self.model.tex_width[tex_id], self.model.tex_height[tex_id]
        if img.ndim == 2:
            rgb = np.repeat(img[:, :, None], 3, axis=2)
        else:
            rgb = img[:, :, ::-1]  # BGR -> RGB
        if vflip:
            rgb = rgb[::-1]
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        adr = self.model.tex_adr[tex_id]
        self.model.tex_data[adr:adr + w * h * 3] = rgb.reshape(-1)
        return tex_id

    def _upload_textures(self) -> None:
        # NOTE: uses the Renderer's private mjrContext; there is no public handle.
        for tex_id in self._tex_ids:
            mujoco.mjr_uploadTexture(self.model, self._renderer._mjr_context,
                                     tex_id)

    def set_screen(self, canvas_bgr: np.ndarray) -> None:
        """Replace the DUT screen texture and re-upload it to the GL context."""
        self._write_texture("screen", canvas_bgr)
        self._upload_textures()

    # -- rendering ---------------------------------------------------------------

    def render_overhead(self) -> np.ndarray:
        """Render the overhead camera; returns a BGR frame (v-flip applied)."""
        self._renderer.update_scene(self.data, camera="overhead")
        frame_rgb = self._renderer.render()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)[::-1].copy()

    def close(self) -> None:
        self._renderer.close()

"""MuJoCo scene: deck, fiducial markers, DUT body, screen texture, gantry.

Builds a programmatic MJCF for the top-down scene — deck plane, four
flat ArUco marker tiles at their deck coordinates, a flush terminal body with
a textured screen quad plus a raised back strip (collision hazard), and the
kinematic M2 toolhead with its own downward camera (see
:mod:`steropes.gantry`). Renders from an overhead pinhole camera or from the
toolhead camera.

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
from . import gantry

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
    """Deck scene with overhead and toolhead cameras and a live screen texture.

    Carries the kinematic M2 gantry: the toolhead moves in deck XY via two
    slide joints (:meth:`move_toolhead`), and contacts between the toolhead
    geoms and the scene are recorded on every move (:attr:`collision_pairs`).
    """

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
        self._tool_renderer: mujoco.Renderer | None = None  # lazy, see render_toolcam
        self._upload_textures()

        # Kinematic gantry state (M2): slide-joint qpos addresses + contact log.
        self._qx = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tool_x")]
        self._qy = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tool_y")]
        self._move_contacts: list[tuple[str, str]] = []

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

        # Collision hazard (M2): the terminal's raised back strip (printer
        # hump). The flat body geom above is visual-only 1 mm; the hump
        # reaches the toolhead/finger envelope, so a scripted XY path can
        # genuinely intercept the terminal while cruise travel over the flat
        # region stays clear. Full footprint width, 25 mm deep, +Y edge.
        riser_h = gantry.TERMINAL_RISER_HEIGHT_MM
        riser_hy = 12.5
        riser_cy = bcy + bhy - riser_hy

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
    <geom name="terminal_riser" type="box"
          size="{bhx / 1000:.4f} {riser_hy / 1000:.4f} {riser_h / 2000:.4f}"
          pos="{bcx / 1000:.4f} {riser_cy / 1000:.4f} {riser_h / 2000:.4f}"
          rgba="0.2 0.2 0.24 1"/>
{marker_geoms}
{gantry.toolhead_xml()}
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
        contexts = [self._renderer._mjr_context]
        if self._tool_renderer is not None:
            contexts.append(self._tool_renderer._mjr_context)
        for tex_id in self._tex_ids:
            for ctx in contexts:
                mujoco.mjr_uploadTexture(self.model, ctx, tex_id)

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

    def render_toolcam(self) -> np.ndarray:
        """Render the toolhead camera from its current pose (BGR, v-flipped)."""
        if self._tool_renderer is None:
            cam = gantry.TOOL_CAMERA
            self._tool_renderer = mujoco.Renderer(
                self.model, height=cam.height_px, width=cam.width_px)
            self._upload_textures()  # the new context needs the textures too
        self._tool_renderer.update_scene(self.data, camera="toolcam")
        frame_rgb = self._tool_renderer.render()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)[::-1].copy()

    # -- gantry (M2) -----------------------------------------------------------

    @property
    def toolhead_position_mm(self) -> tuple[float, float]:
        """Toolhead carriage centre in deck mm."""
        return (float(self.data.qpos[self._qx]) * 1000.0,
                float(self.data.qpos[self._qy]) * 1000.0)

    def _toolhead_contacts(self) -> list[tuple[str, str]]:
        """Geom-name pairs currently in contact that involve the toolhead."""
        pairs = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            names = tuple(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                for g in (con.geom1, con.geom2))
            if any(n.startswith("tool_") for n in names):
                pairs.append(tuple(sorted(names)))
        return pairs

    @property
    def collision_pairs(self) -> list[tuple[str, str]]:
        """Unique toolhead contacts seen during the last move plus right now."""
        return sorted(set(self._move_contacts)
                      | set(self._toolhead_contacts()))

    def move_toolhead(self, x: float, y: float,
                      speed_mm_s: float = gantry.DEFAULT_SPEED_MM_S,
                      dt_s: float = gantry.DEFAULT_DT_S) -> None:
        """Kinematically drive the toolhead to deck (x, y) mm.

        Interpolates at fixed dt (deterministic), stepping the slide joints
        and re-running contact evaluation at every waypoint; contacts found
        along the path are kept in :attr:`collision_pairs` until the next
        move. Out-of-travel targets raise ValueError — see
        :func:`steropes.gantry.clamp_to_deck` for the clamping variant.
        """
        gantry.check_deck_limits(x, y)
        self._move_contacts = []
        start = self.toolhead_position_mm
        for px, py in gantry.interpolate(start, (x, y), speed_mm_s, dt_s):
            self.data.qpos[self._qx] = px / 1000.0
            self.data.qpos[self._qy] = py / 1000.0
            mujoco.mj_forward(self.model, self.data)
            self._move_contacts.extend(self._toolhead_contacts())

    def close(self) -> None:
        self._renderer.close()
        if self._tool_renderer is not None:
            self._tool_renderer.close()

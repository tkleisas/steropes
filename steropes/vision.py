"""Vision pipeline: fiducial detection, planar registration, screen OCR.

Everything here is standard computer vision implemented from first principles
(numpy + OpenCV only):

- ArUco marker detection (version-tolerant across OpenCV APIs).
- Homography solved from four point correspondences with the normalized DLT.
- Perspective rectification of the DUT screen to a canonical canvas.
- Template-matching OCR: digit glyphs and short status strings are compared
  against freshly rendered reference bitmaps with normalized cross-correlation.

Frames are BGR ``uint8`` in the deck image convention (deck +Y is image-up;
see :mod:`steropes.deck`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- ArUco detection -----------------------------------------------------------

def detect_markers(frame_bgr: np.ndarray,
                   dictionary_name: str = "DICT_4X4_50") -> dict[int, tuple[float, float]]:
    """Detect ArUco markers; returns marker id -> centre (x, y) in pixels."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7 API
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # pragma: no cover - legacy OpenCV API
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return {}
    return {int(i): tuple(map(float, c.reshape(4, 2).mean(axis=0)))
            for c, i in zip(corners, ids.flatten())}


# --- homography ------------------------------------------------------------------

def solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping ``src`` points to ``dst`` points (normalized DLT).

    ``src`` and ``dst`` are (N, 2) arrays with N >= 4 corresponding points.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("need >= 4 corresponding (N, 2) point pairs")

    def norm(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = pts.mean(axis=0)
        scale = np.sqrt(2.0) / np.linalg.norm(pts - mean, axis=1).mean()
        t = np.array([[scale, 0, -scale * mean[0]],
                      [0, scale, -scale * mean[1]],
                      [0, 0, 1.0]])
        return t, (pts - mean) * scale

    t_src, s = norm(src)
    t_dst, d = norm(dst)

    a = []
    for (x, y), (u, v) in zip(s, d):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(a))
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h @ t_src
    return h / h[2, 2]


def _apply(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = h @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


@dataclass
class Homography:
    """Planar registration between deck millimetres and image pixels."""

    deck_to_px: np.ndarray  # 3x3, deck mm -> pixel

    def deck_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return _apply(self.deck_to_px, x_mm, y_mm)

    def pixel_to_deck(self, u: float, v: float) -> tuple[float, float]:
        return _apply(np.linalg.inv(self.deck_to_px), u, v)

    def rms_mm(self, deck_points: dict[int, tuple[float, float]],
               pixel_points: dict[int, tuple[float, float]]) -> float:
        """RMS reprojection error in deck mm over shared marker ids."""
        errs = []
        for mid in sorted(set(deck_points) & set(pixel_points)):
            u, v = self.deck_to_pixel(*deck_points[mid])
            dx, dy = self.pixel_to_deck(u, v)
            errs.append((dx - deck_points[mid][0]) ** 2
                        + (dy - deck_points[mid][1]) ** 2)
        if not errs:
            raise ValueError("no shared marker ids")
        return float(np.sqrt(np.mean(errs)))


def calibrate(frame_bgr: np.ndarray,
              deck_markers: dict[int, tuple[float, float]],
              dictionary_name: str = "DICT_4X4_50"
              ) -> tuple[Homography, dict[int, tuple[float, float]], float]:
    """Detect deck markers and solve the deck->pixel homography.

    Returns ``(homography, detected_centres_px, rms_error_mm)``.
    """
    detected = detect_markers(frame_bgr, dictionary_name)
    shared = sorted(set(deck_markers) & set(detected))
    if len(shared) < 4:
        raise RuntimeError(
            f"need 4 deck markers, detected ids {sorted(detected)}")
    src = np.array([deck_markers[i] for i in shared])
    dst = np.array([detected[i] for i in shared])
    hom = Homography(deck_to_px=solve_homography(src, dst))
    return hom, detected, hom.rms_mm(deck_markers, detected)


# --- screen rectification ---------------------------------------------------------

def rectify(frame_bgr: np.ndarray, polygon_px: list[tuple[float, float]],
            size_px: tuple[int, int]) -> np.ndarray:
    """Warp a screen quad (pixels, order TL TR BR BL) to a canonical canvas."""
    if len(polygon_px) != 4:
        raise ValueError("screen polygon must have 4 corners")
    w, h = size_px
    dst = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
                   dtype=np.float64)
    h_mat = solve_homography(np.asarray(polygon_px, dtype=np.float64), dst)
    return cv2.warpPerspective(frame_bgr, h_mat, (w, h))


def split_grid(image: np.ndarray, cols: int, rows: int) -> list[np.ndarray]:
    """Split an image into a cols x rows grid of cells (row-major order)."""
    h, w = image.shape[:2]
    return [image[r * h // rows:(r + 1) * h // rows,
                  c * w // cols:(c + 1) * w // cols]
            for r in range(rows) for c in range(cols)]


# --- template-matching OCR --------------------------------------------------------

_GLYPH_H = 64          # normalised glyph/line height in pixels
_BRIGHT_THRESHOLD = 120  # grey level separating glyphs from the dark screen


def _content_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of bright pixels, or None if the cell is empty."""
    mask = gray > _BRIGHT_THRESHOLD
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _normalize(gray: np.ndarray, canvas_w: int) -> np.ndarray | None:
    """Crop bright content and centre it on a fixed canvas, scaled to glyph height.

    Returns float32 in [0, 1], or None when there is no bright content.
    """
    bbox = _content_bbox(gray)
    if bbox is None:
        return None
    x, y, w, h = bbox
    crop = gray[y:y + h, x:x + w].astype(np.float32) / 255.0
    scale = _GLYPH_H / h
    if w * scale > canvas_w:  # very wide content: fit width instead
        scale = canvas_w / w
    resized = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_GLYPH_H, canvas_w), np.float32)
    ox = (canvas_w - resized.shape[1]) // 2
    oy = (_GLYPH_H - resized.shape[0]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return canvas


def _normalize_line(gray: np.ndarray, canvas_w: int) -> np.ndarray | None:
    """Tight-crop bright content and resize it to fill the canvas exactly.

    Used for multi-character status lines: capture and candidate templates go
    through the identical tight-crop + resize, so any small anisotropic
    stretch from the render/rectify path cancels out instead of accumulating
    across the line.
    """
    bbox = _content_bbox(gray)
    if bbox is None:
        return None
    x, y, w, h = bbox
    crop = gray[y:y + h, x:x + w].astype(np.float32) / 255.0
    return cv2.resize(crop, (canvas_w, _GLYPH_H), interpolation=cv2.INTER_AREA)


def _render_text_bitmap(text: str, canvas_w: int, scale: float = 1.6,
                        thickness: int = 3, for_line: bool = False
                        ) -> np.ndarray:
    """Render text with cv2.putText and normalise it like a screen capture."""
    scratch = np.full((_GLYPH_H * 3, canvas_w * 2, 3), 24, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    org = ((scratch.shape[1] - tw) // 2, (scratch.shape[0] + th) // 2)
    cv2.putText(scratch, text, org, FONT, scale, (225, 225, 225),
                thickness, cv2.LINE_AA)
    gray = cv2.cvtColor(scratch, cv2.COLOR_BGR2GRAY)
    out = _normalize_line(gray, canvas_w) if for_line else _normalize(gray, canvas_w)
    assert out is not None
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two same-size float images."""
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


@dataclass
class KeypadReading:
    """OCR result for one keypad cell."""

    cell: int          # row-major cell index
    digit: str         # "0"-"9", or "" for a blank key
    confidence: float  # NCC score of the best-matching template


class DigitOcr:
    """Template-matching digit reader for light-on-dark keypad cells."""

    _CANVAS_W = 64

    def __init__(self) -> None:
        self._templates = {
            d: _render_text_bitmap(d, self._CANVAS_W) for d in "0123456789"
        }

    def read_cell(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        """Classify one cell image; returns (digit, confidence).

        Empty cells return ``("", best_score)`` where best_score is low.
        """
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
        norm = _normalize(gray, self._CANVAS_W)
        if norm is None:
            return "", 0.0
        scores = {d: _ncc(norm, t) for d, t in self._templates.items()}
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best, scores[best]


def read_keypad(cells: list[np.ndarray],
                ocr: DigitOcr | None = None) -> list[KeypadReading]:
    """OCR a full keypad grid (cells in row-major order)."""
    ocr = ocr or DigitOcr()
    return [KeypadReading(cell=i, digit=d, confidence=c)
            for i, (d, c) in enumerate(ocr.read_cell(img) for img in cells)]


def plan_pin_taps(readings: list[KeypadReading], pin: str) -> list[int]:
    """Map a PIN to keypad cell indices using OCR readings.

    Raises ValueError if a needed digit was not read, or appears on the
    keypad more than once (ambiguous tap target).
    """
    digit_cells: dict[str, list[int]] = {}
    for r in readings:
        if r.digit:
            digit_cells.setdefault(r.digit, []).append(r.cell)
    duplicates = {d: cs for d, cs in digit_cells.items() if len(cs) > 1}
    if duplicates:
        raise ValueError(f"ambiguous keypad: digits read more than once: {duplicates}")
    missing = [ch for ch in pin if ch not in digit_cells]
    if missing:
        raise ValueError(f"PIN digits not found on keypad: {missing}")
    return [digit_cells[ch][0] for ch in pin]


# --- status-line reading --------------------------------------------------------

_LINE_CANVAS_W = 512


def match_text_line(image_bgr: np.ndarray,
                    candidates: list[str]) -> tuple[str, float]:
    """Classify a full-screen status line against candidate strings.

    Renders each candidate like a reference screen and picks the highest
    normalized cross-correlation. Returns (best_candidate, score).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = _normalize_line(gray, _LINE_CANVAS_W)
    if norm is None:
        raise ValueError("no text found in image")
    scores = {text: _ncc(norm, _render_text_bitmap(text, _LINE_CANVAS_W,
                                                   scale=1.1, thickness=3,
                                                   for_line=True))
              for text in candidates}
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best, scores[best]


def text_score(image_bgr: np.ndarray, text: str, scale: float = 2.0,
               thickness: int = 3) -> float:
    """Best NCC score of a putText-rendered ``text`` template over the image.

    Same recipe as the host stack's template fallback (white text on black,
    ``TM_CCOEFF_NORMED`` over the whole frame, defaults scale 2.0 /
    thickness 3), so a screen that scores well here is legible to the
    unmodified client's ``assert_text``.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 \
        else image_bgr
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    template = np.zeros((th + baseline + 8, tw + 8), np.uint8)
    cv2.putText(template, text, (4, th + 2), FONT, scale, 255, thickness,
                cv2.LINE_AA)
    return float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())

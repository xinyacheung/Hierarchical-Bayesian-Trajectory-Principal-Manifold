"""Dependency-light, dual PNG/PDF drawing helpers for the TED analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas


PALETTE = {
    "oa_hb_tpm_map": "#146C94",
    "hb_tpm_latent_mse": "#D1495B",
    "source_mean": "#2A9D8F",
    "target_only": "#7A5195",
    "oa_hb_tpm_isotropic_prior": "#8C8C8C",
    "oa_hb_tpm_no_phase_alignment": "#B5B5B5",
    "oracle": "#E9C46A",
    "ink": "#20262E",
    "muted": "#626B75",
    "grid": "#D9DEE5",
    "paper": "#FFFFFF",
}

DISPLAY_NAMES = {
    "oa_hb_tpm_map": "OA-HB-TPM (MAP)",
    "hb_tpm_latent_mse": "HB-TPM latent-MSE",
    "source_mean": "Source mean",
    "target_only": "Target only",
    "oa_hb_tpm_isotropic_prior": "Isotropic prior",
    "oa_hb_tpm_no_phase_alignment": "No phase alignment",
    "oracle": "Oracle contour",
}


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


class DualCanvas:
    """Draw the same primitives to a raster PNG and a vector PDF."""

    def __init__(self, png_path: Path, pdf_path: Path, width: int = 1800, height: int = 1100):
        self.png_path = Path(png_path)
        self.pdf_path = Path(pdf_path)
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), _hex_rgb(PALETTE["paper"]))
        self.draw = ImageDraw.Draw(self.image)
        self.pdf = pdfcanvas.Canvas(str(self.pdf_path), pagesize=(width, height))
        self._font_regular = Path(r"C:\Windows\Fonts\arial.ttf")
        self._font_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
        try:
            if self._font_regular.exists():
                pdfmetrics.registerFont(TTFont("TEDArial", str(self._font_regular)))
            if self._font_bold.exists():
                pdfmetrics.registerFont(TTFont("TEDArialBold", str(self._font_bold)))
        except Exception:
            pass

    def _pil_font(self, size: int, bold: bool = False):
        path = self._font_bold if bold else self._font_regular
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _pdf_color(value: str):
        r, g, b = _hex_rgb(value)
        return r / 255, g / 255, b / 255

    def line(self, xy: Sequence[float], color: str = PALETTE["ink"], width: int = 2):
        x1, y1, x2, y2 = xy
        self.draw.line((x1, y1, x2, y2), fill=_hex_rgb(color), width=width)
        self.pdf.setStrokeColorRGB(*self._pdf_color(color))
        self.pdf.setLineWidth(width)
        self.pdf.line(x1, self.height - y1, x2, self.height - y2)

    def polyline(self, points: Sequence[tuple[float, float]], color: str, width: int = 4):
        if len(points) < 2:
            return
        self.draw.line(points, fill=_hex_rgb(color), width=width, joint="curve")
        self.pdf.setStrokeColorRGB(*self._pdf_color(color))
        self.pdf.setLineWidth(width)
        path = self.pdf.beginPath()
        path.moveTo(points[0][0], self.height - points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, self.height - y)
        self.pdf.drawPath(path, stroke=1, fill=0)

    def rectangle(
        self,
        xy: Sequence[float],
        fill: str | None = None,
        outline: str | None = None,
        width: int = 2,
    ):
        x1, y1, x2, y2 = xy
        self.draw.rectangle(
            (x1, y1, x2, y2),
            fill=_hex_rgb(fill) if fill else None,
            outline=_hex_rgb(outline) if outline else None,
            width=width,
        )
        if fill:
            self.pdf.setFillColorRGB(*self._pdf_color(fill))
        if outline:
            self.pdf.setStrokeColorRGB(*self._pdf_color(outline))
        self.pdf.setLineWidth(width)
        self.pdf.rect(
            x1,
            self.height - y2,
            x2 - x1,
            y2 - y1,
            stroke=1 if outline else 0,
            fill=1 if fill else 0,
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        outline: str | None = None,
        width: int = 2,
    ):
        self.draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=_hex_rgb(fill),
            outline=_hex_rgb(outline) if outline else None,
            width=width,
        )
        self.pdf.setFillColorRGB(*self._pdf_color(fill))
        if outline:
            self.pdf.setStrokeColorRGB(*self._pdf_color(outline))
        self.pdf.setLineWidth(width)
        self.pdf.circle(x, self.height - y, radius, stroke=1 if outline else 0, fill=1)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 28,
        color: str = PALETTE["ink"],
        bold: bool = False,
        anchor: str = "left",
    ):
        font = self._pil_font(size, bold)
        bbox = self.draw.textbbox((0, 0), value, font=font)
        w = bbox[2] - bbox[0]
        if anchor == "center":
            px = x - w / 2
        elif anchor == "right":
            px = x - w
        else:
            px = x
        self.draw.text((px, y), value, font=font, fill=_hex_rgb(color))
        name = "TEDArialBold" if bold and "TEDArialBold" in pdfmetrics.getRegisteredFontNames() else (
            "TEDArial" if "TEDArial" in pdfmetrics.getRegisteredFontNames() else ("Helvetica-Bold" if bold else "Helvetica")
        )
        self.pdf.setFont(name, size)
        self.pdf.setFillColorRGB(*self._pdf_color(color))
        py = self.height - y - size * 0.82
        if anchor == "center":
            self.pdf.drawCentredString(x, py, value)
        elif anchor == "right":
            self.pdf.drawRightString(x, py, value)
        else:
            self.pdf.drawString(x, py, value)

    def finish(self):
        self.png_path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(self.png_path, dpi=(300, 300), optimize=True)
        self.pdf.showPage()
        self.pdf.save()


def nice_limits(values: Iterable[float], pad: float = 0.08, include_zero: bool = False) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None]
    lo, hi = min(vals), max(vals)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        delta = max(abs(hi) * 0.1, 1.0)
    else:
        delta = (hi - lo) * pad
    return lo - delta, hi + delta


def map_value(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def panel_axes(
    canvas: DualCanvas,
    bounds: tuple[float, float, float, float],
    x_ticks: Sequence[tuple[float, str]],
    y_ticks: Sequence[tuple[float, str]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    title: str,
    x_label: str = "",
    y_label: str = "",
):
    left, top, right, bottom = bounds
    for value, label in y_ticks:
        y = map_value(value, y_limits[0], y_limits[1], bottom, top)
        canvas.line((left, y, right, y), PALETTE["grid"], 2)
        canvas.text(left - 16, y - 14, label, 22, PALETTE["muted"], anchor="right")
    canvas.line((left, top, left, bottom), PALETTE["ink"], 3)
    canvas.line((left, bottom, right, bottom), PALETTE["ink"], 3)
    for value, label in x_ticks:
        x = map_value(value, x_limits[0], x_limits[1], left, right)
        canvas.line((x, bottom, x, bottom + 8), PALETTE["ink"], 2)
        canvas.text(x, bottom + 14, label, 22, PALETTE["muted"], anchor="center")
    canvas.text((left + right) / 2, top - 52, title, 28, bold=True, anchor="center")
    if x_label:
        canvas.text((left + right) / 2, bottom + 56, x_label, 23, PALETTE["muted"], anchor="center")
    if y_label:
        canvas.text(left, top - 18, y_label, 20, PALETTE["muted"])


def legend(canvas: DualCanvas, items: Sequence[str], x: float, y: float, columns: int = 3):
    col_width = 310
    row_height = 38
    for i, key in enumerate(items):
        row, col = divmod(i, columns)
        xx, yy = x + col * col_width, y + row * row_height
        color = PALETTE.get(key, PALETTE["muted"])
        canvas.line((xx, yy + 14, xx + 44, yy + 14), color, 6)
        canvas.circle(xx + 22, yy + 14, 6, color)
        canvas.text(xx + 58, yy, DISPLAY_NAMES.get(key, key), 22)

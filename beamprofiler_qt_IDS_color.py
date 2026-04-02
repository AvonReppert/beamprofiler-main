"""IDS U3-3880LE-C Beamprofiler — True-Color Display

Identical to the monochrome IDS backend except:
- Camera frames are kept as RGB and displayed with full color.
- Beam analysis (FWHM, centroid) still uses the luminance channel.
- Colourmap + log-scale controls are replaced by a brightness/auto-stretch control.
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import xarray as xr

    XRAY_AVAILABLE = True
except Exception:
    xr = None  # type: ignore[assignment]
    XRAY_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.optimize import curve_fit

    MPL_AVAILABLE = True
except Exception:
    plt = None  # type: ignore[assignment]
    MPL_AVAILABLE = False

try:
    from camera import Camera
    from ids_peak import ids_peak
    from ids_peak_common import CommonException
    from ids_peak_icv.pipeline import DefaultPipeline

    IDS_BACKEND_AVAILABLE = True
except Exception as import_error:  # pragma: no cover - depends on local SDK install
    Camera = None  # type: ignore[assignment]
    DefaultPipeline = None  # type: ignore[assignment]
    CommonException = Exception  # type: ignore[assignment]
    ids_peak = None  # type: ignore[assignment]
    IDS_BACKEND_AVAILABLE = False
    IDS_IMPORT_ERROR = import_error
else:
    IDS_IMPORT_ERROR = None


pg.setConfigOptions(antialias=True, background="#101215", foreground="#d8dee9")

PIXEL_SIZE_UM = 2.4
SENSOR_WIDTH_PIXELS = 3088
SENSOR_HEIGHT_PIXELS = 2076
SENSOR_WIDTH_UM = SENSOR_WIDTH_PIXELS * PIXEL_SIZE_UM
SENSOR_HEIGHT_UM = SENSOR_HEIGHT_PIXELS * PIXEL_SIZE_UM

DISPLAY_ROTATE_K = 0

FLUENCE_MODE_FWHM = "fwhm"
FLUENCE_MODE_ONE_OVER_E = "one_over_e"


# ---------------------------------------------------------------------------
# Screen scaling helpers (identical to monochrome version)
# ---------------------------------------------------------------------------

def get_screen_scale_factor() -> float:
    from PySide6.QtGui import QGuiApplication
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return 1.0
    dpi = screen.logicalDotsPerInch()
    geometry = screen.geometry()
    width_px = geometry.width()
    dpi_scale = dpi / 96.0
    width_scale = width_px / 1920.0
    scale = (dpi_scale * width_scale) / 1.5
    return max(0.7, min(scale, 2.0))


def scale_font_size(base_size: int) -> int:
    return max(8, int(round(base_size * get_screen_scale_factor())))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BeamMetrics:
    centroid_x_um: float
    centroid_y_um: float
    fwhm_x_um: float
    fwhm_y_um: float
    peak_value: float
    sum_intensity: float


@dataclass
class FrameData:
    gray_image: np.ndarray          # H×W float32 — used for analysis
    rgb_image: np.ndarray           # H×W×3 uint8  — used for display
    projection_x: np.ndarray
    projection_y: np.ndarray
    x_coordinates_um: np.ndarray
    y_coordinates_um: np.ndarray
    metrics: BeamMetrics
    camera_full_scale_counts: float


@dataclass
class CameraState:
    model_name: str
    serial_number: str
    exposure_ms: float
    exposure_min_ms: float
    exposure_max_ms: float
    gain_value: float
    gain_min: float
    gain_max: float


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def compute_fwhm_1d(profile: np.ndarray) -> float:
    if profile.size == 0:
        return 0.0
    values = profile.astype(np.float64)
    values -= values.min()
    peak = values.max()
    if peak <= 0:
        return 0.0
    half_max = peak / 2.0
    above = np.flatnonzero(values >= half_max)
    if above.size == 0:
        return 0.0
    left = int(above[0])
    right = int(above[-1])
    if left > 0 and values[left] != values[left - 1]:
        left_edge = (left - 1) + (half_max - values[left - 1]) / (values[left] - values[left - 1])
    else:
        left_edge = float(left)
    if right < values.size - 1 and values[right] != values[right + 1]:
        right_edge = right + (half_max - values[right + 1]) / (values[right] - values[right + 1])
    else:
        right_edge = float(right)
    return max(0.0, float(right_edge - left_edge))


def make_axis_um(length: int, pixel_size_um: float) -> np.ndarray:
    return (np.arange(length, dtype=np.float64) - (length - 1) / 2.0) * pixel_size_um


def calc_fluence(
    power_uW: float,
    fwhm_x: float,
    fwhm_y: float,
    angle: float,
    rep_rate: float,
    mode: str = FLUENCE_MODE_FWHM,
) -> float:
    if fwhm_x <= 0 or fwhm_y <= 0 or rep_rate <= 0:
        return 0.0
    angle_rad = np.radians(angle)
    if mode == FLUENCE_MODE_ONE_OVER_E:
        x0 = fwhm_x / (2.0 * np.sqrt(np.log(2.0))) * 1e-4
        y0 = fwhm_y / (2.0 * np.sqrt(np.log(2.0))) * 1e-4
    else:
        x0 = fwhm_x / 2.0 * 1e-4
        y0 = fwhm_y / 2.0 * 1e-4
    area = np.pi * x0 * y0
    fluence = power_uW * np.cos(angle_rad) / (rep_rate * area)
    return float(np.round(fluence, 4))


def fluence_formula_text(mode: str) -> str:
    if mode == FLUENCE_MODE_ONE_OVER_E:
        return (
            "<span style='font-family: Consolas, monospace;'>"
            "x<sub>0</sub> = FWHM<sub>x</sub> / (2&radic;ln 2)<br>"
            "y<sub>0</sub> = FWHM<sub>y</sub> / (2&radic;ln 2)<br>"
            "F = P cos(&theta;) / (f &pi; x<sub>0</sub> y<sub>0</sub>)"
            "</span><br>"
            "Input: µW, Output: µJ/cm²"
        )
    return (
        "<span style='font-family: Consolas, monospace;'>"
        "x<sub>0</sub> = FWHM<sub>x</sub> / 2<br>"
        "y<sub>0</sub> = FWHM<sub>y</sub> / 2<br>"
        "F = P cos(&theta;) / (f &pi; x<sub>0</sub> y<sub>0</sub>)"
        "</span><br>"
        "Input: µW, Output: µJ/cm²"
    )


def compute_metrics(image: np.ndarray, pixel_size_um: float) -> tuple[BeamMetrics, np.ndarray, np.ndarray]:
    projection_x = image.sum(axis=0)
    projection_y = image.sum(axis=1)
    x_coords_um = make_axis_um(image.shape[1], pixel_size_um)
    y_coords_um = make_axis_um(image.shape[0], pixel_size_um)
    sum_x = projection_x.sum()
    sum_y = projection_y.sum()
    centroid_x_um = float((x_coords_um * projection_x).sum() / sum_x) if sum_x > 0 else 0.0
    centroid_y_um = float((y_coords_um * projection_y).sum() / sum_y) if sum_y > 0 else 0.0
    return (
        BeamMetrics(
            centroid_x_um=centroid_x_um,
            centroid_y_um=centroid_y_um,
            fwhm_x_um=compute_fwhm_1d(projection_x) * pixel_size_um,
            fwhm_y_um=compute_fwhm_1d(projection_y) * pixel_size_um,
            peak_value=float(image.max()),
            sum_intensity=float(image.sum()),
        ),
        projection_x,
        projection_y,
    )


def convert_to_gray(image_data: np.ndarray) -> np.ndarray:
    """Convert any array to float32 luminance."""
    if image_data.ndim == 2:
        return image_data.astype(np.float32)
    if image_data.ndim == 3 and image_data.shape[2] >= 3:
        rgb = image_data[..., :3].astype(np.float32)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    return image_data.mean(axis=2, dtype=np.float32)


def build_rgb_uint8(source_image: np.ndarray) -> np.ndarray:
    """Convert camera output to H×W×3 uint8 RGB for display.

    Handles 8-bit, 10-bit, 12-bit, 16-bit integer images and float images.
    If the source is monochrome (2D), the single channel is replicated.
    """
    if source_image.ndim == 2:
        # Monochrome — replicate to RGB
        if np.issubdtype(source_image.dtype, np.integer):
            full_scale = float(np.iinfo(source_image.dtype).max)
        else:
            full_scale = max(1.0, float(source_image.max()))
        gray_u8 = np.clip(source_image.astype(np.float32) / full_scale * 255.0, 0, 255).astype(np.uint8)
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

    # 3-channel (HWC)
    rgb3 = source_image[..., :3]
    if rgb3.dtype == np.uint8:
        return np.ascontiguousarray(rgb3)

    if np.issubdtype(rgb3.dtype, np.integer):
        full_scale = float(np.iinfo(rgb3.dtype).max)
    else:
        full_scale = max(1.0, float(rgb3.max()))

    return np.clip(rgb3.astype(np.float32) / full_scale * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# BeamImageView — color version (no colormap / colorbar)
# ---------------------------------------------------------------------------

class BeamImageView(pg.PlotWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setBackground("#08090b")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=True, y=True, alpha=0.14)
        self.setLabel("bottom", "X", units="µm")
        self.setLabel("left", "Y", units="µm")
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getAxis("left").enableAutoSIPrefix(False)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image_item)

        half_w = SENSOR_WIDTH_UM / 2.0
        half_h = SENSOR_HEIGHT_UM / 2.0
        self.setXRange(-half_w, half_w, padding=0.02)
        self.setYRange(-half_h, half_h, padding=0.02)

        self.horizontal_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#e040fb", width=1)
        )
        self.vertical_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#e040fb", width=1)
        )
        self.addItem(self.horizontal_line)
        self.addItem(self.vertical_line)

        self.slice_h_line = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen("#4dc9f6", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.slice_v_line = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("#4dc9f6", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.slice_h_line.setVisible(False)
        self.slice_v_line.setVisible(False)
        self.addItem(self.slice_h_line)
        self.addItem(self.slice_v_line)

        self.roi_rect = pg.RectROI(
            [-half_w, -half_h], [SENSOR_WIDTH_UM, SENSOR_HEIGHT_UM],
            pen=pg.mkPen("#ffe135", width=2),
            hoverPen=pg.mkPen("#fff176", width=2),
            handlePen=pg.mkPen("#ffe135", width=2),
        )
        self.roi_rect.setVisible(False)
        self.addItem(self.roi_rect)

        self.slice_h_line.setZValue(self.roi_rect.zValue() + 10)
        self.slice_v_line.setZValue(self.roi_rect.zValue() + 10)

        self._crosshair_markers: list[pg.TargetItem] = []

        # No colorbar for true-color display
        self._brightness_scale: float = 1.0

    def set_brightness_scale(self, scale: float) -> None:
        self._brightness_scale = max(0.05, float(scale))

    def set_image(self, frame: FrameData) -> None:
        x_coords_um = frame.x_coordinates_um
        y_coords_um = frame.y_coordinates_um
        pixel_width = PIXEL_SIZE_UM
        pixel_height = PIXEL_SIZE_UM
        x_min = float(x_coords_um[0] - pixel_width / 2.0)
        y_min = float(y_coords_um[0] - pixel_height / 2.0)
        width_um = float(x_coords_um[-1] - x_coords_um[0] + pixel_width)
        height_um = float(y_coords_um[-1] - y_coords_um[0] + pixel_height)

        rgb = frame.rgb_image  # H×W×3 uint8
        # Downsample for display only
        if rgb.shape[0] > 1024 or rgb.shape[1] > 1024:
            step = max(rgb.shape[0] // 1024, rgb.shape[1] // 1024, 1)
            rgb = rgb[::step, ::step, :]

        # Apply brightness scaling
        if self._brightness_scale != 1.0:
            rgb = np.clip(rgb.astype(np.float32) * self._brightness_scale, 0, 255).astype(np.uint8)

        self.image_item.setImage(rgb, autoLevels=False, levels=(0, 255))
        self.image_item.setRect(QtCore.QRectF(x_min, y_min, width_um, height_um))
        self.horizontal_line.setValue(frame.metrics.centroid_y_um)
        self.vertical_line.setValue(frame.metrics.centroid_x_um)


# ---------------------------------------------------------------------------
# ProjectionPlot / TrendPlot (unchanged from monochrome version)
# ---------------------------------------------------------------------------

class ProjectionPlot(pg.PlotWidget):
    def __init__(self, *, vertical: bool = False) -> None:
        super().__init__()
        self.vertical = vertical
        self.setBackground("#15181d")
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.getAxis("left").setPen(pg.mkPen("#5c6773"))
        self.getAxis("bottom").setPen(pg.mkPen("#5c6773"))
        if vertical:
            self.setLabel("left", "Y", units="µm")
            self.setLabel("bottom", "Intensity")
            self.curve = self.plot([], [], pen=pg.mkPen("#ffb347", width=2))
        else:
            self.setLabel("bottom", "X", units="µm")
            self.setLabel("left", "Intensity")
            self.curve = self.plot([], pen=pg.mkPen("#ffb347", width=2))
        self.getAxis("left").enableAutoSIPrefix(False)
        self.getAxis("bottom").enableAutoSIPrefix(False)

    def set_projection(self, axis_coords_um: np.ndarray, projection: np.ndarray) -> None:
        if self.vertical:
            self.curve.setData(projection, axis_coords_um)
            self.setYRange(float(axis_coords_um[0]), float(axis_coords_um[-1]), padding=0.0)
        else:
            self.curve.setData(axis_coords_um, projection)
            self.setXRange(float(axis_coords_um[0]), float(axis_coords_um[-1]), padding=0.0)


class TrendPlot(pg.PlotWidget):
    def __init__(self, title: str, color: str) -> None:
        super().__init__()
        self.setBackground("#15181d")
        self.setTitle(title, color="#d8dee9", size="10pt")
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.getAxis("left").setPen(pg.mkPen("#5c6773"))
        self.getAxis("bottom").setPen(pg.mkPen("#5c6773"))
        self.setLabel("bottom", "Frame")
        self.setLabel("left", "Value", units="µm")
        self.getAxis("left").enableAutoSIPrefix(False)
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.curve = self.plot([], pen=pg.mkPen(color, width=2))

    def set_series(self, values: list[float]) -> None:
        x_values = np.arange(len(values), dtype=np.float64)
        self.curve.setData(x_values, values)
        if values:
            self.setXRange(0, max(len(values) - 1, 1), padding=0.02)


# ---------------------------------------------------------------------------
# ControlPanel — color variant
# ---------------------------------------------------------------------------

class ControlPanel(QtWidgets.QFrame):
    exposure_changed = QtCore.Signal(float)
    average_changed = QtCore.Signal(int)
    start_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    snapshot_requested = QtCore.Signal()
    brightness_changed = QtCore.Signal(float)
    auto_stretch_changed = QtCore.Signal(bool)
    pixel_saturation_changed = QtCore.Signal(int)
    fluence_params_changed = QtCore.Signal()
    slice_mode_changed = QtCore.Signal(str)
    trend_reset_requested = QtCore.Signal()
    trend_length_changed = QtCore.Signal(int)
    roi_toggled = QtCore.Signal(bool)
    zoom_to_roi_toggled = QtCore.Signal(bool)
    auto_exposure_toggled = QtCore.Signal(bool)
    simple_mode_toggled = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("controlPanel")
        self.setMinimumWidth(320)
        self.setMaximumWidth(380)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Beamprofiler Controls  [Color]")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.camera_info_button = QtWidgets.QPushButton("Camera Info")
        self.camera_info_button.clicked.connect(self._show_camera_info_popup)
        layout.addWidget(self.camera_info_button)

        self.model_label = QtWidgets.QLabel("IDS U3-3880LE Rev.1.2")
        self.serial_label = QtWidgets.QLabel("-")
        self.resolution_label = QtWidgets.QLabel(f"{SENSOR_WIDTH_PIXELS} x {SENSOR_HEIGHT_PIXELS}")
        self.pixel_size_label = QtWidgets.QLabel(f"{PIXEL_SIZE_UM:.1f} um")
        self.chip_size_label = QtWidgets.QLabel(
            f"{SENSOR_WIDTH_UM / 1000.0:.2f} x {SENSOR_HEIGHT_UM / 1000.0:.2f} mm"
        )

        # --- Acquisition ---
        acquisition_group = QtWidgets.QGroupBox("Acquisition")
        acquisition_form = QtWidgets.QFormLayout(acquisition_group)
        acquisition_form.setContentsMargins(4, 4, 4, 4)
        acquisition_form.setSpacing(2)
        acquisition_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.average_images = QtWidgets.QSpinBox()
        self.average_images.setRange(1, 512)
        self.average_images.setValue(1)

        self.exposure_time = QtWidgets.QDoubleSpinBox()
        self.exposure_time.setRange(0.01, 5000.0)
        self.exposure_time.setDecimals(2)
        self.exposure_time.setSuffix(" ms")
        self.exposure_time.setValue(10.0)
        self.exposure_time.setKeyboardTracking(False)

        self.auto_exposure_button = QtWidgets.QPushButton("Auto Exposure: OFF")
        self.auto_exposure_button.setCheckable(True)
        self.auto_exposure_button.setChecked(False)
        self.auto_exposure_button.toggled.connect(self._on_auto_exposure_toggled)

        acquisition_form.addRow("Average images", self.average_images)
        acquisition_form.addRow("Exposure time", self.exposure_time)
        acquisition_form.addRow("", self.auto_exposure_button)
        layout.addWidget(acquisition_group)

        # --- Display settings (replaces Beam Analysis colormap controls) ---
        display_group = QtWidgets.QGroupBox("Display")
        display_form = QtWidgets.QFormLayout(display_group)
        display_form.setContentsMargins(4, 4, 4, 4)
        display_form.setSpacing(2)

        self.auto_stretch = QtWidgets.QCheckBox("Auto-stretch brightness")
        self.auto_stretch.setChecked(True)

        self.brightness_spin = QtWidgets.QDoubleSpinBox()
        self.brightness_spin.setRange(0.05, 20.0)
        self.brightness_spin.setDecimals(2)
        self.brightness_spin.setSingleStep(0.1)
        self.brightness_spin.setValue(1.0)
        self.brightness_spin.setToolTip(
            "Manual brightness multiplier (effective when auto-stretch is off)"
        )
        self.brightness_spin.setEnabled(False)

        self.pixel_saturation = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pixel_saturation.setRange(1, 255)
        self.pixel_saturation.setValue(255)
        self.saturation_value_label = QtWidgets.QLabel("255 counts")

        display_form.addRow("Auto-stretch", self.auto_stretch)
        display_form.addRow("Brightness ×", self.brightness_spin)
        display_form.addRow("Pixel saturation", self.pixel_saturation)
        display_form.addRow("Saturation value", self.saturation_value_label)
        layout.addWidget(display_group)

        # --- Current Metrics ---
        metrics_group = QtWidgets.QGroupBox("Current Metrics")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)
        metrics_layout.setContentsMargins(4, 4, 4, 4)
        metrics_layout.setHorizontalSpacing(6)
        metrics_layout.setVerticalSpacing(1)
        metrics_layout.setColumnStretch(1, 1)
        metrics_layout.setColumnStretch(3, 1)
        self.sum_intensity_label = QtWidgets.QLabel("0")
        self.position_x_label = QtWidgets.QLabel("0.0 µm")
        self.position_y_label = QtWidgets.QLabel("0.0 µm")
        self.fwhm_x_label = QtWidgets.QLabel("0.0 µm")
        self.fwhm_y_label = QtWidgets.QLabel("0.0 µm")
        fwhm_value_style = f"font-weight: 800; color: #ffd166; font-size: {scale_font_size(20)}px;"
        self.fwhm_x_label.setStyleSheet(fwhm_value_style)
        self.fwhm_y_label.setStyleSheet(fwhm_value_style)
        self.saturation_progress = QtWidgets.QProgressBar()
        self.saturation_progress.setRange(0, 255)
        self.saturation_progress.setValue(0)
        self.saturation_progress.setFormat("%v / %m")
        self.saturation_progress.setMinimumHeight(16)
        self.saturation_progress.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #313742; border-radius: 4px;"
            f" background: #0f1216; text-align: center; color: #000000;"
            f" font-weight: 600; font-size: {scale_font_size(12)}px; min-height: 16px; }}"
            f"QProgressBar::chunk {{ background: #ffb347; border-radius: 3px; }}"
        )
        title_style = "color: #c7d0db;"
        for label_text, widget, row, col in [
            ("Position X", self.position_x_label, 0, 0),
            ("FWHM X", self.fwhm_x_label, 0, 2),
            ("Position Y", self.position_y_label, 1, 0),
            ("FWHM Y", self.fwhm_y_label, 1, 2),
            ("Sum intensity", self.sum_intensity_label, 2, 0),
            ("Saturation", self.saturation_progress, 2, 2),
        ]:
            if label_text != "Saturation":
                lbl = QtWidgets.QLabel(label_text)
                lbl.setStyleSheet(title_style)
                metrics_layout.addWidget(lbl, row, col)
                metrics_layout.addWidget(widget, row, col + 1)
            else:
                lbl = QtWidgets.QLabel(label_text)
                lbl.setStyleSheet(title_style)
                metrics_layout.addWidget(lbl, row, col)
                metrics_layout.addWidget(widget, row, col + 1)
        layout.addWidget(metrics_group)

        # --- Fluence Calculator ---
        fluence_group = QtWidgets.QGroupBox("Fluence Calculator")
        fluence_form = QtWidgets.QFormLayout(fluence_group)
        fluence_form.setContentsMargins(4, 4, 4, 4)
        fluence_form.setSpacing(2)
        self.power_input = QtWidgets.QDoubleSpinBox()
        self.power_input.setRange(0.1, 1e6)
        self.power_input.setDecimals(1)
        self.power_input.setSuffix(" µW")
        self.power_input.setValue(20.0)
        self.power_input.setKeyboardTracking(False)

        self.rep_rate_input = QtWidgets.QSpinBox()
        self.rep_rate_input.setRange(1, 1000000000)
        self.rep_rate_input.setSuffix(" Hz")
        self.rep_rate_input.setValue(500)
        self.rep_rate_input.setKeyboardTracking(False)

        self.aoi_input = QtWidgets.QDoubleSpinBox()
        self.aoi_input.setRange(0.0, 89.9)
        self.aoi_input.setDecimals(1)
        self.aoi_input.setSuffix(" °")
        self.aoi_input.setValue(0.0)
        self.aoi_input.setKeyboardTracking(False)

        self.fluence_mode_group = QtWidgets.QButtonGroup(self)
        self.fluence_mode_fwhm_radio = QtWidgets.QRadioButton("Top-hat with FWHM/2 radii")
        self.fluence_mode_one_over_e_radio = QtWidgets.QRadioButton("Top-hat with 1/e radii")
        self.fluence_mode_fwhm_radio.setChecked(True)
        self.fluence_mode_group.addButton(self.fluence_mode_fwhm_radio)
        self.fluence_mode_group.addButton(self.fluence_mode_one_over_e_radio)
        fluence_mode_widget = QtWidgets.QWidget()
        fluence_mode_layout = QtWidgets.QVBoxLayout(fluence_mode_widget)
        fluence_mode_layout.setContentsMargins(0, 0, 0, 0)
        fluence_mode_layout.setSpacing(4)
        fluence_mode_layout.addWidget(self.fluence_mode_fwhm_radio)
        fluence_mode_layout.addWidget(self.fluence_mode_one_over_e_radio)

        self.fluence_formula_label = QtWidgets.QLabel(fluence_formula_text(FLUENCE_MODE_FWHM))
        self.fluence_formula_label.setWordWrap(True)
        self.fluence_formula_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.fluence_formula_label.setStyleSheet(
            "background: #10141a; border: 1px solid #2c313a; border-radius: 6px;"
            " padding: 4px; color: #d8dee9;"
        )

        self.fluence_label = QtWidgets.QLabel("0.0 µJ/cm²")
        self.fluence_label.setStyleSheet(
            f"font-weight: 800; color: #ffb347; font-size: {scale_font_size(18)}px;"
        )

        fluence_form.addRow("Power", self.power_input)
        fluence_form.addRow("Rep. rate", self.rep_rate_input)
        fluence_form.addRow("AOI", self.aoi_input)
        fluence_form.addRow("Model", fluence_mode_widget)
        fluence_form.addRow("Formula", self.fluence_formula_label)
        fluence_form.addRow("Fluence", self.fluence_label)
        layout.addWidget(fluence_group)

        # --- Projection / Slice mode ---
        self.slice_group = QtWidgets.QGroupBox("Projection Mode")
        self.slice_group.setMaximumHeight(132)
        slice_layout = QtWidgets.QVBoxLayout(self.slice_group)
        slice_layout.setContentsMargins(4, 4, 4, 4)
        slice_layout.setSpacing(1)
        self.slice_mode_selector = QtWidgets.QButtonGroup(self)
        self.radio_sum = QtWidgets.QRadioButton("Sum projection")
        self.radio_cursor = QtWidgets.QRadioButton("Slice at cursor")
        self.radio_peak = QtWidgets.QRadioButton("Slice at peak intensity")
        self.radio_cursor.setChecked(True)
        self.slice_mode_selector.addButton(self.radio_sum, 0)
        self.slice_mode_selector.addButton(self.radio_cursor, 1)
        self.slice_mode_selector.addButton(self.radio_peak, 2)
        slice_layout.addWidget(self.radio_sum)
        slice_layout.addWidget(self.radio_cursor)
        slice_layout.addWidget(self.radio_peak)

        self.roi_checkbox = QtWidgets.QCheckBox("Enable ROI selection")
        self.roi_checkbox.setChecked(False)
        slice_layout.addWidget(self.roi_checkbox)

        self.zoom_to_roi_radio = QtWidgets.QRadioButton("zoom to ROI")
        self.zoom_to_roi_radio.setAutoExclusive(False)
        self.zoom_to_roi_radio.setChecked(False)
        slice_layout.addWidget(self.zoom_to_roi_radio)
        layout.addWidget(self.slice_group)

        # --- Markers ---
        self.marker_group = QtWidgets.QGroupBox("Markers")
        marker_layout = QtWidgets.QHBoxLayout(self.marker_group)
        marker_layout.setContentsMargins(4, 4, 4, 4)
        marker_layout.setSpacing(4)
        self.add_marker_button = QtWidgets.QPushButton("Add Marker")
        self.clear_markers_button = QtWidgets.QPushButton("Clear Markers")
        marker_layout.addWidget(self.add_marker_button)
        marker_layout.addWidget(self.clear_markers_button)
        layout.addWidget(self.marker_group)

        self.simple_mode_button = QtWidgets.QPushButton("Simple Mode: ON")
        self.simple_mode_button.setCheckable(True)
        self.simple_mode_button.setChecked(True)
        self.simple_mode_button.toggled.connect(self._on_simple_mode_toggled)
        layout.addWidget(self.simple_mode_button)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(4)
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.snapshot_button = QtWidgets.QPushButton("Snapshot")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.snapshot_button)
        layout.addLayout(button_row)

        self.backend_label = QtWidgets.QLabel("Backend: starting...")
        self.backend_label.setWordWrap(True)
        layout.addWidget(self.backend_label)
        layout.addStretch(1)

        # --- Signal connections ---
        self.average_images.valueChanged.connect(self.average_changed.emit)
        self.exposure_time.valueChanged.connect(self.exposure_changed.emit)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.snapshot_button.clicked.connect(self.snapshot_requested.emit)
        self.brightness_spin.valueChanged.connect(self.brightness_changed.emit)
        self.auto_stretch.toggled.connect(self._on_auto_stretch_toggled)
        self.pixel_saturation.valueChanged.connect(self.pixel_saturation_changed.emit)
        self.pixel_saturation.valueChanged.connect(
            lambda v: self.saturation_value_label.setText(f"{v} counts")
        )
        self.power_input.valueChanged.connect(lambda _: self.fluence_params_changed.emit())
        self.rep_rate_input.valueChanged.connect(lambda _: self.fluence_params_changed.emit())
        self.aoi_input.valueChanged.connect(lambda _: self.fluence_params_changed.emit())
        self.fluence_mode_fwhm_radio.toggled.connect(self._on_fluence_mode_toggled)
        self.fluence_mode_one_over_e_radio.toggled.connect(self._on_fluence_mode_toggled)
        self.slice_mode_selector.idToggled.connect(
            lambda id_, checked: self.slice_mode_changed.emit(
                ["sum", "cursor", "peak"][id_]
            ) if checked else None
        )
        self.roi_checkbox.toggled.connect(self.roi_toggled.emit)
        self.zoom_to_roi_radio.toggled.connect(self.zoom_to_roi_toggled.emit)

    def _on_auto_stretch_toggled(self, checked: bool) -> None:
        self.brightness_spin.setEnabled(not checked)
        self.auto_stretch_changed.emit(checked)

    def _on_auto_exposure_toggled(self, checked: bool) -> None:
        self.auto_exposure_button.setText(f"Auto Exposure: {'ON' if checked else 'OFF'}")
        self.exposure_time.setEnabled(not checked)
        self.auto_exposure_toggled.emit(checked)

    def _on_simple_mode_toggled(self, checked: bool) -> None:
        self.simple_mode_button.setText(f"Simple Mode: {'ON' if checked else 'OFF'}")
        self.simple_mode_toggled.emit(checked)

    def _show_camera_info_popup(self) -> None:
        details = (
            f"Model: {self.model_label.text()}\n"
            f"Serial: {self.serial_label.text()}\n"
            f"Resolution: {self.resolution_label.text()}\n"
            f"Pixel size: {self.pixel_size_label.text()}\n"
            f"Chip size: {self.chip_size_label.text()}"
        )
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Camera Information")
        msg.setText(details)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg.setStyleSheet("QLabel { color: #000000; } QPushButton { color: #000000; }")
        msg.exec()

    def _on_fluence_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self.fluence_formula_label.setText(fluence_formula_text(self.get_fluence_mode()))
        self.fluence_params_changed.emit()

    def get_fluence_mode(self) -> str:
        if self.fluence_mode_one_over_e_radio.isChecked():
            return FLUENCE_MODE_ONE_OVER_E
        return FLUENCE_MODE_FWHM

    def update_metrics(self, metrics: BeamMetrics, sum_intensity: float | None = None) -> None:
        sum_counts = metrics.sum_intensity if sum_intensity is None else sum_intensity
        self.sum_intensity_label.setText(f"{sum_counts:.0f}")
        self.position_x_label.setText(f"{metrics.centroid_x_um:.0f} µm")
        self.position_y_label.setText(f"{metrics.centroid_y_um:.0f} µm")
        self.fwhm_x_label.setText(f"{metrics.fwhm_x_um:.0f} µm")
        self.fwhm_y_label.setText(f"{metrics.fwhm_y_um:.0f} µm")
        self.update_fluence(metrics.fwhm_x_um, metrics.fwhm_y_um)

    def update_fluence(self, fwhm_x_um: float, fwhm_y_um: float) -> None:
        fluence = calc_fluence(
            power_uW=self.power_input.value(),
            fwhm_x=fwhm_x_um,
            fwhm_y=fwhm_y_um,
            angle=self.aoi_input.value(),
            rep_rate=self.rep_rate_input.value(),
            mode=self.get_fluence_mode(),
        )
        self.fluence_label.setText(f"{fluence:.1f} µJ/cm²")

    def update_camera_state(self, state: CameraState) -> None:
        self.model_label.setText(state.model_name)
        self.serial_label.setText(state.serial_number)
        exposure_min = max(0.01, float(state.exposure_min_ms))
        exposure_max = max(exposure_min, float(state.exposure_max_ms))
        exposure_val = min(max(float(state.exposure_ms), exposure_min), exposure_max)
        self.exposure_time.blockSignals(True)
        self.exposure_time.setRange(exposure_min, exposure_max)
        self.exposure_time.setValue(exposure_val)
        self.exposure_time.blockSignals(False)

    def set_backend_message(self, text: str) -> None:
        self.backend_label.setText(text)

    def update_saturation_headroom(self, peak_counts: float, full_scale_counts: float) -> None:
        full_scale = max(1.0, float(full_scale_counts))
        level_255 = int(np.clip(np.round((peak_counts / full_scale) * 255.0), 0, 255))
        self.saturation_progress.setValue(level_255)


# ---------------------------------------------------------------------------
# AcquisitionThread — color variant
# ---------------------------------------------------------------------------

class AcquisitionThread(QtCore.QThread):
    frame_ready = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    camera_ready = QtCore.Signal(object)
    error_occurred = QtCore.Signal(str)

    def __init__(self, pixel_size_um: float, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.pixel_size_um = pixel_size_um
        self._lock = Lock()
        self._desired_running = True
        self._average_images = 1
        self._exposure_ms: float | None = None
        self._gain: float | None = None

    def set_average_images(self, value: int) -> None:
        with self._lock:
            self._average_images = max(1, int(value))

    def set_exposure_ms(self, value: float) -> None:
        with self._lock:
            self._exposure_ms = max(0.001, float(value))

    def set_gain(self, value: float) -> None:
        with self._lock:
            self._gain = float(value)

    def request_start(self) -> None:
        with self._lock:
            self._desired_running = True

    def request_stop(self) -> None:
        with self._lock:
            self._desired_running = False

    def stop_thread(self) -> None:
        self.request_stop()
        self.requestInterruption()
        self.wait(3000)

    def _snapshot_config(self) -> tuple[bool, int, float | None, float | None]:
        with self._lock:
            return self._desired_running, self._average_images, self._exposure_ms, self._gain

    # -- Synthetic fallback ---------------------------------------------------

    def _run_synthetic(self, reason: str) -> None:
        self.status_changed.emit(reason)
        frame_index = 0
        while not self.isInterruptionRequested():
            desired_running, average_images, _, _ = self._snapshot_config()
            if not desired_running:
                self.msleep(50)
                continue
            gray, rgb = self._generate_mock_frame_color(frame_index)
            frame_index += max(1, average_images)
            self.frame_ready.emit(self._build_frame(gray, rgb, 255.0))
            self.msleep(50)

    def _generate_mock_frame_color(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (gray float32 H×W, rgb uint8 H×W×3) synthetic Gaussian with warm colour."""
        width, height = 640, 480
        x_values = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y_values = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        t = frame_index / 20.0
        center_x = 0.20 * math.sin(t * 0.8)
        center_y = 0.14 * math.cos(t * 0.6)
        sigma_x = 0.16 + 0.02 * math.sin(t * 0.45)
        sigma_y = 0.12 + 0.02 * math.cos(t * 0.52)
        angle = 0.35 * math.sin(t * 0.3)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = cos_a * (x_grid - center_x) + sin_a * (y_grid - center_y)
        y_rot = -sin_a * (x_grid - center_x) + cos_a * (y_grid - center_y)

        beam = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        side_lobe = 0.12 * np.exp(
            -0.5 * (((x_grid + 0.32) / 0.09) ** 2 + ((y_grid - 0.25) / 0.07) ** 2)
        )
        rng = np.random.default_rng(frame_index)
        noise = rng.normal(0.0, 0.01, size=beam.shape).astype(np.float32)
        gray = np.clip(beam + side_lobe + noise, 0.0, None).astype(np.float32)

        # Simulate a warm red/orange beam colour
        r = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        g = np.clip(gray * 100.0, 0, 255).astype(np.uint8)
        b = np.clip(gray * 20.0, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return gray, rgb

    def _build_frame(
        self,
        gray_image: np.ndarray,
        rgb_image: np.ndarray,
        full_scale_counts: float,
    ) -> FrameData:
        oriented_gray = np.rot90(gray_image, k=DISPLAY_ROTATE_K).astype(np.float32)
        oriented_rgb = np.rot90(rgb_image, k=DISPLAY_ROTATE_K)
        metrics, projection_x, projection_y = compute_metrics(oriented_gray, self.pixel_size_um)
        return FrameData(
            gray_image=oriented_gray,
            rgb_image=np.ascontiguousarray(oriented_rgb),
            projection_x=projection_x,
            projection_y=projection_y,
            x_coordinates_um=make_axis_um(oriented_gray.shape[1], self.pixel_size_um),
            y_coordinates_um=make_axis_um(oriented_gray.shape[0], self.pixel_size_um),
            metrics=metrics,
            camera_full_scale_counts=float(full_scale_counts),
        )

    # -- Real camera ----------------------------------------------------------

    def run(self) -> None:
        if not IDS_BACKEND_AVAILABLE:
            reason = "Synthetic mode: IDS backend unavailable"
            if IDS_IMPORT_ERROR is not None:
                reason += f" ({IDS_IMPORT_ERROR})"
            self._run_synthetic(reason)
            return

        camera: Any | None = None
        acquisition_active = False
        accumulation_gray: np.ndarray | None = None
        accumulation_rgb: np.ndarray | None = None
        accumulated_frames = 0
        applied_exposure_ms: float | None = None
        applied_gain: float | None = None

        try:
            camera = Camera()
            camera.restore_coordinate_flip()

            _, average_images, desired_exposure_ms, desired_gain = self._snapshot_config()
            if desired_exposure_ms is None:
                desired_exposure_ms = camera.exposure / 1000.0
            if desired_gain is None:
                desired_gain = camera.master_gain

            exposure_range = camera.exposure_range
            gain_range = camera.master_gain_range
            self.camera_ready.emit(
                CameraState(
                    model_name=str(camera.device.ModelName()),
                    serial_number=str(camera.device.SerialNumber()),
                    exposure_ms=float(camera.exposure / 1000.0),
                    exposure_min_ms=float(exposure_range.minimum / 1000.0),
                    exposure_max_ms=float(exposure_range.maximum / 1000.0),
                    gain_value=float(camera.master_gain),
                    gain_min=float(gain_range.minimum),
                    gain_max=float(gain_range.maximum),
                )
            )

            pipeline = DefaultPipeline()
            self.status_changed.emit("Live camera mode: IDS acquisition active")

            while not self.isInterruptionRequested():
                desired_running, average_images, desired_exposure_ms, desired_gain = self._snapshot_config()

                if desired_running and not acquisition_active:
                    camera.start_acquisition()
                    acquisition_active = True
                    accumulation_gray = None
                    accumulation_rgb = None
                    accumulated_frames = 0
                    self.status_changed.emit("Live camera mode: acquisition running")

                if not desired_running and acquisition_active:
                    camera.kill_datastream_wait()
                    camera.stop_acquisition()
                    acquisition_active = False
                    accumulation_gray = None
                    accumulation_rgb = None
                    accumulated_frames = 0
                    self.status_changed.emit("Live camera mode: acquisition stopped")

                if desired_exposure_ms is not None and desired_exposure_ms != applied_exposure_ms:
                    try:
                        camera.exposure = desired_exposure_ms * 1000.0
                        applied_exposure_ms = desired_exposure_ms
                    except CommonException as exc:
                        self.error_occurred.emit(f"Failed to set exposure: {exc}")

                if desired_gain is not None and desired_gain != applied_gain:
                    try:
                        camera.master_gain = desired_gain
                        applied_gain = desired_gain
                    except (CommonException, StopIteration) as exc:
                        self.error_occurred.emit(f"Failed to set gain: {exc}")

                if not acquisition_active:
                    self.msleep(30)
                    continue

                try:
                    image_view = camera.wait_for_image_view(500)
                except ids_peak.TimeoutException:
                    continue
                except ids_peak.AbortedException:
                    if self.isInterruptionRequested():
                        break
                    continue

                if image_view.parent_buffer.IsIncomplete():
                    camera.queue_buffer(image_view.parent_buffer)
                    continue

                processed_image = None
                try:
                    processed_image = pipeline.process(image_view)
                except CommonException as exc:
                    self.error_occurred.emit(f"Failed to process frame: {exc}")
                finally:
                    camera.queue_buffer(image_view.parent_buffer)

                if processed_image is None:
                    continue

                source_image = processed_image.to_numpy_array()

                if np.issubdtype(source_image.dtype, np.integer):
                    full_scale_counts = float(np.iinfo(source_image.dtype).max)
                else:
                    full_scale_counts = max(255.0, float(np.max(source_image)))

                gray_image = convert_to_gray(source_image)
                rgb_image = build_rgb_uint8(source_image)

                # Accumulate for temporal averaging
                if accumulation_gray is None:
                    accumulation_gray = gray_image.astype(np.float64)
                    accumulation_rgb = rgb_image.astype(np.float64)
                else:
                    accumulation_gray += gray_image
                    accumulation_rgb = accumulation_rgb + rgb_image.astype(np.float64)  # type: ignore[operator]
                accumulated_frames += 1

                if accumulated_frames < max(1, average_images):
                    continue

                avg_gray = (accumulation_gray / accumulated_frames).astype(np.float32)
                avg_rgb = np.clip(
                    accumulation_rgb / accumulated_frames, 0, 255  # type: ignore[operator]
                ).astype(np.uint8)
                accumulation_gray = None
                accumulation_rgb = None
                accumulated_frames = 0
                self.frame_ready.emit(self._build_frame(avg_gray, avg_rgb, full_scale_counts))

        except BaseException as exc:  # pragma: no cover
            message = f"Falling back to synthetic mode: {exc}"
            self.error_occurred.emit(message)
            self._run_synthetic(message)
        finally:
            if camera is not None:
                for fn in (
                    camera.kill_datastream_wait,
                    camera.stop_acquisition,
                    camera.restore_coordinate_flip,
                ):
                    try:
                        fn()
                    except Exception:
                        pass
                try:
                    del camera
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# BeamProfilerApp — color variant
# ---------------------------------------------------------------------------

class BeamProfilerApp(QtWidgets.QMainWindow):
    history_length = 180

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IDS Beamprofiler — Color")

        self.position_x_history: list[float] = []
        self.position_y_history: list[float] = []
        self.fwhm_x_history: list[float] = []
        self.fwhm_y_history: list[float] = []
        self.sum_intensity_history: list[float] = []

        self._last_frame: FrameData | None = None
        self._auto_stretch = True
        self._brightness_scale = 1.0
        self._saturation_level = 255
        self._slice_mode = "cursor"
        self._pending_frame: FrameData | None = None
        self._auto_exposure = False
        self._simple_mode = True
        self._zoom_to_roi = False
        self._frame_times: list[float] = []

        self._cached_roi: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cached_roi_projections: tuple[np.ndarray, np.ndarray] | None = None

        self._cursor_timer = QtCore.QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(30)
        self._cursor_timer.timeout.connect(self._on_cursor_timer)

        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._process_pending_frame)

        # --- UI layout ---
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QtWidgets.QHBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self.control_panel = ControlPanel()
        root_layout.addWidget(self.control_panel)

        center_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        center_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(center_splitter, 1)

        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QGridLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        self.image_view = BeamImageView()
        self.x_projection_plot = ProjectionPlot(vertical=False)
        self.y_projection_plot = ProjectionPlot(vertical=True)

        image_header = QtWidgets.QLabel("Beam Image  [Color]")
        image_header.setObjectName("sectionTitle")
        x_projection_header = QtWidgets.QLabel("Horizontal Projection (luminance)")
        x_projection_header.setObjectName("sectionTitle")
        y_projection_header = QtWidgets.QLabel("Vertical Projection (luminance)")
        y_projection_header.setObjectName("sectionTitle")

        center_layout.addWidget(image_header, 0, 0)
        center_layout.addWidget(y_projection_header, 0, 1)
        center_layout.addWidget(self.image_view, 1, 0)
        center_layout.addWidget(self.y_projection_plot, 1, 1)
        center_layout.addWidget(x_projection_header, 2, 0)
        center_layout.addWidget(self.x_projection_plot, 3, 0)

        # Move projection-mode group to bottom-right corner
        self.control_panel.layout().removeWidget(self.control_panel.slice_group)
        self.control_panel.slice_group.setParent(None)
        self.control_panel.slice_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        center_layout.addWidget(
            self.control_panel.slice_group, 3, 1,
            alignment=QtCore.Qt.AlignmentFlag.AlignTop,
        )

        self.control_panel.layout().removeWidget(self.control_panel.marker_group)
        self.control_panel.marker_group.setParent(None)
        self.control_panel.marker_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        center_layout.addWidget(
            self.control_panel.marker_group, 4, 1,
            alignment=QtCore.Qt.AlignmentFlag.AlignTop,
        )

        center_layout.setColumnStretch(0, 9)
        center_layout.setColumnStretch(1, 1)
        center_layout.setRowStretch(1, 5)
        center_layout.setRowStretch(3, 3)
        center_layout.setRowStretch(4, 0)

        center_splitter.addWidget(center_widget)

        # --- Trend panel ---
        trend_panel = QtWidgets.QFrame()
        trend_panel.setObjectName("trendPanel")
        trend_panel.setMinimumWidth(280)
        trend_panel.setMaximumWidth(380)
        trend_layout = QtWidgets.QVBoxLayout(trend_panel)
        trend_layout.setContentsMargins(12, 12, 12, 12)
        trend_layout.setSpacing(8)

        trend_title = QtWidgets.QLabel("Live Trends")
        trend_title.setObjectName("panelTitle")
        trend_layout.addWidget(trend_title)

        trend_controls = QtWidgets.QHBoxLayout()
        trend_controls.setSpacing(6)
        self.trend_length_spin = QtWidgets.QSpinBox()
        self.trend_length_spin.setRange(10, 10000)
        self.trend_length_spin.setValue(180)
        self.trend_length_spin.setSuffix(" pts")
        self.trend_reset_button = QtWidgets.QPushButton("Reset")
        trend_controls.addWidget(QtWidgets.QLabel("Length:"))
        trend_controls.addWidget(self.trend_length_spin, 1)
        trend_controls.addWidget(self.trend_reset_button)
        trend_layout.addLayout(trend_controls)

        self.trend_reset_button.clicked.connect(self.control_panel.trend_reset_requested.emit)
        self.trend_length_spin.valueChanged.connect(self.control_panel.trend_length_changed.emit)

        self.position_x_plot = TrendPlot("Peak Position X", "#52b788")
        self.position_y_plot = TrendPlot("Peak Position Y", "#2ec4b6")
        self.fwhm_x_plot = TrendPlot("FWHM X", "#ffb347")
        self.fwhm_y_plot = TrendPlot("FWHM Y", "#ff8fab")
        self.sum_intensity_plot = TrendPlot("Sum Intensity", "#f6bd60")

        for p in (
            self.position_x_plot, self.position_y_plot,
            self.fwhm_x_plot, self.fwhm_y_plot, self.sum_intensity_plot,
        ):
            trend_layout.addWidget(p)

        center_splitter.addWidget(trend_panel)
        center_splitter.setSizes([1300, 340])

        self._apply_styles()

        # --- Acquisition thread ---
        self.acquisition_thread = AcquisitionThread(pixel_size_um=PIXEL_SIZE_UM, parent=self)
        self.acquisition_thread.frame_ready.connect(self._enqueue_frame)
        self.acquisition_thread.status_changed.connect(self._show_status)
        self.acquisition_thread.camera_ready.connect(self._handle_camera_state)
        self.acquisition_thread.error_occurred.connect(self._show_error)

        self.control_panel.average_changed.connect(self.acquisition_thread.set_average_images)
        self.control_panel.exposure_changed.connect(self.acquisition_thread.set_exposure_ms)
        self.control_panel.start_requested.connect(self.acquisition_thread.request_start)
        self.control_panel.stop_requested.connect(self.acquisition_thread.request_stop)
        self.control_panel.snapshot_requested.connect(self._save_snapshot)
        self.control_panel.brightness_changed.connect(self._on_brightness_changed)
        self.control_panel.auto_stretch_changed.connect(self._on_auto_stretch_changed)
        self.control_panel.pixel_saturation_changed.connect(self._on_saturation_changed)
        self.control_panel.fluence_params_changed.connect(self._on_fluence_params_changed)
        self.control_panel.slice_mode_changed.connect(self._on_slice_mode_changed)
        self.control_panel.trend_reset_requested.connect(self._reset_trends)
        self.control_panel.trend_length_changed.connect(self._on_trend_length_changed)
        self.control_panel.average_changed.connect(lambda _: self._reset_trends())
        self.control_panel.roi_toggled.connect(self._on_roi_toggled)
        self.control_panel.zoom_to_roi_toggled.connect(self._on_zoom_to_roi_toggled)
        self.control_panel.auto_exposure_toggled.connect(self._on_auto_exposure_toggled)
        self.image_view.slice_h_line.sigPositionChanged.connect(self._on_cursor_moved)
        self.image_view.slice_v_line.sigPositionChanged.connect(self._on_cursor_moved)
        self.image_view.roi_rect.sigRegionChangeFinished.connect(self._on_roi_changed)
        self.image_view.roi_rect.sigRegionChangeFinished.connect(self._reset_trends)
        self.control_panel.add_marker_button.clicked.connect(self._add_crosshair_marker)
        self.control_panel.clear_markers_button.clicked.connect(self._clear_crosshair_markers)
        self.control_panel.simple_mode_toggled.connect(self._on_simple_mode_toggled)

        self._fps_label = QtWidgets.QLabel("FPS: -")
        self._fps_label.setStyleSheet(
            f"font-weight: 700; color: #52b788; font-size: {scale_font_size(13)}px; padding: 0 12px;"
        )
        self.statusBar().addPermanentWidget(self._fps_label)
        self.statusBar().showMessage("Starting IDS Color beamprofiler...")
        self.acquisition_thread.start()

    # -- Styles ---------------------------------------------------------------

    def _apply_styles(self) -> None:
        panel_title_size = scale_font_size(18)
        section_title_size = scale_font_size(13)
        self.setStyleSheet(
            """
            QMainWindow {{ background: #0b0d10; }}
            QFrame#controlPanel, QFrame#trendPanel {{
                background: #171a1f;
                border: 1px solid #282d35;
                border-radius: 12px;
            }}
            QLabel {{ color: #d8dee9; }}
            QLabel#panelTitle {{ font-size: {pt}px; font-weight: 700; color: #ffffff; }}
            QLabel#sectionTitle {{ font-size: {st}px; font-weight: 600; color: #c7d0db; }}
            QGroupBox {{
                color: #e5e9f0; border: 1px solid #2c313a; border-radius: 10px;
                margin-top: 12px; padding-top: 12px; font-weight: 600;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background: #0f1216; color: #eef2f6;
                border: 1px solid #313742; border-radius: 6px;
                min-height: 28px; padding: 2px 8px;
            }}
            QComboBox QAbstractItemView {{
                background: #0f1216; color: #eef2f6;
                border: 1px solid #313742;
                selection-background-color: #355070; selection-color: #ffffff; outline: 0;
            }}
            QPushButton {{
                background: #233142; color: #eef2f6;
                border: 1px solid #355070; border-radius: 7px;
                min-height: 30px; padding: 0 10px;
            }}
            QPushButton:hover {{ background: #2d4157; }}
            QCheckBox {{ color: #d8dee9; }}
            QRadioButton {{ color: #ffffff; }}
            QSlider::groove:horizontal {{
                border-radius: 4px; height: 8px; background: #232831;
            }}
            QSlider::handle:horizontal {{
                background: #ffb347; width: 16px; margin: -4px 0; border-radius: 8px;
            }}
            """.format(pt=panel_title_size, st=section_title_size)
        )

    # -- Frame pipeline -------------------------------------------------------

    def _compute_display_rgb(self, frame: FrameData) -> np.ndarray:
        """Return H×W×3 uint8 ready for display, with brightness applied."""
        rgb = frame.rgb_image
        if self._auto_stretch:
            # Stretch each channel to use full 0-255 range
            max_lum = float(rgb.max())
            if max_lum > 0:
                scale = 255.0 / max_lum
                rgb = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        else:
            if self._brightness_scale != 1.0:
                rgb = np.clip(
                    rgb.astype(np.float32) * self._brightness_scale, 0, 255
                ).astype(np.uint8)
        return rgb

    def _refresh_last_frame_display(self) -> None:
        if self._last_frame is None:
            return
        display_rgb = self._compute_display_rgb(self._last_frame)
        # Build a display frame with replaced rgb for image_view
        display_frame = FrameData(
            gray_image=self._last_frame.gray_image,
            rgb_image=display_rgb,
            projection_x=self._last_frame.projection_x,
            projection_y=self._last_frame.projection_y,
            x_coordinates_um=self._last_frame.x_coordinates_um,
            y_coordinates_um=self._last_frame.y_coordinates_um,
            metrics=self._last_frame.metrics,
            camera_full_scale_counts=self._last_frame.camera_full_scale_counts,
        )
        self.image_view.set_image(display_frame)

    @QtCore.Slot(object)
    def _handle_camera_state(self, state: CameraState) -> None:
        self.control_panel.update_camera_state(state)
        self._show_status(f"Connected to {state.model_name} ({state.serial_number})")

    @QtCore.Slot(object)
    def _enqueue_frame(self, frame: FrameData) -> None:
        self._pending_frame = frame
        if not self._frame_timer.isActive():
            self._frame_timer.start()

    @QtCore.Slot()
    def _process_pending_frame(self) -> None:
        frame = self._pending_frame
        if frame is None:
            return
        self._pending_frame = None
        self._handle_frame(frame)

    @QtCore.Slot(bool)
    def _on_simple_mode_toggled(self, enabled: bool) -> None:
        self._simple_mode = enabled

    @QtCore.Slot(object)
    def _handle_frame(self, frame: FrameData) -> None:
        self._last_frame = frame
        if self._simple_mode:
            self._handle_frame_simple(frame)
            return

        self._saturation_level = self.control_panel.pixel_saturation.value()
        display_rgb = self._compute_display_rgb(frame)
        display_frame = FrameData(
            gray_image=frame.gray_image,
            rgb_image=display_rgb,
            projection_x=frame.projection_x,
            projection_y=frame.projection_y,
            x_coordinates_um=frame.x_coordinates_um,
            y_coordinates_um=frame.y_coordinates_um,
            metrics=frame.metrics,
            camera_full_scale_counts=frame.camera_full_scale_counts,
        )
        self.image_view.set_image(display_frame)

        sub_gray, sub_rgb, sub_x, sub_y = self._get_roi_region(frame)
        self._cached_roi = (sub_gray, sub_rgb, sub_x, sub_y)
        roi_metrics, roi_proj_x, roi_proj_y = self._compute_roi_metrics(sub_gray, sub_x, sub_y)
        self._cached_roi_projections = (roi_proj_x, roi_proj_y)

        if self._slice_mode in ("cursor", "peak"):
            roi_metrics = self._slice_metrics(sub_gray, sub_x, sub_y, roi_metrics)

        sum_intensity_total = float(frame.gray_image.sum())
        self.image_view.horizontal_line.setValue(roi_metrics.centroid_y_um)
        self.image_view.vertical_line.setValue(roi_metrics.centroid_x_um)

        self._update_projections_cached(sub_gray, sub_x, sub_y, roi_proj_x, roi_proj_y)
        self.control_panel.update_metrics(roi_metrics, sum_intensity_total)
        self.control_panel.update_saturation_headroom(
            roi_metrics.peak_value, frame.camera_full_scale_counts
        )

        self._push_history(self.position_x_history, roi_metrics.centroid_x_um)
        self._push_history(self.position_y_history, roi_metrics.centroid_y_um)
        self._push_history(self.fwhm_x_history, roi_metrics.fwhm_x_um)
        self._push_history(self.fwhm_y_history, roi_metrics.fwhm_y_um)
        self._push_history(self.sum_intensity_history, sum_intensity_total)

        self.position_x_plot.set_series(self.position_x_history)
        self.position_y_plot.set_series(self.position_y_history)
        self.fwhm_x_plot.set_series(self.fwhm_x_history)
        self.fwhm_y_plot.set_series(self.fwhm_y_history)
        self.sum_intensity_plot.set_series(self.sum_intensity_history)

        self._auto_adjust_exposure(roi_metrics.peak_value, frame.camera_full_scale_counts)
        self._update_fps()

    def _handle_frame_simple(self, frame: FrameData) -> None:
        self._saturation_level = self.control_panel.pixel_saturation.value()
        display_rgb = self._compute_display_rgb(frame)
        display_frame = FrameData(
            gray_image=frame.gray_image,
            rgb_image=display_rgb,
            projection_x=frame.projection_x,
            projection_y=frame.projection_y,
            x_coordinates_um=frame.x_coordinates_um,
            y_coordinates_um=frame.y_coordinates_um,
            metrics=frame.metrics,
            camera_full_scale_counts=frame.camera_full_scale_counts,
        )
        self.image_view.set_image(display_frame)

        image = frame.gray_image  # luminance for projections
        x_um = frame.x_coordinates_um
        y_um = frame.y_coordinates_um

        if self._slice_mode == "peak":
            peak_idx = np.unravel_index(np.argmax(image), image.shape)
            proj_x = image[peak_idx[0], :].astype(np.float64)
            proj_y = image[:, peak_idx[1]].astype(np.float64)
            self.image_view.horizontal_line.setValue(y_um[peak_idx[0]])
            self.image_view.vertical_line.setValue(x_um[peak_idx[1]])
        elif self._slice_mode == "cursor":
            cursor_y_um = self.image_view.slice_h_line.value()
            cursor_x_um = self.image_view.slice_v_line.value()
            row_idx = int(np.clip(np.argmin(np.abs(y_um - cursor_y_um)), 0, image.shape[0] - 1))
            col_idx = int(np.clip(np.argmin(np.abs(x_um - cursor_x_um)), 0, image.shape[1] - 1))
            proj_x = image[row_idx, :].astype(np.float64)
            proj_y = image[:, col_idx].astype(np.float64)
        else:
            proj_x = frame.projection_x.astype(np.float64)
            proj_y = frame.projection_y.astype(np.float64)

        self.x_projection_plot.set_projection(x_um, proj_x)
        self.y_projection_plot.set_projection(y_um, proj_y)

        peak_val = float(image.max())
        self._auto_adjust_exposure(peak_val, frame.camera_full_scale_counts)
        self.control_panel.update_saturation_headroom(peak_val, frame.camera_full_scale_counts)
        self._update_fps()

    # -- Brightness / stretch -------------------------------------------------

    @QtCore.Slot(float)
    def _on_brightness_changed(self, scale: float) -> None:
        self._brightness_scale = max(0.05, float(scale))
        self._refresh_last_frame_display()

    @QtCore.Slot(bool)
    def _on_auto_stretch_changed(self, enabled: bool) -> None:
        self._auto_stretch = enabled
        self._refresh_last_frame_display()

    @QtCore.Slot(int)
    def _on_saturation_changed(self, value: int) -> None:
        self._saturation_level = value

    # -- FPS ------------------------------------------------------------------

    def _update_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 2.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / dt if dt > 0 else 0.0
            self._fps_label.setText(f"FPS: {fps:.1f}")
        else:
            self._fps_label.setText("FPS: -")

    # -- ROI ------------------------------------------------------------------

    def _get_roi_region(
        self, frame: FrameData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (gray_sub, rgb_sub, x_um_sub, y_um_sub) for current ROI."""
        gray = frame.gray_image
        rgb = frame.rgb_image
        x_um = frame.x_coordinates_um
        y_um = frame.y_coordinates_um
        if not self.image_view.roi_rect.isVisible():
            return gray, rgb, x_um, y_um
        pos = self.image_view.roi_rect.pos()
        size = self.image_view.roi_rect.size()
        roi_x0, roi_y0 = float(pos.x()), float(pos.y())
        roi_x1 = roi_x0 + float(size.x())
        roi_y1 = roi_y0 + float(size.y())
        col0 = int(np.clip(np.searchsorted(x_um, roi_x0), 0, gray.shape[1] - 1))
        col1 = int(np.clip(np.searchsorted(x_um, roi_x1), 0, gray.shape[1]))
        row0 = int(np.clip(np.searchsorted(y_um, roi_y0), 0, gray.shape[0] - 1))
        row1 = int(np.clip(np.searchsorted(y_um, roi_y1), 0, gray.shape[0]))
        col1 = max(col1, col0 + 1)
        row1 = max(row1, row0 + 1)
        return (
            gray[row0:row1, col0:col1],
            rgb[row0:row1, col0:col1, :],
            x_um[col0:col1],
            y_um[row0:row1],
        )

    def _compute_roi_metrics(
        self, sub_gray: np.ndarray, sub_x: np.ndarray, sub_y: np.ndarray
    ) -> tuple[BeamMetrics, np.ndarray, np.ndarray]:
        proj_x = sub_gray.sum(axis=0).astype(np.float64)
        proj_y = sub_gray.sum(axis=1).astype(np.float64)
        sum_x = proj_x.sum()
        sum_y = proj_y.sum()
        cx = float((sub_x * proj_x).sum() / sum_x) if sum_x > 0 else 0.0
        cy = float((sub_y * proj_y).sum() / sum_y) if sum_y > 0 else 0.0
        return (
            BeamMetrics(
                centroid_x_um=cx,
                centroid_y_um=cy,
                fwhm_x_um=compute_fwhm_1d(proj_x) * PIXEL_SIZE_UM,
                fwhm_y_um=compute_fwhm_1d(proj_y) * PIXEL_SIZE_UM,
                peak_value=float(sub_gray.max()),
                sum_intensity=float(sub_gray.sum()),
            ),
            proj_x,
            proj_y,
        )

    def _slice_metrics(
        self,
        sub_image: np.ndarray,
        sub_x: np.ndarray,
        sub_y: np.ndarray,
        base_metrics: BeamMetrics,
    ) -> BeamMetrics:
        if self._slice_mode == "peak":
            peak_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)
            prof_x = sub_image[peak_idx[0], :].astype(np.float64)
            prof_y = sub_image[:, peak_idx[1]].astype(np.float64)
        else:
            cursor_y_um = self.image_view.slice_h_line.value()
            cursor_x_um = self.image_view.slice_v_line.value()
            row_idx = int(np.clip(np.argmin(np.abs(sub_y - cursor_y_um)), 0, sub_image.shape[0] - 1))
            col_idx = int(np.clip(np.argmin(np.abs(sub_x - cursor_x_um)), 0, sub_image.shape[1] - 1))
            prof_x = sub_image[row_idx, :].astype(np.float64)
            prof_y = sub_image[:, col_idx].astype(np.float64)
        fwhm_x = compute_fwhm_1d(prof_x) * PIXEL_SIZE_UM
        fwhm_y = compute_fwhm_1d(prof_y) * PIXEL_SIZE_UM
        sx = prof_x.sum()
        sy = prof_y.sum()
        cx = float((sub_x * prof_x).sum() / sx) if sx > 0 else base_metrics.centroid_x_um
        cy = float((sub_y * prof_y).sum() / sy) if sy > 0 else base_metrics.centroid_y_um
        return BeamMetrics(
            centroid_x_um=cx,
            centroid_y_um=cy,
            fwhm_x_um=fwhm_x,
            fwhm_y_um=fwhm_y,
            peak_value=base_metrics.peak_value,
            sum_intensity=base_metrics.sum_intensity,
        )

    def _update_projections_cached(
        self,
        sub_image: np.ndarray,
        sub_x_um: np.ndarray,
        sub_y_um: np.ndarray,
        roi_proj_x: np.ndarray,
        roi_proj_y: np.ndarray,
    ) -> None:
        if self._slice_mode == "sum":
            proj_x = roi_proj_x
            proj_y = roi_proj_y
        elif self._slice_mode == "peak":
            peak_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)
            proj_x = sub_image[peak_idx[0], :].astype(np.float64)
            proj_y = sub_image[:, peak_idx[1]].astype(np.float64)
            self.image_view.horizontal_line.setValue(sub_y_um[peak_idx[0]])
            self.image_view.vertical_line.setValue(sub_x_um[peak_idx[1]])
        else:
            cursor_y_um = self.image_view.slice_h_line.value()
            cursor_x_um = self.image_view.slice_v_line.value()
            row_idx = int(np.clip(np.argmin(np.abs(sub_y_um - cursor_y_um)), 0, sub_image.shape[0] - 1))
            col_idx = int(np.clip(np.argmin(np.abs(sub_x_um - cursor_x_um)), 0, sub_image.shape[1] - 1))
            proj_x = sub_image[row_idx, :].astype(np.float64)
            proj_y = sub_image[:, col_idx].astype(np.float64)
        self.x_projection_plot.set_projection(sub_x_um, proj_x)
        self.y_projection_plot.set_projection(sub_y_um, proj_y)

    @QtCore.Slot(bool)
    def _on_roi_toggled(self, enabled: bool) -> None:
        self.image_view.roi_rect.setVisible(enabled)
        self._apply_zoom_to_roi_view()
        self._reset_trends()
        self._cached_roi = None
        self._cached_roi_projections = None

    @QtCore.Slot()
    def _on_roi_changed(self) -> None:
        if self._last_frame is not None and self.image_view.roi_rect.isVisible():
            sub_gray, sub_rgb, sub_x, sub_y = self._get_roi_region(self._last_frame)
            self._cached_roi = (sub_gray, sub_rgb, sub_x, sub_y)
            roi_metrics, roi_proj_x, roi_proj_y = self._compute_roi_metrics(sub_gray, sub_x, sub_y)
            self._cached_roi_projections = (roi_proj_x, roi_proj_y)
            self._update_projections_cached(sub_gray, sub_x, sub_y, roi_proj_x, roi_proj_y)
            self.image_view.horizontal_line.setValue(roi_metrics.centroid_y_um)
            self.image_view.vertical_line.setValue(roi_metrics.centroid_x_um)
            self.control_panel.update_metrics(
                roi_metrics, float(self._last_frame.gray_image.sum())
            )
        self._apply_zoom_to_roi_view()

    @QtCore.Slot(bool)
    def _on_zoom_to_roi_toggled(self, enabled: bool) -> None:
        self._zoom_to_roi = enabled
        self._apply_zoom_to_roi_view()

    def _apply_zoom_to_roi_view(self) -> None:
        if self._zoom_to_roi and self.image_view.roi_rect.isVisible():
            pos = self.image_view.roi_rect.pos()
            size = self.image_view.roi_rect.size()
            self.image_view.setXRange(float(pos.x()), float(pos.x() + size.x()), padding=0.02)
            self.image_view.setYRange(float(pos.y()), float(pos.y() + size.y()), padding=0.02)
            return
        half_w = SENSOR_WIDTH_UM / 2.0
        half_h = SENSOR_HEIGHT_UM / 2.0
        self.image_view.setXRange(-half_w, half_w, padding=0.02)
        self.image_view.setYRange(-half_h, half_h, padding=0.02)

    @QtCore.Slot()
    def _on_fluence_params_changed(self) -> None:
        if self._last_frame is not None:
            m = self._last_frame.metrics
            self.control_panel.update_fluence(m.fwhm_x_um, m.fwhm_y_um)

    @QtCore.Slot(str)
    def _on_slice_mode_changed(self, mode: str) -> None:
        self._slice_mode = mode
        self._reset_trends()
        cursor_visible = mode == "cursor"
        self.image_view.slice_h_line.setVisible(cursor_visible)
        self.image_view.slice_v_line.setVisible(cursor_visible)
        if self._last_frame is not None and mode == "cursor":
            self.image_view.slice_h_line.setValue(self._last_frame.metrics.centroid_y_um)
            self.image_view.slice_v_line.setValue(self._last_frame.metrics.centroid_x_um)

    @QtCore.Slot()
    def _on_cursor_moved(self) -> None:
        if self._last_frame is not None and self._slice_mode == "cursor":
            if not self._cursor_timer.isActive():
                self._cursor_timer.start()

    @QtCore.Slot()
    def _on_cursor_timer(self) -> None:
        if self._cached_roi is not None and self._slice_mode == "cursor":
            sub_gray, _, sub_x, sub_y = self._cached_roi
            proj_x = self._cached_roi_projections[0] if self._cached_roi_projections else sub_gray.sum(axis=0).astype(np.float64)
            proj_y = self._cached_roi_projections[1] if self._cached_roi_projections else sub_gray.sum(axis=1).astype(np.float64)
            self._update_projections_cached(sub_gray, sub_x, sub_y, proj_x, proj_y)

    @QtCore.Slot(bool)
    def _on_auto_exposure_toggled(self, enabled: bool) -> None:
        self._auto_exposure = enabled

    def _auto_adjust_exposure(self, peak_counts: float, full_scale_counts: float) -> None:
        if not self._auto_exposure or full_scale_counts <= 0:
            return
        threshold = float(self._saturation_level)
        if threshold <= 0 or peak_counts <= 0:
            return
        ratio = peak_counts / threshold
        target = 0.70
        if 0.60 <= ratio <= 0.80:
            return
        current_ms = self.control_panel.exposure_time.value()
        min_ms = max(0.01, self.control_panel.exposure_time.minimum())
        max_ms = 100.0
        ideal_factor = target / ratio
        damped_factor = ideal_factor ** 0.30
        new_ms = max(min_ms, min(max_ms, current_ms * damped_factor))
        new_ms = round(new_ms, 1)
        if abs(new_ms - current_ms) < 0.05:
            return
        self.control_panel.exposure_time.blockSignals(True)
        self.control_panel.exposure_time.setValue(new_ms)
        self.control_panel.exposure_time.blockSignals(False)
        self.acquisition_thread.set_exposure_ms(new_ms)

    # -- Utilities ------------------------------------------------------------

    def _push_history(self, values: list[float], value: float) -> None:
        values.append(value)
        if len(values) > self.history_length:
            del values[0]

    @QtCore.Slot()
    def _reset_trends(self) -> None:
        for h in (
            self.position_x_history, self.position_y_history,
            self.fwhm_x_history, self.fwhm_y_history, self.sum_intensity_history,
        ):
            h.clear()
        for p in (
            self.position_x_plot, self.position_y_plot,
            self.fwhm_x_plot, self.fwhm_y_plot, self.sum_intensity_plot,
        ):
            p.set_series([])

    @QtCore.Slot(int)
    def _on_trend_length_changed(self, value: int) -> None:
        self.history_length = max(10, value)
        for h in (
            self.position_x_history, self.position_y_history,
            self.fwhm_x_history, self.fwhm_y_history, self.sum_intensity_history,
        ):
            while len(h) > self.history_length:
                del h[0]

    @QtCore.Slot(str)
    def _show_status(self, message: str) -> None:
        self.control_panel.set_backend_message(message)
        self.statusBar().showMessage(message, 5000)

    @QtCore.Slot(str)
    def _show_error(self, message: str) -> None:
        self.control_panel.set_backend_message(message)
        self.statusBar().showMessage(message, 8000)
        print(message)

    def _add_crosshair_marker(self) -> None:
        if self._last_frame is None:
            return
        m = self._last_frame.metrics
        marker = pg.TargetItem(
            pos=(m.centroid_x_um, m.centroid_y_um),
            size=16,
            pen=pg.mkPen("#00e5ff", width=2),
        )
        self.image_view.addItem(marker)
        self.image_view._crosshair_markers.append(marker)

    def _clear_crosshair_markers(self) -> None:
        for marker in self.image_view._crosshair_markers:
            self.image_view.removeItem(marker)
        self.image_view._crosshair_markers.clear()

    def _save_snapshot(self) -> None:
        if self._last_frame is None:
            self._show_status("No frame available for snapshot")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"{timestamp}_beamprofile_color.png"
        default_path = Path.cwd() / default_name
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Color Beam Snapshot",
            str(default_path),
            "PNG Image (*.png);;ASCII Text (*.txt)",
        )
        if not file_path:
            return

        if selected_filter == "ASCII Text (*.txt)" or file_path.lower().endswith(".txt"):
            self._save_ascii(file_path)
            return

        # --- PNG export of colour image ---
        rgb = self._last_frame.rgb_image  # H×W×3 uint8
        height, width, _ = rgb.shape
        rgb_contiguous = np.ascontiguousarray(rgb)
        qimage = QtGui.QImage(
            rgb_contiguous.data,
            width,
            height,
            3 * width,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        if not qimage.save(file_path):
            self._show_error(f"Failed to save snapshot to {file_path}")
            return

        saved = [file_path]

        # --- Analysis figure ---
        if MPL_AVAILABLE:
            fig_path = str(Path(file_path).with_name(
                Path(file_path).stem + "_analysis.png"
            ))
            try:
                self._save_analysis_figure(fig_path)
                saved.append(fig_path)
            except Exception as exc:
                self._show_error(f"Analysis figure failed: {exc}")

        # --- H5 export ---
        if XRAY_AVAILABLE:
            h5_path = str(Path(file_path).with_suffix(".h5"))
            try:
                frame = self._last_frame
                dataset = xr.Dataset(
                    data_vars={
                        "luminance": (
                            ("y_um", "x_um"),
                            frame.gray_image.astype(np.float32),
                        ),
                        "rgb": (
                            ("y_um", "x_um", "channel"),
                            frame.rgb_image,
                        ),
                        "projection_x": (("x_um",), frame.projection_x.astype(np.float32)),
                        "projection_y": (("y_um",), frame.projection_y.astype(np.float32)),
                    },
                    coords={
                        "x_um": frame.x_coordinates_um.astype(np.float64),
                        "y_um": frame.y_coordinates_um.astype(np.float64),
                        "channel": ["R", "G", "B"],
                    },
                    attrs={
                        "pixel_size_um": PIXEL_SIZE_UM,
                        "centroid_x_um": frame.metrics.centroid_x_um,
                        "centroid_y_um": frame.metrics.centroid_y_um,
                        "fwhm_x_um": frame.metrics.fwhm_x_um,
                        "fwhm_y_um": frame.metrics.fwhm_y_um,
                        "display_mode": "color",
                    },
                )
                dataset.to_netcdf(h5_path)
                saved.append(h5_path)
            except Exception as exc:
                self._show_error(f"H5 export failed: {exc}")

        self._show_status(f"Saved: {', '.join(saved)}")

    def _save_ascii(self, file_path: str) -> None:
        if self._last_frame is None:
            return
        frame = self._last_frame
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# IDS Color Beam Snapshot — luminance values (float32)\n")
                f.write(f"# pixel_size_um={PIXEL_SIZE_UM}\n")
                f.write(f"# centroid_x_um={frame.metrics.centroid_x_um:.3f}\n")
                f.write(f"# centroid_y_um={frame.metrics.centroid_y_um:.3f}\n")
                f.write(f"# fwhm_x_um={frame.metrics.fwhm_x_um:.3f}\n")
                f.write(f"# fwhm_y_um={frame.metrics.fwhm_y_um:.3f}\n")
                np.savetxt(f, frame.gray_image, fmt="%.4f", delimiter="\t")
            self._show_status(f"ASCII saved: {file_path}")
        except OSError as exc:
            self._show_error(f"Failed to save ASCII: {exc}")

    def _save_analysis_figure(self, fig_path: str) -> None:
        frame = self._last_frame
        if frame is None or plt is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0b0d10")
        for ax in axes.flat:
            ax.set_facecolor("#15181d")
            for spine in ax.spines.values():
                spine.set_color("#444c56")
            ax.tick_params(colors="#d8dee9")
            ax.xaxis.label.set_color("#d8dee9")
            ax.yaxis.label.set_color("#d8dee9")
            ax.title.set_color("#d8dee9")

        extent = [
            float(frame.x_coordinates_um[0]),
            float(frame.x_coordinates_um[-1]),
            float(frame.y_coordinates_um[0]),
            float(frame.y_coordinates_um[-1]),
        ]

        # Colour image
        ax_img = axes[0, 0]
        ax_img.imshow(
            frame.rgb_image,
            extent=extent,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        ax_img.set_title("Beam Image (color)")
        ax_img.set_xlabel("X (µm)")
        ax_img.set_ylabel("Y (µm)")
        ax_img.axhline(frame.metrics.centroid_y_um, color="#e040fb", lw=0.8, ls="--")
        ax_img.axvline(frame.metrics.centroid_x_um, color="#e040fb", lw=0.8, ls="--")

        # Luminance image
        ax_lum = axes[0, 1]
        im = ax_lum.imshow(
            frame.gray_image, extent=extent, origin="lower", aspect="equal",
            cmap="magma", interpolation="nearest",
        )
        fig.colorbar(im, ax=ax_lum, label="Counts")
        ax_lum.set_title("Beam Image (luminance)")
        ax_lum.set_xlabel("X (µm)")
        ax_lum.set_ylabel("Y (µm)")

        # X projection
        ax_x = axes[1, 0]
        ax_x.plot(frame.x_coordinates_um, frame.projection_x, "#ffb347", lw=1.5)
        ax_x.axvline(frame.metrics.centroid_x_um, color="#e040fb", lw=0.8, ls="--")
        ax_x.set_xlabel("X (µm)")
        ax_x.set_ylabel("Counts")
        ax_x.set_title(
            f"Horizontal projection  FWHM = {frame.metrics.fwhm_x_um:.1f} µm"
        )

        # Y projection
        ax_y = axes[1, 1]
        ax_y.plot(frame.projection_y, frame.y_coordinates_um, "#ff8fab", lw=1.5)
        ax_y.axhline(frame.metrics.centroid_y_um, color="#e040fb", lw=0.8, ls="--")
        ax_y.set_ylabel("Y (µm)")
        ax_y.set_xlabel("Counts")
        ax_y.set_title(
            f"Vertical projection  FWHM = {frame.metrics.fwhm_y_um:.1f} µm"
        )

        fig.suptitle(
            f"IDS Color Beamprofiler — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Centroid: ({frame.metrics.centroid_x_um:.1f}, {frame.metrics.centroid_y_um:.1f}) µm  "
            f"FWHM: ({frame.metrics.fwhm_x_um:.1f} × {frame.metrics.fwhm_y_um:.1f}) µm",
            color="#d8dee9",
        )
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.acquisition_thread.stop_thread()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = BeamProfilerApp()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
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

DCX_SDK_ROOT = Path(__file__).resolve().parent / "dcx_camera_interfaces_2018_09"
DCX_DLL_PATH = DCX_SDK_ROOT / "DCx_Camera_Drivers" / "64" / "uc480_64.dll"
DCX_BACKEND_AVAILABLE = DCX_DLL_PATH.exists()


pg.setConfigOptions(antialias=True, background="#101215", foreground="#d8dee9")

PIXEL_SIZE_UM = 5.2
SENSOR_WIDTH_PIXELS = 1280
SENSOR_HEIGHT_PIXELS = 1024
SENSOR_WIDTH_UM = 6656.0
SENSOR_HEIGHT_UM = 5325.0

# Rotation applied to incoming camera image for display alignment.
# Camera is physically rotated 90°, so no software rotation needed.
# The long sensor axis (3088 px) is now horizontal → >7000 µm on x.
DISPLAY_ROTATE_K = 0

FLUENCE_MODE_FWHM = "fwhm"
FLUENCE_MODE_ONE_OVER_E = "one_over_e"


def get_screen_scale_factor() -> float:
    """Calculate font scaling based on physical screen DPI/resolution."""
    from PySide6.QtGui import QGuiApplication
    
    # Get primary screen
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return 1.0
    
    # Get DPI and size
    dpi = screen.logicalDotsPerInch()
    geometry = screen.geometry()
    width_px = geometry.width()
    
    # Reference: 96 DPI at 1920px width = scale 1.0
    # Formula: (current_dpi / 96) * (current_width / 1920)
    dpi_scale = dpi / 96.0
    width_scale = width_px / 1920.0
    scale = (dpi_scale * width_scale) / 1.5  # Reduce by 1.5x for better readability
    
    return max(0.7, min(scale, 2.0))  # Clamp between 0.7 and 2.0


def scale_font_size(base_size: int) -> int:
    """Scale a base font size according to screen resolution."""
    scale = get_screen_scale_factor()
    return max(8, int(round(base_size * scale)))


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
    gray_image: np.ndarray
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


class DcxBoardInfo(ctypes.Structure):
    _fields_ = [
        ("serial_number", ctypes.c_char * 12),
        ("camera_id", ctypes.c_char * 20),
        ("version", ctypes.c_char * 10),
        ("date", ctypes.c_char * 12),
        ("select", ctypes.c_ubyte),
        ("type", ctypes.c_ubyte),
        ("reserved", ctypes.c_char * 8),
    ]


class DcxSensorInfo(ctypes.Structure):
    _fields_ = [
        ("sensor_id", ctypes.c_ushort),
        ("sensor_name", ctypes.c_char * 32),
        ("color_mode", ctypes.c_char),
        ("max_width", ctypes.c_ulong),
        ("max_height", ctypes.c_ulong),
        ("master_gain_supported", ctypes.c_int),
        ("red_gain_supported", ctypes.c_int),
        ("green_gain_supported", ctypes.c_int),
        ("blue_gain_supported", ctypes.c_int),
        ("global_shutter_supported", ctypes.c_int),
        ("pixel_size_100nm", ctypes.c_ushort),
        ("upper_left_bayer_pixel", ctypes.c_char),
        ("reserved", ctypes.c_char * 13),
    ]


class DcxCamera:
    IS_SUCCESS = 0
    IS_WAIT = 1
    IS_CM_MONO8 = 6
    IS_IGNORE_PARAMETER = -1
    IS_GET_MASTER_GAIN = 0x8000
    IS_MIN_GAIN = 0
    IS_MAX_GAIN = 100
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN = 3
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX = 4
    IS_EXPOSURE_CMD_GET_EXPOSURE = 7
    IS_EXPOSURE_CMD_SET_EXPOSURE = 12

    def __init__(self, dll_path: Path | None = None, camera_id: int = 0) -> None:
        self._dll_path = dll_path or DCX_DLL_PATH
        if not self._dll_path.exists():
            raise RuntimeError(f"DCx SDK DLL not found: {self._dll_path}")

        os.add_dll_directory(str(self._dll_path.parent))
        self._lib = ctypes.WinDLL(str(self._dll_path))
        self._configure_signatures()

        self._handle = ctypes.c_int(camera_id | 0x8000 if camera_id > 0 else 0)
        self._image_ptr = ctypes.c_void_p()
        self._image_id = ctypes.c_int(0)
        self.width = SENSOR_WIDTH_PIXELS
        self.height = SENSOR_HEIGHT_PIXELS
        self.bits_per_pixel = 8
        self.pitch = self.width
        self._board_info = DcxBoardInfo()
        self._sensor_info = DcxSensorInfo()

        self._open()

    @staticmethod
    def get_num_cameras(dll_path: Path | None = None) -> int:
        """Return the number of connected DCx cameras."""
        p = dll_path or DCX_DLL_PATH
        if not p.exists():
            return 0
        os.add_dll_directory(str(p.parent))
        lib = ctypes.WinDLL(str(p))
        lib.is_GetNumberOfCameras.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.is_GetNumberOfCameras.restype = ctypes.c_int
        num = ctypes.c_int(0)
        if lib.is_GetNumberOfCameras(ctypes.byref(num)) != 0:
            return 0
        return num.value

    @staticmethod
    def list_cameras(dll_path: Path | None = None) -> list[tuple[int, str]]:
        """Return a list of (device_id, serial_number) for each connected camera."""
        p = dll_path or DCX_DLL_PATH
        if not p.exists():
            return []
        os.add_dll_directory(str(p.parent))
        lib = ctypes.WinDLL(str(p))
        lib.is_GetNumberOfCameras.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.is_GetNumberOfCameras.restype = ctypes.c_int
        num = ctypes.c_int(0)
        if lib.is_GetNumberOfCameras(ctypes.byref(num)) != 0 or num.value == 0:
            return []
        n = num.value

        class UC480_CAMERA_INFO(ctypes.Structure):
            _fields_ = [
                ("dwCameraID", ctypes.c_ulong),
                ("dwDeviceID", ctypes.c_ulong),
                ("dwSensorID", ctypes.c_ulong),
                ("dwInUse", ctypes.c_ulong),
                ("SerNo", ctypes.c_char * 16),
                ("Model", ctypes.c_char * 16),
                ("dwStatus", ctypes.c_ulong),
                ("dwReserved", ctypes.c_ulong * 2),
                ("FullModelName", ctypes.c_char * 32),
                ("dwReserved2", ctypes.c_ulong * 5),
            ]

        class UC480_CAMERA_LIST(ctypes.Structure):
            _fields_ = [
                ("dwCount", ctypes.c_ulong),
                ("uci", UC480_CAMERA_INFO * n),
            ]

        cam_list = UC480_CAMERA_LIST()
        cam_list.dwCount = n
        lib.is_GetCameraList.argtypes = [ctypes.POINTER(UC480_CAMERA_LIST)]
        lib.is_GetCameraList.restype = ctypes.c_int
        if lib.is_GetCameraList(ctypes.byref(cam_list)) != 0:
            return []
        result: list[tuple[int, str]] = []
        for i in range(cam_list.dwCount):
            info = cam_list.uci[i]
            dev_id = int(info.dwDeviceID)
            serial = info.SerNo.split(b"\0", 1)[0].decode("ascii", errors="ignore").strip()
            result.append((dev_id, serial))
        return result

    def _configure_signatures(self) -> None:
        self._lib.is_InitCamera.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
        self._lib.is_InitCamera.restype = ctypes.c_int
        self._lib.is_ExitCamera.argtypes = [ctypes.c_int]
        self._lib.is_ExitCamera.restype = ctypes.c_int
        self._lib.is_GetCameraInfo.argtypes = [ctypes.c_int, ctypes.POINTER(DcxBoardInfo)]
        self._lib.is_GetCameraInfo.restype = ctypes.c_int
        self._lib.is_GetSensorInfo.argtypes = [ctypes.c_int, ctypes.POINTER(DcxSensorInfo)]
        self._lib.is_GetSensorInfo.restype = ctypes.c_int
        self._lib.is_SetColorMode.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.is_SetColorMode.restype = ctypes.c_int
        self._lib.is_AllocImageMem.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.is_AllocImageMem.restype = ctypes.c_int
        self._lib.is_SetImageMem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self._lib.is_SetImageMem.restype = ctypes.c_int
        self._lib.is_InquireImageMem.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.is_InquireImageMem.restype = ctypes.c_int
        self._lib.is_FreezeVideo.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.is_FreezeVideo.restype = ctypes.c_int
        self._lib.is_FreeImageMem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self._lib.is_FreeImageMem.restype = ctypes.c_int
        self._lib.is_Exposure.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        self._lib.is_Exposure.restype = ctypes.c_int
        self._lib.is_SetHardwareGain.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.is_SetHardwareGain.restype = ctypes.c_int

    def _check(self, result: int, action: str) -> None:
        if result != self.IS_SUCCESS:
            raise RuntimeError(f"{action} failed with error code {result}")

    @staticmethod
    def _decode(value: bytes) -> str:
        return value.split(b"\0", 1)[0].decode("ascii", errors="ignore").strip()

    def _open(self) -> None:
        self._check(self._lib.is_InitCamera(ctypes.byref(self._handle), None), "is_InitCamera")
        try:
            self._check(self._lib.is_GetCameraInfo(self._handle, ctypes.byref(self._board_info)), "is_GetCameraInfo")
            self._check(self._lib.is_GetSensorInfo(self._handle, ctypes.byref(self._sensor_info)), "is_GetSensorInfo")
            self._check(self._lib.is_SetColorMode(self._handle, self.IS_CM_MONO8), "is_SetColorMode")

            self.width = int(self._sensor_info.max_width)
            self.height = int(self._sensor_info.max_height)

            self._check(
                self._lib.is_AllocImageMem(
                    self._handle,
                    self.width,
                    self.height,
                    self.bits_per_pixel,
                    ctypes.byref(self._image_ptr),
                    ctypes.byref(self._image_id),
                ),
                "is_AllocImageMem",
            )
            self._check(self._lib.is_SetImageMem(self._handle, self._image_ptr, self._image_id), "is_SetImageMem")

            width = ctypes.c_int()
            height = ctypes.c_int()
            bits = ctypes.c_int()
            pitch = ctypes.c_int()
            self._check(
                self._lib.is_InquireImageMem(
                    self._handle,
                    self._image_ptr,
                    self._image_id,
                    ctypes.byref(width),
                    ctypes.byref(height),
                    ctypes.byref(bits),
                    ctypes.byref(pitch),
                ),
                "is_InquireImageMem",
            )
            self.width = width.value
            self.height = height.value
            self.bits_per_pixel = bits.value
            self.pitch = pitch.value
        except Exception:
            self.close()
            raise

    @property
    def model_name(self) -> str:
        sensor_name = self._decode(self._sensor_info.sensor_name)
        return f"Thorlabs DCC1545M-GL ({sensor_name})"

    @property
    def serial_number(self) -> str:
        return self._decode(self._board_info.serial_number)

    @property
    def pixel_size_um(self) -> float:
        pixel_size = float(self._sensor_info.pixel_size_100nm) / 100.0
        return pixel_size if pixel_size > 0 else PIXEL_SIZE_UM

    def _exposure_command(self, command: int, value: float | None = None) -> float:
        parameter = ctypes.c_double(0.0 if value is None else float(value))
        self._check(
            self._lib.is_Exposure(
                self._handle,
                command,
                ctypes.byref(parameter),
                ctypes.sizeof(parameter),
            ),
            f"is_Exposure({command})",
        )
        return float(parameter.value)

    def get_exposure_ms(self) -> float:
        return self._exposure_command(self.IS_EXPOSURE_CMD_GET_EXPOSURE)

    def set_exposure_ms(self, exposure_ms: float) -> float:
        return self._exposure_command(self.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_ms)

    def get_exposure_range_ms(self) -> tuple[float, float]:
        minimum = self._exposure_command(self.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN)
        maximum = self._exposure_command(self.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX)
        return float(minimum), float(maximum)

    def get_gain(self) -> float:
        return float(
            self._lib.is_SetHardwareGain(
                self._handle,
                self.IS_GET_MASTER_GAIN,
                self.IS_IGNORE_PARAMETER,
                self.IS_IGNORE_PARAMETER,
                self.IS_IGNORE_PARAMETER,
            )
        )

    def set_gain(self, gain: float) -> float:
        applied = int(np.clip(round(gain), self.IS_MIN_GAIN, self.IS_MAX_GAIN))
        result = self._lib.is_SetHardwareGain(
            self._handle,
            applied,
            self.IS_IGNORE_PARAMETER,
            self.IS_IGNORE_PARAMETER,
            self.IS_IGNORE_PARAMETER,
        )
        if result == self.IS_SUCCESS:
            return float(applied)
        raise RuntimeError(f"is_SetHardwareGain failed with error code {result}")

    def capture_frame(self) -> np.ndarray:
        self._check(self._lib.is_FreezeVideo(self._handle, self.IS_WAIT), "is_FreezeVideo")
        buffer = (ctypes.c_ubyte * (self.pitch * self.height)).from_address(self._image_ptr.value)
        frame = np.ctypeslib.as_array(buffer).reshape((self.height, self.pitch))
        return np.array(frame[:, : self.width], dtype=np.float32, copy=True)

    def camera_state(self) -> CameraState:
        exposure_min_ms, exposure_max_ms = self.get_exposure_range_ms()
        return CameraState(
            model_name=self.model_name,
            serial_number=self.serial_number,
            exposure_ms=self.get_exposure_ms(),
            exposure_min_ms=exposure_min_ms,
            exposure_max_ms=exposure_max_ms,
            gain_value=self.get_gain(),
            gain_min=float(self.IS_MIN_GAIN),
            gain_max=float(self.IS_MAX_GAIN),
        )

    def close(self) -> None:
        if getattr(self, "_image_ptr", None) is not None and self._image_ptr.value:
            try:
                self._lib.is_FreeImageMem(self._handle, self._image_ptr, self._image_id)
            except Exception:
                pass
            self._image_ptr = ctypes.c_void_p()
            self._image_id = ctypes.c_int(0)
        if getattr(self, "_handle", None) is not None and self._handle.value:
            try:
                self._lib.is_ExitCamera(self._handle)
            except Exception:
                pass
            self._handle = ctypes.c_int(0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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


def normalize_image(image: np.ndarray) -> np.ndarray:
    data = image.astype(np.float32)
    data -= data.min()
    peak = data.max()
    if peak > 0:
        data /= peak
    return data


def get_colormap(name: str) -> pg.ColorMap:
    normalized = name.strip().lower()
    try:
        return pg.colormap.get(normalized)
    except Exception:
        try:
            return pg.colormap.getFromMatplotlib(normalized)
        except Exception:
            return pg.colormap.get("gray")


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
    """Return fluence in µJ/cm² using a selectable top-hat beam approximation.

    Parameters
    ----------
    power_uW : incident power in µW
    fwhm_x, fwhm_y : FWHM in µm
    angle : angle of incidence in degrees
    rep_rate : repetition rate in Hz
    """
    if fwhm_x <= 0 or fwhm_y <= 0 or rep_rate <= 0:
        return 0.0
    angle_rad = np.radians(angle)
    if mode == FLUENCE_MODE_ONE_OVER_E:
        x0 = fwhm_x / (2.0 * np.sqrt(np.log(2.0))) * 1e-4
        y0 = fwhm_y / (2.0 * np.sqrt(np.log(2.0))) * 1e-4
    else:
        x0 = fwhm_x / 2.0 * 1e-4
        y0 = fwhm_y / 2.0 * 1e-4
    area = np.pi * x0 * y0  # cm²
    fluence = power_uW * np.cos(angle_rad) / (rep_rate * area)  # µJ/cm²
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
    centroid_x_um = (
        float((x_coords_um * projection_x).sum() / sum_x) if sum_x > 0 else 0.0
    )
    centroid_y_um = (
        float((y_coords_um * projection_y).sum() / sum_y) if sum_y > 0 else 0.0
    )

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
    if image_data.ndim == 2:
        return image_data.astype(np.float32)

    if image_data.ndim == 3:
        if image_data.shape[2] >= 3:
            rgb = image_data[..., :3].astype(np.float32)
            return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
                np.float32
            )
        return image_data.mean(axis=2, dtype=np.float32)

    flat = image_data.astype(np.float32)
    side = int(math.sqrt(flat.size))
    return flat.reshape((side, side))


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

        # Set fixed view range once — never call setRange per-frame
        half_w = SENSOR_WIDTH_UM / 2.0
        half_h = SENSOR_HEIGHT_UM / 2.0
        self.setXRange(-half_w, half_w, padding=0.02)
        self.setYRange(-half_h, half_h, padding=0.02)

        self.horizontal_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#e040fb", width=1))
        self.vertical_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#e040fb", width=1))
        self.addItem(self.horizontal_line)
        self.addItem(self.vertical_line)

        # Draggable cursor lines for slice mode
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

        # ROI rectangle — defaults to full sensor area
        half_w = SENSOR_WIDTH_UM / 2.0
        half_h = SENSOR_HEIGHT_UM / 2.0
        self.roi_rect = pg.RectROI(
            [-half_w, -half_h], [SENSOR_WIDTH_UM, SENSOR_HEIGHT_UM],
            pen=pg.mkPen("#ffe135", width=2),
            hoverPen=pg.mkPen("#fff176", width=2),
            handlePen=pg.mkPen("#ffe135", width=2),
        )
        self.roi_rect.setVisible(False)
        self.addItem(self.roi_rect)

        # Ensure cursor lines are above ROI so they can be dragged even when ROI is visible
        self.slice_h_line.setZValue(self.roi_rect.zValue() + 10)
        self.slice_v_line.setZValue(self.roi_rect.zValue() + 10)

        # Crosshair markers list
        self._crosshair_markers: list[pg.TargetItem] = []

        self.color_bar = pg.ColorBarItem(
            values=(0.0, 1.0),
            interactive=False,
            colorMap=get_colormap("magma"),
            width=14,
            label="Counts",
        )
        self.color_bar.setImageItem(self.image_item, insert_in=self.getPlotItem())

    def set_colormap(self, colormap_name: str) -> None:
        self.color_bar.setColorMap(get_colormap(colormap_name))

    def set_color_levels(self, minimum: float, maximum: float, label: str) -> None:
        self.image_item.setLevels((minimum, maximum))
        self.color_bar.setLevels((minimum, maximum))
        if hasattr(self.color_bar, "axis"):
            self.color_bar.axis.setLabel(text=label)

    def set_image(self, frame: FrameData) -> None:
        x_coords_um = frame.x_coordinates_um
        y_coords_um = frame.y_coordinates_um
        pixel_width = PIXEL_SIZE_UM
        pixel_height = PIXEL_SIZE_UM
        x_min = float(x_coords_um[0] - pixel_width / 2.0)
        y_min = float(y_coords_um[0] - pixel_height / 2.0)
        width_um = float(x_coords_um[-1] - x_coords_um[0] + pixel_width)
        height_um = float(y_coords_um[-1] - y_coords_um[0] + pixel_height)

        # Downsample for display — the widget is much smaller than the sensor
        img = frame.gray_image
        if img.shape[0] > 1024 or img.shape[1] > 1024:
            step = max(img.shape[0] // 1024, img.shape[1] // 1024, 1)
            img = img[::step, ::step]

        self.image_item.setImage(img, autoLevels=False)
        self.image_item.setRect(QtCore.QRectF(x_min, y_min, width_um, height_um))
        self.horizontal_line.setValue(frame.metrics.centroid_y_um)
        self.vertical_line.setValue(frame.metrics.centroid_x_um)


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


class ControlPanel(QtWidgets.QFrame):
    exposure_changed = QtCore.Signal(float)
    average_changed = QtCore.Signal(int)
    start_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    snapshot_requested = QtCore.Signal()
    colormap_changed = QtCore.Signal(str)
    intensity_scale_mode_changed = QtCore.Signal(str)
    scale_colormap_changed = QtCore.Signal(bool)
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

        title = QtWidgets.QLabel("Beamprofiler Controls")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.camera_info_button = QtWidgets.QPushButton("Camera Info")
        self.camera_info_button.clicked.connect(self._show_camera_info_popup)
        layout.addWidget(self.camera_info_button)

        # Camera info values are kept as labels and shown via popup to save vertical space.
        self.model_label = QtWidgets.QLabel("Thorlabs DCC1545M-GL")
        self.serial_label = QtWidgets.QLabel("-")
        self.resolution_label = QtWidgets.QLabel(f"{SENSOR_WIDTH_PIXELS} x {SENSOR_HEIGHT_PIXELS}")
        self.pixel_size_label = QtWidgets.QLabel(f"{PIXEL_SIZE_UM:.1f} um")
        self.chip_size_label = QtWidgets.QLabel(f"{SENSOR_WIDTH_UM / 1000.0:.2f} x {SENSOR_HEIGHT_UM / 1000.0:.2f} mm")

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

        beam_group = QtWidgets.QGroupBox("Beam Analysis")
        beam_form = QtWidgets.QFormLayout(beam_group)
        beam_form.setContentsMargins(4, 4, 4, 4)
        beam_form.setSpacing(2)
        self.pixel_saturation = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pixel_saturation.setRange(1, 255)
        self.pixel_saturation.setValue(255)
        self.scale_colormap = QtWidgets.QCheckBox("Scale colormap automatically")
        self.scale_colormap.setChecked(True)

        self.intensity_scale_selector = QtWidgets.QComboBox()
        self.intensity_scale_selector.addItems(["Linear", "Logarithmic"])

        self.colormap_selector = QtWidgets.QComboBox()
        self.colormap_selector.addItems(["Gray", "Viridis", "Inferno", "Plasma", "Magma", "Cividis", "Turbo"])
        self.colormap_selector.setCurrentText("Magma")

        self.saturation_value_label = QtWidgets.QLabel("255 counts")

        beam_form.addRow("Pixel saturation", self.pixel_saturation)
        beam_form.addRow("Saturation value", self.saturation_value_label)
        beam_form.addRow("Intensity scale", self.intensity_scale_selector)
        beam_form.addRow("Colormap", self.colormap_selector)
        beam_form.addRow("Auto scale", self.scale_colormap)
        layout.addWidget(beam_group)

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
        self.saturation_limit_label = QtWidgets.QLabel("255 counts")
        self.saturation_limit_label.setStyleSheet(f"font-weight: 600; color: #eef2f6; font-size: {scale_font_size(13)}px;")
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
        pos_x_title = QtWidgets.QLabel("Position X")
        pos_x_title.setStyleSheet(title_style)
        pos_y_title = QtWidgets.QLabel("Position Y")
        pos_y_title.setStyleSheet(title_style)
        sum_intensity_title = QtWidgets.QLabel("Sum intensity")
        sum_intensity_title.setStyleSheet(title_style)
        fwhm_x_title = QtWidgets.QLabel("FWHM X")
        fwhm_x_title.setStyleSheet(title_style)
        fwhm_y_title = QtWidgets.QLabel("FWHM Y")
        fwhm_y_title.setStyleSheet(title_style)
        sat_level_title = QtWidgets.QLabel("Saturation level")
        sat_level_title.setStyleSheet(title_style)

        metrics_layout.addWidget(pos_x_title, 0, 0)
        metrics_layout.addWidget(self.position_x_label, 0, 1)
        metrics_layout.addWidget(fwhm_x_title, 0, 2)
        metrics_layout.addWidget(self.fwhm_x_label, 0, 3)

        metrics_layout.addWidget(pos_y_title, 1, 0)
        metrics_layout.addWidget(self.position_y_label, 1, 1)
        metrics_layout.addWidget(fwhm_y_title, 1, 2)
        metrics_layout.addWidget(self.fwhm_y_label, 1, 3)

        metrics_layout.addWidget(sum_intensity_title, 2, 0)
        metrics_layout.addWidget(self.sum_intensity_label, 2, 1)
        metrics_layout.addWidget(sat_level_title, 2, 2)
        metrics_layout.addWidget(self.saturation_progress, 2, 3)
        layout.addWidget(metrics_group)

        # --- Fluence calculation group ---
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
            f"background: #10141a; border: 1px solid #2c313a; border-radius: 6px; padding: 4px;"
            f" color: #d8dee9;"
        )

        self.fluence_label = QtWidgets.QLabel("0.0 µJ/cm²")
        self.fluence_label.setStyleSheet(f"font-weight: 800; color: #ffb347; font-size: {scale_font_size(18)}px;")

        fluence_form.addRow("Power", self.power_input)
        fluence_form.addRow("Rep. rate", self.rep_rate_input)
        fluence_form.addRow("AOI", self.aoi_input)
        fluence_form.addRow("Model", fluence_mode_widget)
        fluence_form.addRow("Formula", self.fluence_formula_label)
        fluence_form.addRow("Fluence", self.fluence_label)
        layout.addWidget(fluence_group)

        # --- Projection / Slice mode group ---
        self.slice_group = QtWidgets.QGroupBox("Projection Mode")
        self.slice_group.setMaximumHeight(132)
        slice_layout = QtWidgets.QVBoxLayout(self.slice_group)
        slice_layout.setContentsMargins(4, 4, 4, 4)
        slice_layout.setSpacing(1)
        self.slice_mode_selector = QtWidgets.QButtonGroup(self)
        self.radio_sum = QtWidgets.QRadioButton("Sum projection")
        self.radio_cursor = QtWidgets.QRadioButton("Slice at cursor")
        self.radio_peak = QtWidgets.QRadioButton("Slice at peak intensity")
        self.radio_peak.setChecked(True)
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

        # --- Marker buttons ---
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

        self.average_images.valueChanged.connect(self.average_changed.emit)
        self.exposure_time.valueChanged.connect(self.exposure_changed.emit)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.snapshot_button.clicked.connect(self.snapshot_requested.emit)
        self.colormap_selector.currentTextChanged.connect(self.colormap_changed.emit)
        self.intensity_scale_selector.currentTextChanged.connect(
            self.intensity_scale_mode_changed.emit
        )
        self.scale_colormap.toggled.connect(self.scale_colormap_changed.emit)
        self.pixel_saturation.valueChanged.connect(self.pixel_saturation_changed.emit)
        self.pixel_saturation.valueChanged.connect(
            lambda value: self.saturation_value_label.setText(f"{value} counts")
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
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Camera Information")
        msg_box.setText(details)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.setStyleSheet("QLabel { color: #ffffff; } QPushButton { color: #ffffff; }")
        msg_box.exec()

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

    def set_camera_full_scale(self, full_scale_counts: float) -> None:
        self.saturation_limit_label.setText(f"{max(1, int(round(full_scale_counts)))} counts")
        self.saturation_value_label.setText(f"{self.pixel_saturation.value()} counts")

    def update_saturation_headroom(self, peak_counts: float, full_scale_counts: float) -> None:
        # Display saturation utilization in an 8-bit style bar for quick visual feedback.
        full_scale = max(1.0, float(full_scale_counts))
        level_255 = int(np.clip(np.round((peak_counts / full_scale) * 255.0), 0, 255))
        self.saturation_progress.setValue(level_255)


class AcquisitionThread(QtCore.QThread):
    frame_ready = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    camera_ready = QtCore.Signal(object)
    error_occurred = QtCore.Signal(str)

    def __init__(self, pixel_size_um: float, camera_id: int = 0, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.pixel_size_um = pixel_size_um
        self._camera_id = camera_id
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

    def _run_synthetic(self, reason: str) -> None:
        self.status_changed.emit(reason)
        frame_index = 0
        while not self.isInterruptionRequested():
            desired_running, average_images, _, _ = self._snapshot_config()
            if not desired_running:
                self.msleep(50)
                continue

            image = self._generate_mock_frame(frame_index)
            frame_index += max(1, average_images)
            self.frame_ready.emit(self._build_frame(image, 255.0))
            self.msleep(50)

    def _generate_mock_frame(self, frame_index: int) -> np.ndarray:
        width = 640
        height = 480
        x_values = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y_values = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        t = frame_index / 20.0
        center_x = 0.20 * math.sin(t * 0.8)
        center_y = 0.14 * math.cos(t * 0.6)
        sigma_x = 0.16 + 0.02 * math.sin(t * 0.45)
        sigma_y = 0.12 + 0.02 * math.cos(t * 0.52)
        angle = 0.35 * math.sin(t * 0.3)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_rot = cos_a * (x_grid - center_x) + sin_a * (y_grid - center_y)
        y_rot = -sin_a * (x_grid - center_x) + cos_a * (y_grid - center_y)

        beam = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        side_lobe = 0.12 * np.exp(-0.5 * (((x_grid + 0.32) / 0.09) ** 2 + ((y_grid - 0.25) / 0.07) ** 2))
        rng = np.random.default_rng(frame_index)
        noise = rng.normal(0.0, 0.01, size=beam.shape)
        return np.clip(beam + side_lobe + noise, 0.0, None).astype(np.float32)

    def _build_frame(self, gray_image: np.ndarray, full_scale_counts: float) -> FrameData:
        oriented_image = np.rot90(gray_image, k=DISPLAY_ROTATE_K).astype(np.float32)
        metrics, projection_x, projection_y = compute_metrics(oriented_image, self.pixel_size_um)
        return FrameData(
            gray_image=oriented_image,
            projection_x=projection_x,
            projection_y=projection_y,
            x_coordinates_um=make_axis_um(oriented_image.shape[1], self.pixel_size_um),
            y_coordinates_um=make_axis_um(oriented_image.shape[0], self.pixel_size_um),
            metrics=metrics,
            camera_full_scale_counts=float(full_scale_counts),
        )

    def run(self) -> None:
        if not DCX_BACKEND_AVAILABLE:
            reason = f"Synthetic mode: DCx backend unavailable ({DCX_DLL_PATH})"
            self._run_synthetic(reason)
            return

        camera: Any | None = None
        accumulation: np.ndarray | None = None
        accumulated_frames = 0
        applied_exposure_ms: float | None = None
        applied_gain: float | None = None

        try:
            camera = DcxCamera(camera_id=self._camera_id)

            _, average_images, desired_exposure_ms, desired_gain = self._snapshot_config()
            if desired_exposure_ms is None:
                desired_exposure_ms = camera.get_exposure_ms()
            if desired_gain is None:
                desired_gain = camera.get_gain()

            self.camera_ready.emit(camera.camera_state())
            self.status_changed.emit("Live camera mode: DCx acquisition active")

            while not self.isInterruptionRequested():
                desired_running, average_images, desired_exposure_ms, desired_gain = self._snapshot_config()

                if not desired_running:
                    accumulation = None
                    accumulated_frames = 0
                    self.status_changed.emit("Live camera mode: acquisition stopped")
                    self.msleep(30)
                    continue

                if desired_exposure_ms is not None and desired_exposure_ms != applied_exposure_ms:
                    try:
                        applied_exposure_ms = camera.set_exposure_ms(desired_exposure_ms)
                    except Exception as exc:
                        self.error_occurred.emit(f"Failed to set exposure: {exc}")

                if desired_gain is not None and desired_gain != applied_gain:
                    try:
                        applied_gain = camera.set_gain(desired_gain)
                    except Exception as exc:
                        self.error_occurred.emit(f"Failed to set gain: {exc}")

                try:
                    gray_image = camera.capture_frame()
                    full_scale_counts = 255.0
                except Exception as exc:
                    self.error_occurred.emit(f"Failed to capture frame: {exc}")
                    self.msleep(20)
                    continue

                if accumulation is None:
                    accumulation = gray_image.astype(np.float64)
                else:
                    accumulation += gray_image
                accumulated_frames += 1

                if accumulated_frames < max(1, average_images):
                    continue

                averaged_image = (accumulation / accumulated_frames).astype(np.float32)
                accumulation = None
                accumulated_frames = 0
                self.frame_ready.emit(self._build_frame(averaged_image, full_scale_counts))

        except BaseException as exc:  # pragma: no cover - depends on runtime hardware state
            message = f"Falling back to synthetic mode: {exc}"
            self.error_occurred.emit(message)
            self._run_synthetic(message)
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass


class BeamProfilerApp(QtWidgets.QMainWindow):
    history_length = 180

    def __init__(self, camera_id: int = 0) -> None:
        super().__init__()
        self._camera_id = camera_id
        title = "Thorlabs DCC1545M-GL Beamprofiler"
        if camera_id > 0:
            title += f" — Camera {camera_id}"
        self.setWindowTitle(title)

        self.position_x_history: list[float] = []
        self.position_y_history: list[float] = []
        self.fwhm_x_history: list[float] = []
        self.fwhm_y_history: list[float] = []
        self.sum_intensity_history: list[float] = []
        self._last_frame: FrameData | None = None
        self._last_display_image: np.ndarray | None = None
        self._scale_mode = "Linear"
        self._auto_scale_enabled = True
        self._saturation_level = 255
        self._fixed_scale_limits: tuple[float, float] | None = None
        self._last_display_levels: tuple[float, float] | None = None
        self._last_colorbar_label = "Counts"
        self._slice_mode = "peak"  # "sum", "cursor", or "peak"
        self._pending_frame: FrameData | None = None
        self._auto_exposure = False
        self._simple_mode = True
        self._zoom_to_roi = False
        self._frame_times: list[float] = []  # timestamps for FPS calculation

        # Throttle timer for cursor movement
        self._cursor_timer = QtCore.QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(30)  # max ~33 updates/sec for cursor drag
        self._cursor_timer.timeout.connect(self._on_cursor_timer)

        # Cached ROI data from the last frame
        self._cached_roi: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cached_roi_projections: tuple[np.ndarray, np.ndarray] | None = None

        # Timer for frame coalescing — processes only the latest frame
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(33)  # ~30 FPS cap to keep UI responsive
        self._frame_timer.timeout.connect(self._process_pending_frame)

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

        image_header = QtWidgets.QLabel("Beam Image")
        image_header.setObjectName("sectionTitle")
        x_projection_header = QtWidgets.QLabel("Horizontal Projection")
        x_projection_header.setObjectName("sectionTitle")
        y_projection_header = QtWidgets.QLabel("Vertical Projection")
        y_projection_header.setObjectName("sectionTitle")

        center_layout.addWidget(image_header, 0, 0)
        center_layout.addWidget(y_projection_header, 0, 1)
        center_layout.addWidget(self.image_view, 1, 0)
        center_layout.addWidget(self.y_projection_plot, 1, 1)
        center_layout.addWidget(x_projection_header, 2, 0)
        center_layout.addWidget(self.x_projection_plot, 3, 0)

        # Move projection mode controls to the free bottom-right corner beside the x-projection.
        self.control_panel.layout().removeWidget(self.control_panel.slice_group)
        self.control_panel.slice_group.setParent(None)
        self.control_panel.slice_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        center_layout.addWidget(
            self.control_panel.slice_group,
            3,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignTop,
        )

        # Place markers directly below projection mode in the same right column.
        self.control_panel.layout().removeWidget(self.control_panel.marker_group)
        self.control_panel.marker_group.setParent(None)
        self.control_panel.marker_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        center_layout.addWidget(
            self.control_panel.marker_group,
            4,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignTop,
        )

        center_layout.setColumnStretch(0, 9)
        center_layout.setColumnStretch(1, 1)
        center_layout.setRowStretch(1, 5)
        center_layout.setRowStretch(3, 3)
        center_layout.setRowStretch(4, 0)

        center_splitter.addWidget(center_widget)

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
        self.trend_length_spin.setToolTip("Number of trend data points")
        self.trend_reset_button = QtWidgets.QPushButton("Reset")
        self.trend_reset_button.setToolTip("Clear all trend histories")
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

        trend_layout.addWidget(self.position_x_plot)
        trend_layout.addWidget(self.position_y_plot)
        trend_layout.addWidget(self.fwhm_x_plot)
        trend_layout.addWidget(self.fwhm_y_plot)
        trend_layout.addWidget(self.sum_intensity_plot)

        center_splitter.addWidget(trend_panel)
        center_splitter.setSizes([1300, 340])

        self._apply_styles()

        self.acquisition_thread = AcquisitionThread(pixel_size_um=PIXEL_SIZE_UM, camera_id=self._camera_id, parent=self)
        self.acquisition_thread.frame_ready.connect(self._enqueue_frame)
        self.acquisition_thread.status_changed.connect(self._show_status)
        self.acquisition_thread.camera_ready.connect(self._handle_camera_state)
        self.acquisition_thread.error_occurred.connect(self._show_error)

        self.control_panel.average_changed.connect(self.acquisition_thread.set_average_images)
        self.control_panel.exposure_changed.connect(self.acquisition_thread.set_exposure_ms)
        self.control_panel.start_requested.connect(self.acquisition_thread.request_start)
        self.control_panel.stop_requested.connect(self.acquisition_thread.request_stop)
        self.control_panel.snapshot_requested.connect(self._save_snapshot)
        self.control_panel.colormap_changed.connect(self._on_colormap_changed)
        self.control_panel.intensity_scale_mode_changed.connect(self._on_scale_mode_changed)
        self.control_panel.scale_colormap_changed.connect(self._on_scale_colormap_changed)
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

        self.image_view.set_colormap(self.control_panel.colormap_selector.currentText())

        self._fps_label = QtWidgets.QLabel("FPS: -")
        self._fps_label.setStyleSheet(f"font-weight: 700; color: #52b788; font-size: {scale_font_size(13)}px; padding: 0 12px;")
        self.statusBar().addPermanentWidget(self._fps_label)
        self.statusBar().showMessage("Starting DCx beamprofiler...")
        self.acquisition_thread.start()

    def _apply_styles(self) -> None:
        panel_title_size = scale_font_size(18)
        section_title_size = scale_font_size(13)
        
        self.setStyleSheet(
            """
            QMainWindow {{
                background: #0b0d10;
            }}
            QFrame#controlPanel, QFrame#trendPanel {{
                background: #171a1f;
                border: 1px solid #282d35;
                border-radius: 12px;
            }}
            QLabel {{
                color: #d8dee9;
            }}
            QLabel#panelTitle {{
                font-size: {panel_title}px;
                font-weight: 700;
                color: #ffffff;
            }}
            QLabel#sectionTitle {{
                font-size: {section_title}px;
                font-weight: 600;
                color: #c7d0db;
            }}
            QGroupBox {{
                color: #e5e9f0;
                border: 1px solid #2c313a;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }}
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background: #0f1216;
                color: #eef2f6;
                border: 1px solid #313742;
                border-radius: 6px;
                min-height: 28px;
                padding: 2px 8px;
            }}
            QComboBox QAbstractItemView {{
                background: #0f1216;
                color: #eef2f6;
                border: 1px solid #313742;
                selection-background-color: #355070;
                selection-color: #ffffff;
                outline: 0;
            }}
            QPushButton {{
                background: #233142;
                color: #eef2f6;
                border: 1px solid #355070;
                border-radius: 7px;
                min-height: 30px;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background: #2d4157;
            }}
            QCheckBox {{
                color: #d8dee9;
            }}
            QRadioButton {{
                color: #ffffff;
            }}
            QSlider::groove:horizontal {{
                border-radius: 4px;
                height: 8px;
                background: #232831;
            }}
            QSlider::handle:horizontal {{
                background: #ffb347;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            """.format(panel_title=panel_title_size, section_title=section_title_size)
        )

    def _push_history(self, values: list[float], value: float) -> None:
        values.append(value)
        if len(values) > self.history_length:
            del values[0]

    def _compute_display_image(self, gray_image: np.ndarray) -> tuple[np.ndarray, tuple[float, float], str]:
        source = gray_image if gray_image.dtype == np.float32 else gray_image.astype(np.float32)
        threshold = max(1.0, float(self._saturation_level))

        if self._auto_scale_enabled:
            min_value = float(source.min())
            max_value = min(float(source.max()), threshold)
            self._fixed_scale_limits = (min_value, max_value)
        else:
            if self._fixed_scale_limits is None:
                self._fixed_scale_limits = (float(source.min()), threshold)
            min_value, max_value = self._fixed_scale_limits

        max_value = min(max_value, threshold)

        if max_value <= min_value:
            display_image = np.zeros_like(source, dtype=np.float32)
            return display_image, (0.0, 1.0), "Counts"

        clipped = np.clip(source, min_value, max_value)

        if self._scale_mode == "Logarithmic":
            display_image = np.log10(clipped + 1.0).astype(np.float32)
            levels = (float(np.log10(min_value + 1.0)), float(np.log10(max_value + 1.0)))
            label = "log10(counts + 1)"
        else:
            display_image = clipped.astype(np.float32)
            levels = (min_value, max_value)
            label = "Counts"

        return display_image, levels, label

    def _refresh_last_frame_display(self) -> None:
        if self._last_frame is None:
            return
        display_image, levels, label = self._compute_display_image(self._last_frame.gray_image)
        display_frame = FrameData(
            gray_image=display_image,
            projection_x=self._last_frame.projection_x,
            projection_y=self._last_frame.projection_y,
            x_coordinates_um=self._last_frame.x_coordinates_um,
            y_coordinates_um=self._last_frame.y_coordinates_um,
            metrics=self._last_frame.metrics,
            camera_full_scale_counts=self._last_frame.camera_full_scale_counts,
        )
        self._last_display_image = display_frame.gray_image
        self._last_display_levels = levels
        self._last_colorbar_label = label
        self.image_view.set_image(display_frame)
        self.image_view.set_color_levels(levels[0], levels[1], label)

    @QtCore.Slot(object)
    def _handle_camera_state(self, state: CameraState) -> None:
        self.control_panel.update_camera_state(state)
        self._show_status(f"Connected to {state.model_name} ({state.serial_number})")

    @QtCore.Slot(object)
    def _enqueue_frame(self, frame: FrameData) -> None:
        """Store latest frame and schedule processing; drops intermediate frames."""
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

        self.control_panel.set_camera_full_scale(frame.camera_full_scale_counts)
        self._saturation_level = self.control_panel.pixel_saturation.value()

        display_image, levels, label = self._compute_display_image(frame.gray_image)
        display_frame = FrameData(
            gray_image=display_image,
            projection_x=frame.projection_x,
            projection_y=frame.projection_y,
            x_coordinates_um=frame.x_coordinates_um,
            y_coordinates_um=frame.y_coordinates_um,
            metrics=frame.metrics,
            camera_full_scale_counts=frame.camera_full_scale_counts,
        )
        self._last_display_image = display_frame.gray_image
        self._last_display_levels = levels
        self._last_colorbar_label = label
        self.image_view.set_image(display_frame)
        self.image_view.set_color_levels(levels[0], levels[1], label)

        # Compute metrics on ROI sub-image (cache for reuse in projections)
        sub_image, sub_x, sub_y = self._get_roi_region(frame)
        self._cached_roi = (sub_image, sub_x, sub_y)
        roi_metrics, roi_proj_x, roi_proj_y = self._compute_roi_metrics(sub_image, sub_x, sub_y)
        self._cached_roi_projections = (roi_proj_x, roi_proj_y)

        # In cursor/peak mode, recompute metrics from the 1D slice profile
        if self._slice_mode in ("cursor", "peak"):
            roi_metrics = self._slice_metrics(sub_image, sub_x, sub_y, roi_metrics)

        sum_intensity_total = float(frame.gray_image.sum())

        self.image_view.horizontal_line.setValue(roi_metrics.centroid_y_um)
        self.image_view.vertical_line.setValue(roi_metrics.centroid_x_um)

        self._update_projections_cached(sub_image, sub_x, sub_y, roi_proj_x, roi_proj_y)
        self.control_panel.update_metrics(roi_metrics, sum_intensity_total)
        self.control_panel.update_saturation_headroom(
            roi_metrics.peak_value,
            frame.camera_full_scale_counts,
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
        self._apply_zoom_to_roi_view()

    def _handle_frame_simple(self, frame: FrameData) -> None:
        """Fast path: display image and projections only — no metrics, trends, or auto-exposure."""
        self._saturation_level = self.control_panel.pixel_saturation.value()
        display_image, levels, label = self._compute_display_image(frame.gray_image)
        display_frame = FrameData(
            gray_image=display_image,
            projection_x=frame.projection_x,
            projection_y=frame.projection_y,
            x_coordinates_um=frame.x_coordinates_um,
            y_coordinates_um=frame.y_coordinates_um,
            metrics=frame.metrics,
            camera_full_scale_counts=frame.camera_full_scale_counts,
        )
        self._last_display_image = display_frame.gray_image
        self._last_display_levels = levels
        self._last_colorbar_label = label
        self.image_view.set_image(display_frame)
        self.image_view.set_color_levels(levels[0], levels[1], label)

        # Show projections according to current slice mode
        image = frame.gray_image
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

        # Auto-exposure uses peak of full image
        peak_val = float(image.max())
        self._auto_adjust_exposure(peak_val, frame.camera_full_scale_counts)

        # Update saturation bar
        self.control_panel.update_saturation_headroom(peak_val, frame.camera_full_scale_counts)

        self._update_fps()
        self._apply_zoom_to_roi_view()

    @QtCore.Slot(str)
    def _on_colormap_changed(self, colormap_name: str) -> None:
        self.image_view.set_colormap(colormap_name)

    def _update_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        # Keep only timestamps from the last 2 seconds
        cutoff = now - 2.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / dt if dt > 0 else 0.0
            self._fps_label.setText(f"FPS: {fps:.1f}")
        else:
            self._fps_label.setText("FPS: -")

    @QtCore.Slot(str)
    def _on_scale_mode_changed(self, mode: str) -> None:
        self._scale_mode = mode
        self._refresh_last_frame_display()

    @QtCore.Slot(bool)
    def _on_scale_colormap_changed(self, enabled: bool) -> None:
        self._auto_scale_enabled = enabled
        if enabled:
            self._fixed_scale_limits = None
        self._refresh_last_frame_display()

    @QtCore.Slot(int)
    def _on_saturation_changed(self, value: int) -> None:
        self._saturation_level = value
        self._fixed_scale_limits = None
        # Don't re-display immediately; the next frame tick will pick up the new level.

    @QtCore.Slot()
    def _reset_trends(self) -> None:
        self.position_x_history.clear()
        self.position_y_history.clear()
        self.fwhm_x_history.clear()
        self.fwhm_y_history.clear()
        self.sum_intensity_history.clear()
        self.position_x_plot.set_series([])
        self.position_y_plot.set_series([])
        self.fwhm_x_plot.set_series([])
        self.fwhm_y_plot.set_series([])
        self.sum_intensity_plot.set_series([])

    @QtCore.Slot(int)
    def _on_trend_length_changed(self, value: int) -> None:
        self.history_length = max(10, value)
        # Trim existing histories
        for h in (
            self.position_x_history,
            self.position_y_history,
            self.fwhm_x_history,
            self.fwhm_y_history,
            self.sum_intensity_history,
        ):
            while len(h) > self.history_length:
                del h[0]

    @QtCore.Slot(bool)
    def _on_auto_exposure_toggled(self, enabled: bool) -> None:
        self._auto_exposure = enabled

    def _auto_adjust_exposure(self, peak_counts: float, full_scale_counts: float) -> None:
        """Adjust exposure so peak stays near 70 % of saturation level.

        Uses a damped proportional controller: the ideal correction factor
        is ``target / actual`` but we apply only 30 % of the log-space
        correction per frame to avoid overshoot and oscillation.  A dead
        band of ±10 % around the target prevents constant micro-adjustments.
        """
        if not self._auto_exposure or full_scale_counts <= 0:
            return
        threshold = float(self._saturation_level)
        if threshold <= 0 or peak_counts <= 0:
            return

        ratio = peak_counts / threshold  # current fraction of saturation
        target = 0.70  # aim for 70 %

        # Dead band: do nothing if within ±10 % of target
        if 0.60 <= ratio <= 0.80:
            return

        current_ms = self.control_panel.exposure_time.value()
        min_ms = max(0.01, self.control_panel.exposure_time.minimum())
        max_ms = 100.0

        # Ideal new exposure = current * (target / ratio).
        # Apply only 30 % of the correction in log-space (damping).
        ideal_factor = target / ratio
        damped_factor = ideal_factor ** 0.30  # cube-root-ish damping
        new_ms = current_ms * damped_factor

        new_ms = max(min_ms, min(max_ms, new_ms))
        new_ms = round(new_ms, 1)
        if abs(new_ms - current_ms) < 0.05:
            return  # Below spinbox resolution

        self.control_panel.exposure_time.blockSignals(True)
        self.control_panel.exposure_time.setValue(new_ms)
        self.control_panel.exposure_time.blockSignals(False)
        self.acquisition_thread.set_exposure_ms(new_ms)

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
            sub_image, sub_x, sub_y = self._get_roi_region(self._last_frame)
            self._cached_roi = (sub_image, sub_x, sub_y)
            roi_metrics, roi_proj_x, roi_proj_y = self._compute_roi_metrics(sub_image, sub_x, sub_y)
            self._cached_roi_projections = (roi_proj_x, roi_proj_y)
            self._update_projections_cached(sub_image, sub_x, sub_y, roi_proj_x, roi_proj_y)
            self.image_view.horizontal_line.setValue(roi_metrics.centroid_y_um)
            self.image_view.vertical_line.setValue(roi_metrics.centroid_x_um)
            self.control_panel.update_metrics(roi_metrics, float(self._last_frame.gray_image.sum()))
        self._apply_zoom_to_roi_view()

    @QtCore.Slot(bool)
    def _on_zoom_to_roi_toggled(self, enabled: bool) -> None:
        self._zoom_to_roi = enabled
        self._apply_zoom_to_roi_view()

    def _apply_zoom_to_roi_view(self) -> None:
        if self._zoom_to_roi and self.image_view.roi_rect.isVisible():
            pos = self.image_view.roi_rect.pos()
            size = self.image_view.roi_rect.size()
            roi_x0 = float(pos.x())
            roi_y0 = float(pos.y())
            roi_x1 = roi_x0 + float(size.x())
            roi_y1 = roi_y0 + float(size.y())
            self.image_view.setXRange(roi_x0, roi_x1, padding=0.02)
            self.image_view.setYRange(roi_y0, roi_y1, padding=0.02)
            # Also zoom projection plots to ROI range
            self.x_projection_plot.setXRange(roi_x0, roi_x1, padding=0.02)
            self.y_projection_plot.setYRange(roi_y0, roi_y1, padding=0.02)
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
        if self._last_frame is not None:
            if mode == "cursor":
                self.image_view.slice_h_line.setValue(self._last_frame.metrics.centroid_y_um)
                self.image_view.slice_v_line.setValue(self._last_frame.metrics.centroid_x_um)
            if self._cached_roi is not None:
                sub_image, sub_x, sub_y = self._cached_roi
                proj_x = self._cached_roi_projections[0] if self._cached_roi_projections else sub_image.sum(axis=0).astype(np.float64)
                proj_y = self._cached_roi_projections[1] if self._cached_roi_projections else sub_image.sum(axis=1).astype(np.float64)
                self._update_projections_cached(sub_image, sub_x, sub_y, proj_x, proj_y)

    @QtCore.Slot()
    def _on_cursor_moved(self) -> None:
        if self._last_frame is not None and self._slice_mode == "cursor":
            if not self._cursor_timer.isActive():
                self._cursor_timer.start()

    @QtCore.Slot()
    def _on_cursor_timer(self) -> None:
        """Throttled cursor update — runs at most ~33 times/sec."""
        if self._cached_roi is not None and self._slice_mode == "cursor":
            sub_image, sub_x, sub_y = self._cached_roi
            proj_x_sum = self._cached_roi_projections[0] if self._cached_roi_projections else sub_image.sum(axis=0).astype(np.float64)
            proj_y_sum = self._cached_roi_projections[1] if self._cached_roi_projections else sub_image.sum(axis=1).astype(np.float64)
            self._update_projections_cached(sub_image, sub_x, sub_y, proj_x_sum, proj_y_sum)

    def _get_roi_region(self, frame: FrameData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract sub-image and coordinate arrays for the current ROI."""
        image = frame.gray_image
        x_um = frame.x_coordinates_um
        y_um = frame.y_coordinates_um
        if not self.image_view.roi_rect.isVisible():
            return image, x_um, y_um
        pos = self.image_view.roi_rect.pos()
        size = self.image_view.roi_rect.size()
        roi_x0, roi_y0 = float(pos.x()), float(pos.y())
        roi_x1, roi_y1 = roi_x0 + float(size.x()), roi_y0 + float(size.y())
        col0 = int(np.clip(np.searchsorted(x_um, roi_x0), 0, image.shape[1] - 1))
        col1 = int(np.clip(np.searchsorted(x_um, roi_x1), 0, image.shape[1]))
        row0 = int(np.clip(np.searchsorted(y_um, roi_y0), 0, image.shape[0] - 1))
        row1 = int(np.clip(np.searchsorted(y_um, roi_y1), 0, image.shape[0]))
        col1 = max(col1, col0 + 1)
        row1 = max(row1, row0 + 1)
        return image[row0:row1, col0:col1], x_um[col0:col1], y_um[row0:row1]

    def _compute_roi_metrics(self, sub_image: np.ndarray, sub_x: np.ndarray, sub_y: np.ndarray) -> tuple[BeamMetrics, np.ndarray, np.ndarray]:
        """Compute beam metrics on the given sub-image region. Returns (metrics, proj_x, proj_y)."""
        proj_x = sub_image.sum(axis=0).astype(np.float64)
        proj_y = sub_image.sum(axis=1).astype(np.float64)
        sum_x = proj_x.sum()
        sum_y = proj_y.sum()
        cx = float((sub_x * proj_x).sum() / sum_x) if sum_x > 0 else 0.0
        cy = float((sub_y * proj_y).sum() / sum_y) if sum_y > 0 else 0.0
        return BeamMetrics(
            centroid_x_um=cx,
            centroid_y_um=cy,
            fwhm_x_um=compute_fwhm_1d(proj_x) * PIXEL_SIZE_UM,
            fwhm_y_um=compute_fwhm_1d(proj_y) * PIXEL_SIZE_UM,
            peak_value=float(sub_image.max()),
            sum_intensity=float(sub_image.sum()),
        ), proj_x, proj_y

    def _slice_metrics(self, sub_image: np.ndarray, sub_x: np.ndarray, sub_y: np.ndarray,
                       base_metrics: BeamMetrics) -> BeamMetrics:
        """Recompute centroid and FWHM from the 1D slice used in cursor/peak mode."""
        if self._slice_mode == "peak":
            peak_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)
            prof_x = sub_image[peak_idx[0], :].astype(np.float64)
            prof_y = sub_image[:, peak_idx[1]].astype(np.float64)
        else:  # cursor
            cursor_y_um = self.image_view.slice_h_line.value()
            cursor_x_um = self.image_view.slice_v_line.value()
            row_idx = int(np.clip(np.argmin(np.abs(sub_y - cursor_y_um)), 0, sub_image.shape[0] - 1))
            col_idx = int(np.clip(np.argmin(np.abs(sub_x - cursor_x_um)), 0, sub_image.shape[1] - 1))
            prof_x = sub_image[row_idx, :].astype(np.float64)
            prof_y = sub_image[:, col_idx].astype(np.float64)

        fwhm_x = compute_fwhm_1d(prof_x) * PIXEL_SIZE_UM
        fwhm_y = compute_fwhm_1d(prof_y) * PIXEL_SIZE_UM

        # Centroid from 1D slice
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

    def _update_projections_cached(self, sub_image: np.ndarray, sub_x_um: np.ndarray, sub_y_um: np.ndarray,
                                    roi_proj_x: np.ndarray, roi_proj_y: np.ndarray) -> None:
        """Update projection plots using pre-computed ROI data."""
        if self._slice_mode == "sum":
            proj_x = roi_proj_x
            proj_y = roi_proj_y
        elif self._slice_mode == "peak":
            peak_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)
            proj_x = sub_image[peak_idx[0], :].astype(np.float64)
            proj_y = sub_image[:, peak_idx[1]].astype(np.float64)
            self.image_view.horizontal_line.setValue(sub_y_um[peak_idx[0]])
            self.image_view.vertical_line.setValue(sub_x_um[peak_idx[1]])
        else:  # cursor
            cursor_y_um = self.image_view.slice_h_line.value()
            cursor_x_um = self.image_view.slice_v_line.value()
            row_idx = int(np.clip(
                np.argmin(np.abs(sub_y_um - cursor_y_um)), 0, sub_image.shape[0] - 1
            ))
            col_idx = int(np.clip(
                np.argmin(np.abs(sub_x_um - cursor_x_um)), 0, sub_image.shape[1] - 1
            ))
            proj_x = sub_image[row_idx, :].astype(np.float64)
            proj_y = sub_image[:, col_idx].astype(np.float64)

        self.x_projection_plot.set_projection(sub_x_um, proj_x)
        self.y_projection_plot.set_projection(sub_y_um, proj_y)

    @QtCore.Slot(str)
    def _show_status(self, message: str) -> None:
        self.control_panel.set_backend_message(message)
        self.statusBar().showMessage(message, 5000)

    @QtCore.Slot(str)
    def _show_error(self, message: str) -> None:
        self.control_panel.set_backend_message(message)
        self.statusBar().showMessage(message, 8000)
        print(message)

    def _save_snapshot(self) -> None:
        if self._last_frame is None:
            self._show_status("No frame available for snapshot")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"{timestamp}_beamprofile.png"
        default_path = Path.cwd() / default_name
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Beam Snapshot",
            str(default_path),
            "PNG Image (*.png);;ASCII Text (*.txt)",
        )
        if not file_path:
            return

        # --- ASCII export ---
        if selected_filter == "ASCII Text (*.txt)" or file_path.lower().endswith(".txt"):
            self._save_ascii(file_path)
            return

        # --- PNG + H5 + analysis figure export ---

        if self._last_display_image is None:
            self._show_status("No display frame available for snapshot")
            return

        if self._last_display_levels is None:
            self._show_status("No display level information available")
            return

        levels_min, levels_max = self._last_display_levels
        level_span = max(1e-9, levels_max - levels_min)
        normalized = np.clip((self._last_display_image - levels_min) / level_span, 0.0, 1.0)
        rgba = get_colormap(self.control_panel.colormap_selector.currentText()).map(
            normalized,
            mode="byte",
        )
        rgba = np.ascontiguousarray(rgba)

        height, width, _ = rgba.shape
        bytes_per_line = 4 * width
        qimage = QtGui.QImage(
            rgba.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGBA8888,
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
                dataset = xr.Dataset(
                    data_vars={
                        "counts": (
                            ("y_um", "x_um"),
                            self._last_frame.gray_image.astype(np.float32),
                        ),
                        "projection_x_counts": (
                            ("x_um",),
                            self._last_frame.projection_x.astype(np.float32),
                        ),
                        "projection_y_counts": (
                            ("y_um",),
                            self._last_frame.projection_y.astype(np.float32),
                        ),
                    },
                    coords={
                        "x_um": self._last_frame.x_coordinates_um.astype(np.float64),
                        "y_um": self._last_frame.y_coordinates_um.astype(np.float64),
                    },
                    attrs={
                        "units": "counts",
                        "axis_unit": "um",
                        "pixel_size_um": PIXEL_SIZE_UM,
                        "model": self.control_panel.model_label.text(),
                        "serial": self.control_panel.serial_label.text(),
                        "intensity_scale_mode": self._scale_mode,
                        "colormap": self.control_panel.colormap_selector.currentText(),
                    },
                )
                dataset.to_netcdf(h5_path, engine="h5netcdf")
                saved.append(h5_path)
            except Exception as exc:
                self._show_error(f"H5 export failed: {exc}")

        self._show_status(f"Saved: {', '.join(saved)}")

    def _save_analysis_figure(self, fig_path: str) -> None:
        """Generate a publication-style analysis figure with beam image and projections."""
        frame = self._last_frame
        if frame is None:
            return

        image = frame.gray_image.astype(np.float64)
        x_um = frame.x_coordinates_um
        y_um = frame.y_coordinates_um
        metrics = frame.metrics

        # Sum projections (integral)
        proj_x_sum = image.sum(axis=0)
        proj_y_sum = image.sum(axis=1)

        # Slice at peak
        peak_idx = np.unravel_index(np.argmax(image), image.shape)
        slice_x = image[peak_idx[0], :].astype(np.float64)
        slice_y = image[:, peak_idx[1]].astype(np.float64)

        # Normalise for plotting
        def _norm(arr: np.ndarray) -> np.ndarray:
            mx = arr.max()
            return arr / mx if mx > 0 else arr

        proj_x_n = _norm(proj_x_sum)
        proj_y_n = _norm(proj_y_sum)
        slice_x_n = _norm(slice_x)
        slice_y_n = _norm(slice_y)

        # Gaussian fit helper
        def _gauss(x: np.ndarray, a: float, mu: float, sigma: float, bg: float) -> np.ndarray:
            return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg

        def _fit_gauss(axis: np.ndarray, profile: np.ndarray) -> tuple[np.ndarray, float] | None:
            try:
                peak_val = profile.max()
                bg = profile.min()
                center = float(axis[np.argmax(profile)])
                sigma0 = float(np.abs(axis[-1] - axis[0])) / 10.0
                popt, _ = curve_fit(
                    _gauss, axis, profile,
                    p0=[peak_val - bg, center, sigma0, bg],
                    maxfev=5000,
                )
                fitted = _gauss(axis, *popt)
                fwhm = abs(popt[2]) * 2.3548  # 2*sqrt(2*ln2)
                return fitted, fwhm
            except Exception:
                return None

        fit_x_sum = _fit_gauss(x_um, proj_x_n)
        fit_y_sum = _fit_gauss(y_um, proj_y_n)
        fit_x_slice = _fit_gauss(x_um, slice_x_n)
        fit_y_slice = _fit_gauss(y_um, slice_y_n)

        # Colormap name for matplotlib
        cmap_name = self.control_panel.colormap_selector.currentText().lower()
        try:
            matplotlib.colormaps[cmap_name]
        except (KeyError, AttributeError):
            cmap_name = "inferno"

        # --- Build figure with manual axes placement for tight, clean layout ---
        fig = plt.figure(figsize=(10, 8))

        # Layout constants (figure-fraction coordinates)
        left = 0.08      # left edge of image / h-proj
        img_right = 0.72  # right edge of image
        img_bottom = 0.10  # bottom edge of image
        img_top = 0.72    # top edge of image
        proj_h = 0.22     # height of h-projection panel
        proj_w = 0.22     # width of v-projection panel
        gap = 0.005       # tiny gap between image and projections

        # Axes positions: [left, bottom, width, height]
        ax_img = fig.add_axes([left, img_bottom, img_right - left, img_top - img_bottom])
        ax_hproj = fig.add_axes([left, img_top + gap, img_right - left, proj_h],
                                sharex=ax_img)
        ax_vproj = fig.add_axes([img_right + gap, img_bottom, proj_w, img_top - img_bottom],
                                sharey=ax_img)

        # --- 2D image (log scale) ---
        img_plot = image.copy()
        img_plot[img_plot < 1] = 1
        extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
        im = ax_img.imshow(
            img_plot, origin="lower", aspect="auto",
            extent=extent, cmap=cmap_name,
            norm=LogNorm(vmin=max(1, img_plot.min()), vmax=img_plot.max()),
        )
        ax_img.set_xlabel("x (µm)")
        ax_img.set_ylabel("y (µm)")

        # Colorbar as inset at bottom of image
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_img, width="60%", height="3%", loc="lower center",
                         borderpad=1.5)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7, direction="in", color="white")
        cbar.outline.set_edgecolor("white")
        cbar.ax.xaxis.set_tick_params(color="white")
        for label in cbar.ax.get_xticklabels():
            label.set_color("white")
            label.set_fontsize(7)

        # --- Horizontal projection (top) ---
        ax_hproj.plot(x_um, proj_x_n, "k-", linewidth=0.8, label="Integral")
        ax_hproj.plot(x_um, slice_x_n, "r-", linewidth=0.8, label="Slice")
        if fit_x_sum is not None:
            ax_hproj.plot(
                x_um,
                fit_x_sum[0],
                color="#7c8da3",
                linewidth=1.2,
                linestyle="--",
                label=f"Integral fit FWHM {fit_x_sum[1]:.0f} µm",
            )
        if fit_x_slice is not None:
            ax_hproj.plot(
                x_um,
                fit_x_slice[0],
                color="#e8a020",
                linewidth=1.2,
                linestyle="--",
                label=f"Slice fit FWHM {fit_x_slice[1]:.0f} µm",
            )
        ax_hproj.set_ylabel("I (a.u.)")
        ax_hproj.set_ylim(-0.05, 1.12)
        ax_hproj.legend(loc="upper right", fontsize=8, framealpha=0.8)

        # --- Vertical projection (right) ---
        ax_vproj.plot(proj_y_n, y_um, "k-", linewidth=0.8, label="Integral")
        ax_vproj.plot(slice_y_n, y_um, "r-", linewidth=0.8, label="Slice")
        if fit_y_sum is not None:
            ax_vproj.plot(
                fit_y_sum[0],
                y_um,
                color="#7c8da3",
                linewidth=1.2,
                linestyle="--",
                label=f"Integral fit FWHM {fit_y_sum[1]:.0f} µm",
            )
        if fit_y_slice is not None:
            ax_vproj.plot(
                fit_y_slice[0],
                y_um,
                color="#e8a020",
                linewidth=1.2,
                linestyle="--",
                label=f"Slice fit FWHM {fit_y_slice[1]:.0f} µm",
            )
        ax_vproj.set_xlabel("I (a.u.)")
        ax_vproj.set_xlim(-0.05, 1.12)
        ax_vproj.legend(loc="lower left", fontsize=8, framealpha=0.8)

        # --- Tick configuration ---
        # Image: ticks in, labels bottom+left
        ax_img.tick_params(axis="both", direction="in", which="both",
                           top=True, right=True)
        ax_img.minorticks_on()
        ax_img.tick_params(axis="both", which="minor", direction="in",
                           top=True, right=True)

        # Horizontal projection: hide shared-x labels, keep top x + left y
        plt.setp(ax_hproj.get_xticklabels(), visible=False)
        ax_hproj.tick_params(axis="both", direction="in", which="both",
                             bottom=True, top=True, left=True, right=True,
                             labelbottom=False, labeltop=False, labelleft=True)
        ax_hproj.minorticks_on()
        ax_hproj.tick_params(axis="both", which="minor", direction="in",
                             bottom=True, top=True, left=True, right=True)
        # Top twin axis with x tick labels
        ax_hproj_top = ax_hproj.twiny()
        ax_hproj_top.set_xlim(ax_hproj.get_xlim())
        ax_hproj_top.set_xlabel("x (µm)")
        ax_hproj_top.tick_params(axis="x", direction="in", which="both",
                                 labeltop=True)
        ax_hproj_top.minorticks_on()
        ax_hproj_top.tick_params(axis="x", which="minor", direction="in")

        # Vertical projection: hide shared-y labels, keep right y + bottom x
        plt.setp(ax_vproj.get_yticklabels(), visible=False)
        ax_vproj.tick_params(axis="both", direction="in", which="both",
                             top=True, bottom=True, left=True, right=True,
                             labelleft=False, labelright=False, labelbottom=True)
        ax_vproj.minorticks_on()
        ax_vproj.tick_params(axis="both", which="minor", direction="in",
                             top=True, bottom=True, left=True, right=True)
        # Right twin axis with y tick labels
        ax_vproj_right = ax_vproj.twinx()
        ax_vproj_right.set_ylim(ax_vproj.get_ylim())
        ax_vproj_right.set_ylabel("y (µm)")
        ax_vproj_right.tick_params(axis="y", direction="in", which="both",
                                   labelright=True)
        ax_vproj_right.minorticks_on()
        ax_vproj_right.tick_params(axis="y", which="minor", direction="in")

        fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _save_ascii(self, file_path: str) -> None:
        """Save the raw image as ASCII with 0-255 scale and axis headers."""
        frame = self._last_frame
        if frame is None:
            self._show_status("No frame available")
            return

        image = frame.gray_image.astype(np.float64)
        img_max = image.max()
        if img_max > 0:
            scaled = image * (255.0 / img_max)
        else:
            scaled = image

        x_um = frame.x_coordinates_um
        y_um = frame.y_coordinates_um

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Beam snapshot — pixel values scaled to 0..255\n")
                f.write(f"# Pixel size: {PIXEL_SIZE_UM} µm\n")
                f.write(f"# Image shape: {image.shape[0]} rows x {image.shape[1]} cols\n")
                f.write(f"# Original peak counts: {img_max:.1f}\n")
                f.write(f"# First row is X axis in µm, first column is Y axis in µm\n")
                # Header row: empty corner + x coordinates
                f.write("Y\\X_um")
                for x in x_um:
                    f.write(f"\t{x:.1f}")
                f.write("\n")
                # Data rows: y coordinate + pixel values
                for i, y in enumerate(y_um):
                    f.write(f"{y:.1f}")
                    for j in range(scaled.shape[1]):
                        f.write(f"\t{scaled[i, j]:.1f}")
                    f.write("\n")
            self._show_status(f"Saved ASCII: {file_path}")
        except Exception as exc:
            self._show_error(f"ASCII export failed: {exc}")

    @QtCore.Slot()
    def _add_crosshair_marker(self) -> None:
        """Place a draggable crosshair marker at the current centroid position."""
        if self._last_frame is None:
            return
        cx = self._last_frame.metrics.centroid_x_um
        cy = self._last_frame.metrics.centroid_y_um
        marker = pg.TargetItem(
            pos=(cx, cy),
            size=14,
            movable=True,
            pen=pg.mkPen("#ffffff", width=1.5),
            hoverPen=pg.mkPen("#ffff00", width=2),
            symbol="crosshair",
        )
        marker.setZValue(100)
        self.image_view.addItem(marker)
        self.image_view._crosshair_markers.append(marker)

    @QtCore.Slot()
    def _clear_crosshair_markers(self) -> None:
        """Remove all crosshair markers from the beam image."""
        for marker in self.image_view._crosshair_markers:
            self.image_view.removeItem(marker)
        self.image_view._crosshair_markers.clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.acquisition_thread.stop_thread()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    cameras = DcxCamera.list_cameras() if DCX_BACKEND_AVAILABLE else []
    camera_id = 0
    if len(cameras) > 1:
        items = [f"{serial}  (Camera {cam_id})" for cam_id, serial in cameras]
        choice, ok = QtWidgets.QInputDialog.getItem(
            None, "Select Camera",
            f"{len(cameras)} cameras detected. Choose one:",
            items, 0, False,
        )
        if not ok:
            return 0
        camera_id = cameras[items.index(choice)][0]
    elif len(cameras) == 1:
        camera_id = cameras[0][0]
    window = BeamProfilerApp(camera_id=camera_id)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # pragma: no cover - top-level fallback for desktop app
        traceback.print_exc()
        raise
# Beam Profiler

A real-time laser beam profiler for **Thorlabs DCC1545M-GL** cameras, built with PySide6 and pyqtgraph.
When no hardware is connected, it falls back to a synthetic Gaussian beam for testing.

## Features

- Live 2D beam image with selectable colormaps (Gray, Viridis, Inferno, Plasma, Magma, Cividis, Turbo) and linear/log intensity scale
- Axes in µm using the camera sensor geometry
- Projection modes: sum projection, slice at cursor, slice at peak intensity (default)
- ROI selection with zoom-to-ROI (applies to main image and projections)
- Centroid and FWHM tracking with live trend plots
- Auto-exposure
- Fluence calculator (µJ/cm²) with selectable beam-area model: `FWHM/2` or `1/e` radii
- Draggable crosshair markers
- Simple mode (default) for maximum frame rate; full mode adds trends and advanced metrics
- Multi-camera support — when multiple cameras are connected, a dialog lets you choose by serial number; run the script twice to use two cameras simultaneously
- Export snapshots as PNG, HDF5 (.h5), or ASCII (.txt)
- Analysis snapshot figure with Gaussian fits for integral and slice projections including FWHM values in the legend
- Synthetic camera fallback when no hardware is available

## Prerequisites

1. **Python 3.11+** (tested with 3.13)
2. **DCx / uc480 SDK** for the Thorlabs DCC1545M-GL.
   This repository includes the legacy SDK files under `dcx_camera_interfaces_2018_09/`.
   The driver DLL (`uc480_64.dll`) is loaded via `ctypes` — no additional pip package is required.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd beamprofiler-main

# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
python beamprofiler_qt_Thorcam.py
```

If multiple Thorlabs cameras are connected, a dialog will appear letting you select one by serial number.
To open a second camera, run the script again in a separate terminal and pick the other serial number.

If no camera hardware is detected, the application starts in synthetic mode with a simulated Gaussian beam.

## Files

| File | Description |
|------|-------------|
| `beamprofiler_qt_Thorcam.py` | Main application — Thorlabs DCx beam profiler with full Qt UI |
| `requirements.txt` | Python dependencies |
| `run_demo.bat` | Windows batch launcher |
| `dcx_camera_interfaces_2018_09/` | Legacy DCx / uc480 SDK (drivers, headers, firmware) |

## Fluence Modes

The fluence panel offers two selectable models:

- **Top-hat with FWHM/2 radii** (default)
- **Top-hat with 1/e radii**

The active formula is shown directly in the UI so the current calculation model is always visible.

## Snapshot Analysis

Using the **Snapshot** button stores:

- A rendered `.png` image
- A `.h5` file containing a Python-readable xarray dataset with `x_um` and `y_um` coordinates and image/projection data in counts
- An analysis figure (`.png`) with Gaussian fits for both integral and slice projections, including FWHM values in the legends
- An ASCII `.txt` export of the raw image data

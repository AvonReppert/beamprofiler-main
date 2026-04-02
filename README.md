# Beam Profiler

A real-time laser beam profiler built with PySide6, pyqtgraph, and vendor camera SDKs.
It supports both the IDS U3-3880LE-C-HQ and the Thorlabs DCC1545M-GL with the same Qt user interface.

## Features

- Live 2D beam image with selectable colormaps (Gray, Viridis, Inferno, …) and linear/log scale
- Axes in µm using the active camera geometry
- Projection modes: sum projection, slice at cursor, slice at peak intensity
- ROI selection for region-based analysis
- Centroid and FWHM tracking with trend plots
- Auto-exposure (1 ms – 100 ms)
- Fluence calculator (µJ/cm²) with selectable beam-area model: `FWHM/2` or `1/e` radii
- Draggable crosshair markers
- Simple mode (default) for maximum frame rate
- Export snapshots as PNG, HDF5 (.h5), or ASCII (.txt)
- Analysis snapshot figure with Gaussian fits for integral and slice projections including FWHM values in the legend
- Unified launcher to choose camera backend at startup
- Synthetic camera fallback when a hardware backend is unavailable

## Prerequisites

1. **Python 3.11+** (tested with 3.13)
2. **IDS Peak SDK** for the IDS backend — install from [ids-imaging.com](https://en.ids-imaging.com/download-peak.html).
  The installer provides the GenTL producer and the Python packages (`ids-peak`, `ids-peak-common`, `ids-peak-icv`).
3. **DCx / uc480 SDK** for the Thorlabs DCC1545M-GL backend.
  This repository expects the legacy SDK files under [dcx_camera_interfaces_2018_09](dcx_camera_interfaces_2018_09).
4. A working GenTL producer on the system path if you want to use the IDS backend.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The IDS Peak Python packages (`ids-peak`, `ids-peak-common`, `ids-peak-icv`) are
> distributed through the IDS Peak SDK installer, not PyPI. Make sure the SDK is installed
> before running `pip install`. If the packages are not found, point pip at the SDK's wheel
> directory, e.g. `pip install --find-links "C:\Program Files\IDS\ids_peak\sdk\python" ids-peak ids-peak-common ids-peak-icv`.
>
> The Thorlabs DCC1545M-GL backend uses the legacy `uc480_64.dll` through `ctypes`; no extra
> pip package is required for that backend as long as [dcx_camera_interfaces_2018_09](dcx_camera_interfaces_2018_09) is present.

## Running

```bash
python beamprofiler_qt_unified.py
```

You can also preselect the backend from the command line:

```bash
python beamprofiler_qt_unified.py --camera ids
python beamprofiler_qt_unified.py --camera thorlabs
```

Or use the provided batch file on Windows (sets IDS SDK environment variables automatically):

```cmd
run_demo.bat beamprofiler_qt_unified.py
```

If a selected camera backend is unavailable, the application falls back to a synthetic Gaussian beam for testing.

## Files

| File | Description |
|------|-------------|
| `beamprofiler_qt_unified.py` | Unified launcher with camera selection |
| `beamprofiler_qt_IDS-U3-3880LE.py` | IDS backend with the shared Qt UI |
| `beamprofiler_qt_Thorcam.py` | Thorlabs DCx backend with the shared Qt UI |
| `camera.py` | IDS camera abstraction (exposure, pixel format, acquisition) |
| `requirements.txt` | Python dependencies |
| `run_demo.bat` | Windows launcher that sets IDS SDK paths |
| `.gitignore` | Git ignore rules |
| `dcx_camera_interfaces_2018_09/` | Legacy DCx / uc480 SDK used for the Thorlabs camera |

## Fluence Modes

The fluence panel offers two selectable models:

- `Top-hat with FWHM/2 radii`
- `Top-hat with 1/e radii`

The active formula is shown directly in the UI so the current calculation model is always visible.

## Snapshot Analysis

Using the `Snapshot` button stores:

- A rendered `.png` image
- A `.h5` file containing a Python-readable xarray dataset with `x_um` and `y_um` coordinates and image/projection data in counts
- An analysis figure with Gaussian fits for both integral and slice projections, including FWHM values in the legends

## Notes

### General notes

* Kivy and OpenGL use a coordinate system that is inverted relative to
  the camera image.
  To account for this, the sample enables vertical image flipping using
  the ReverseY node.
* The flip state is normally restored when the application exits.
  However, if the program terminates unexpectedly, the state may not
  reset correctly.

  If this occurs, reload the default user settings in the camera software
  to restore normal behavior.
* Due to this coordinate inversion, clockwise rotations performed in
  this sample will appear as counter-clockwise rotations when the same
  pipeline settings are loaded in other applications.

### Notes on linux

The plyer file chooser used in this sample requires one of the following command line tools to be installed:
* zenity (GTK)
* yad (a zenity fork)
* kdialog (KDE)

If none of these utilities are available, a dialog will notify the user,
and loading or saving pipeline settings will not be possible.

> Note for Python 3.12 or later: If you’re using plyer 2.1.0 or earlier, you must install `setuptools`.

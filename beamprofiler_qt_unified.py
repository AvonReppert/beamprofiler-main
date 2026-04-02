from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets


@dataclass(frozen=True)
class CameraOption:
    key: str
    label: str
    script_name: str


BASE_DIR = Path(__file__).resolve().parent
CAMERA_OPTIONS = {
    "ids": CameraOption(
        key="ids",
        label="IDS U3-3880LE-C-HQ  (monochrome display)",
        script_name="beamprofiler_qt_IDS-U3-3880LE.py",
    ),
    "ids_color": CameraOption(
        key="ids_color",
        label="IDS U3-3880LE-C-HQ  (color display)",
        script_name="beamprofiler_qt_IDS_color.py",
    ),
    "thorlabs": CameraOption(
        key="thorlabs",
        label="Thorlabs DCC1545M-GL",
        script_name="beamprofiler_qt_Thorcam.py",
    ),
}


def load_backend_module(option: CameraOption) -> Any:
    module_path = BASE_DIR / option.script_name
    spec = importlib.util.spec_from_file_location(f"beamprofiler_backend_{option.key}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load backend module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "BeamProfilerApp"):
        raise RuntimeError(f"Backend script does not expose BeamProfilerApp: {module_path}")
    return module


class CameraChooserDialog(QtWidgets.QDialog):
    def __init__(self, initial_key: str = "thorlabs", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Camera Backend")
        self.setModal(True)
        self.resize(420, 150)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        label = QtWidgets.QLabel("Choose which camera should drive the beamprofiler UI:")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.combo = QtWidgets.QComboBox()
        for option in CAMERA_OPTIONS.values():
            self.combo.addItem(option.label, option.key)
        index = max(0, self.combo.findData(initial_key))
        self.combo.setCurrentIndex(index)
        layout.addWidget(self.combo)

        self.remember_checkbox = QtWidgets.QCheckBox("Start directly with this choice next time via --camera")
        self.remember_checkbox.setChecked(False)
        self.remember_checkbox.setEnabled(False)
        layout.addWidget(self.remember_checkbox)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_key(self) -> str:
        return str(self.combo.currentData())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Qt beamprofiler launcher")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_OPTIONS.keys()),
        help="Start directly with a given camera backend: ids or thorlabs",
    )
    return parser.parse_args()


def choose_camera_key(app: QtWidgets.QApplication, cli_key: str | None) -> str | None:
    if cli_key is not None:
        return cli_key

    dialog = CameraChooserDialog(parent=None)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog.selected_key


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    args = parse_args()

    selected_key = choose_camera_key(app, args.camera)
    if selected_key is None:
        return 0

    option = CAMERA_OPTIONS[selected_key]
    backend_module = load_backend_module(option)

    window = backend_module.BeamProfilerApp()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
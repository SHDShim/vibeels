from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "vibeels-mpl")

import matplotlib

matplotlib.use("Qt5Agg")

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets

from .processing import (
    EV_TO_CMINV,
    MapProcessingResult,
    StackProcessingResult,
    align_spectra_1d,
    load_signal,
    process_map_dataset,
    process_snapshot_stack,
)


@dataclass
class LoadedData:
    eels_signal: object
    image_signal: Optional[object]
    eels_path: str
    image_path: Optional[str]


class PlotCanvas(FigureCanvas):
    def __init__(self, subplot_spec=(1, 1), parent=None):
        self.figure = Figure(figsize=(7, 5), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        if subplot_spec == (1, 1):
            self.axes = [self.figure.add_subplot(111)]
        else:
            rows, cols = subplot_spec
            self.axes = [self.figure.add_subplot(rows, cols, index + 1) for index in range(rows * cols)]


class VibeelsWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("vibeels")
        self.resize(1440, 900)

        self.loaded: Optional[LoadedData] = None
        self.current_result: Optional[object] = None
        self.selector: Optional[object] = None
        self.selection_rect: Optional[Rectangle] = None
        self.selection_polygon: Optional[Polygon] = None
        self._image_view_limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._corrected_view_limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._syncing_corrected_xlim = False
        self.map_polygon_vertices: Optional[list[tuple[float, float]]] = None
        self._map_intensity_min = 0.0
        self._map_intensity_max = 1.0

        self._build_ui()
        self._apply_default_ranges()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        top_row = QtWidgets.QHBoxLayout()

        image_panel = QtWidgets.QVBoxLayout()
        self.image_canvas = PlotCanvas((1, 1), self)
        self.image_toolbar = NavigationToolbar(self.image_canvas, self)
        image_panel.addWidget(self.image_toolbar)
        image_panel.addWidget(self.image_canvas, 1)

        corrected_panel = QtWidgets.QVBoxLayout()
        self.corrected_canvas = PlotCanvas((1, 1), self)
        self.corrected_toolbar = NavigationToolbar(self.corrected_canvas, self)
        corrected_panel.addWidget(self.corrected_toolbar)
        corrected_panel.addWidget(self.corrected_canvas, 1)

        fit_panel = QtWidgets.QVBoxLayout()
        self.fit_canvas = PlotCanvas((1, 1), self)
        self.fit_toolbar = NavigationToolbar(self.fit_canvas, self)
        fit_panel.addWidget(self.fit_toolbar)
        fit_panel.addWidget(self.fit_canvas, 1)

        top_row.addLayout(image_panel, 1)
        top_row.addLayout(corrected_panel, 1)
        top_row.addLayout(fit_panel, 1)
        left_layout.addLayout(top_row, 1)

        self.spectrum_canvas = PlotCanvas((1, 1), self)
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
        left_layout.addWidget(self.spectrum_toolbar)
        left_layout.addWidget(self.spectrum_canvas, 1)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs, 1)
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(160)
        right_layout.addWidget(self.status_box)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)

        self._build_data_tab()
        self._build_roi_tab()
        self._build_calibration_tab()

    def _build_data_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["2D map (y, x, energy)", "Snapshot stack (frame, vertical, energy)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.mode_combo.setEnabled(False)

        self.eels_path_label = QtWidgets.QLabel("No EELS file loaded")
        self.eels_path_label.setWordWrap(True)
        self.image_path_label = QtWidgets.QLabel("Optional image not loaded")
        self.image_path_label.setWordWrap(True)

        load_eels = QtWidgets.QPushButton("Load EELS DM3/DM4")
        load_eels.clicked.connect(self._load_eels)
        load_image = QtWidgets.QPushButton("Load Reference Image")
        load_image.clicked.connect(self._load_image)
        refresh = QtWidgets.QPushButton("Process Current Settings")
        refresh.clicked.connect(self._process_current)

        self.shape_label = QtWidgets.QLabel("-")
        self.snapshot_index_spin = QtWidgets.QSpinBox()
        self.snapshot_index_spin.setMaximum(0)
        self.snapshot_index_spin.valueChanged.connect(self._on_snapshot_index_changed)
        self.snapshot_prev_button = QtWidgets.QPushButton("Previous Snapshot")
        self.snapshot_prev_button.clicked.connect(self._show_previous_snapshot)
        self.snapshot_next_button = QtWidgets.QPushButton("Next Snapshot")
        self.snapshot_next_button.clicked.connect(self._show_next_snapshot)

        layout.addRow("Mode", self.mode_combo)
        layout.addRow(load_eels)
        layout.addRow("EELS file", self.eels_path_label)
        layout.addRow(load_image)
        layout.addRow("Reference image", self.image_path_label)
        layout.addRow("Loaded shape", self.shape_label)
        layout.addRow("Snapshot index", self.snapshot_index_spin)
        layout.addRow(self.snapshot_prev_button)
        layout.addRow(self.snapshot_next_button)
        layout.addRow(refresh)

        self.tabs.addTab(tab, "Data")

    def _build_roi_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.energy_start_spin = QtWidgets.QSpinBox()
        self.energy_stop_spin = QtWidgets.QSpinBox()
        self.energy_start_spin.setMaximum(100000)
        self.energy_stop_spin.setMaximum(100000)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setMaximum(1e12)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(18000.0)
        self.threshold_spin.hide()

        self.roi_x0 = QtWidgets.QSpinBox()
        self.roi_x1 = QtWidgets.QSpinBox()
        self.roi_y0 = QtWidgets.QSpinBox()
        self.roi_y1 = QtWidgets.QSpinBox()
        for widget in [self.roi_x0, self.roi_x1, self.roi_y0, self.roi_y1]:
            widget.setMaximum(100000)
            widget.valueChanged.connect(self._update_selection_overlay)

        self.stack_y0 = QtWidgets.QSpinBox()
        self.stack_y1 = QtWidgets.QSpinBox()
        self.frame_start = QtWidgets.QSpinBox()
        self.frame_stop = QtWidgets.QSpinBox()
        for widget in [self.stack_y0, self.stack_y1, self.frame_start, self.frame_stop]:
            widget.setMaximum(100000)
        self.stack_y0.valueChanged.connect(self._update_selection_overlay)
        self.stack_y1.valueChanged.connect(self._update_selection_overlay)
        self.frame_start.valueChanged.connect(self._draw_initial_image)
        self.frame_stop.valueChanged.connect(self._draw_initial_image)

        self.enable_selector = QtWidgets.QCheckBox("Drag ROI / extraction band on image")
        self.enable_selector.setChecked(True)
        self.enable_selector.toggled.connect(self._sync_selector_state)

        self.map_mask_min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.map_mask_max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for slider in [self.map_mask_min_slider, self.map_mask_max_slider]:
            slider.setRange(0, 1000)
            slider.valueChanged.connect(self._on_map_mask_slider_changed)
        self.map_mask_min_label = QtWidgets.QLabel("Mask min: -")
        self.map_mask_max_label = QtWidgets.QLabel("Mask max: -")
        self.map_hist_canvas = PlotCanvas((1, 1), self)
        self.map_hist_canvas.setMinimumHeight(180)

        process_button = QtWidgets.QPushButton("Apply ROI / Alignment")
        process_button.clicked.connect(self._process_current)

        self.map_mode_widgets = [
            self.energy_start_spin,
            self.energy_stop_spin,
            self.roi_x0,
            self.roi_x1,
            self.roi_y0,
            self.roi_y1,
            self.map_mask_min_slider,
            self.map_mask_max_slider,
            self.map_hist_canvas,
            self.map_mask_min_label,
            self.map_mask_max_label,
        ]
        self.snapshot_mode_widgets = [
            self.stack_y0,
            self.stack_y1,
            self.frame_start,
            self.frame_stop,
        ]

        layout.addRow("Map energy start", self.energy_start_spin)
        layout.addRow("Map energy stop", self.energy_stop_spin)
        layout.addRow("ROI x start", self.roi_x0)
        layout.addRow("ROI x stop", self.roi_x1)
        layout.addRow("ROI y start", self.roi_y0)
        layout.addRow("ROI y stop", self.roi_y1)
        layout.addRow(self.enable_selector)
        layout.addRow(self.map_mask_min_label)
        layout.addRow(self.map_mask_min_slider)
        layout.addRow(self.map_mask_max_label)
        layout.addRow(self.map_mask_max_slider)
        layout.addRow(self.map_hist_canvas)
        layout.addRow("Detector y start", self.stack_y0)
        layout.addRow("Detector y stop", self.stack_y1)
        layout.addRow("Snapshot start", self.frame_start)
        layout.addRow("Snapshot stop", self.frame_stop)
        layout.addRow(process_button)

        self.tabs.addTab(tab, "ROI + Align")

    def _build_calibration_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.fit_start_spin = QtWidgets.QDoubleSpinBox()
        self.fit_stop_spin = QtWidgets.QDoubleSpinBox()
        self.guess_start_spin = QtWidgets.QDoubleSpinBox()
        self.guess_stop_spin = QtWidgets.QDoubleSpinBox()
        for widget, value in [
            (self.fit_start_spin, -0.05),
            (self.fit_stop_spin, 0.05),
            (self.guess_start_spin, -0.04),
            (self.guess_stop_spin, 0.04),
        ]:
            widget.setRange(-10.0, 10.0)
            widget.setDecimals(4)
            widget.setSingleStep(0.001)
            widget.setValue(value)

        self.zlp_center_label = QtWidgets.QLabel("-")
        self.pixel_count_label = QtWidgets.QLabel("-")
        self.axis_span_label = QtWidgets.QLabel("-")
        process_button = QtWidgets.QPushButton("Run Zero-Loss Calibration")
        process_button.clicked.connect(self._process_current)
        save_button = QtWidgets.QPushButton("Save Results")
        save_button.clicked.connect(self._save_results)

        layout.addRow("Fit start (eV)", self.fit_start_spin)
        layout.addRow("Fit stop (eV)", self.fit_stop_spin)
        layout.addRow("Guess start (eV)", self.guess_start_spin)
        layout.addRow("Guess stop (eV)", self.guess_stop_spin)
        layout.addRow("Zero-loss center", self.zlp_center_label)
        layout.addRow("Selected spectra", self.pixel_count_label)
        layout.addRow("Calibrated axis", self.axis_span_label)
        layout.addRow(process_button)
        layout.addRow(save_button)

        self.tabs.addTab(tab, "Calibration")

    def _apply_default_ranges(self):
        self.energy_start_spin.setValue(150)
        self.energy_stop_spin.setValue(550)
        self.stack_y0.setValue(220)
        self.stack_y1.setValue(320)
        self._update_snapshot_navigation_enabled()
        self._update_mode_specific_ui()

    def _log(self, message: str):
        self.status_box.appendPlainText(message)

    def _load_eels(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load EELS signal",
            "",
            "DigitalMicrograph (*.dm3 *.dm4);;All files (*)",
        )
        if not path:
            return
        try:
            signal = load_signal(path)
        except Exception as exc:
            self._log(f"Failed to load EELS file: {exc}")
            return

        image_signal = self.loaded.image_signal if self.loaded else None
        image_path = self.loaded.image_path if self.loaded else None
        self.loaded = LoadedData(signal, image_signal, path, image_path)
        self._image_view_limits = None
        self._corrected_view_limits = None
        self.map_polygon_vertices = None
        self.eels_path_label.setText(path)
        self.shape_label.setText(str(signal.data.shape))
        self._set_mode_from_signal(signal)
        self._sync_ranges_to_loaded_data()
        self._update_mode_specific_ui()
        if self.mode_combo.currentIndex() == 0:
            self._reset_map_histogram_controls()
        self._draw_initial_image()
        self._log(f"Loaded EELS signal: {path}")

    def _load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load reference image",
            "",
            "DigitalMicrograph (*.dm3 *.dm4);;All files (*)",
        )
        if not path:
            return
        try:
            image_signal = load_signal(path)
        except Exception as exc:
            self._log(f"Failed to load image file: {exc}")
            return

        if self.loaded is None:
            self.loaded = LoadedData(image_signal, image_signal, "", path)
        else:
            self.loaded.image_signal = image_signal
            self.loaded.image_path = path
        self.image_path_label.setText(path)
        self._draw_initial_image()
        self._log(f"Loaded reference image: {path}")

    def _sync_ranges_to_loaded_data(self):
        if self.loaded is None:
            return
        shape = self.loaded.eels_signal.data.shape
        if len(shape) != 3:
            return
        self.energy_start_spin.setMaximum(shape[-1] - 1)
        self.energy_stop_spin.setMaximum(shape[-1])
        self.energy_stop_spin.setValue(min(self.energy_stop_spin.value(), shape[-1]))

        if self.mode_combo.currentIndex() == 0:
            self.roi_x0.setMaximum(shape[1] - 1)
            self.roi_x1.setMaximum(shape[1])
            self.roi_y0.setMaximum(shape[0] - 1)
            self.roi_y1.setMaximum(shape[0])
            self.roi_x1.setValue(shape[1])
            self.roi_y1.setValue(shape[0])
        else:
            self.stack_y0.setMaximum(shape[1] - 1)
            self.stack_y1.setMaximum(shape[1])
            self.stack_y1.setValue(min(max(self.stack_y1.value(), self.stack_y0.value() + 1), shape[1]))
            self.frame_start.setMaximum(shape[0] - 1)
            self.frame_stop.setMaximum(shape[0])
            self.frame_stop.setValue(shape[0])
            self.snapshot_index_spin.setMaximum(shape[0] - 1)
            self.snapshot_index_spin.setValue(min(self.snapshot_index_spin.value(), shape[0] - 1))
        self._update_snapshot_navigation_enabled()

    def _detect_mode_index(self, signal) -> int:
        class_name = signal.__class__.__name__
        if class_name == "Signal2D":
            return 1
        if class_name == "Signal1D":
            return 0
        signal_dimension = getattr(signal.axes_manager, "signal_dimension", None)
        if signal_dimension == 2:
            return 1
        if signal_dimension == 1:
            return 0
        raise ValueError(
            f"Unsupported HyperSpy signal type '{class_name}'. Expected Signal1D for map data or Signal2D for snapshot data."
        )

    def _set_mode_from_signal(self, signal):
        mode_index = self._detect_mode_index(signal)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(mode_index)
        self.mode_combo.blockSignals(False)
        mode_label = "map workflow (Signal1D)" if mode_index == 0 else "snapshot workflow (Signal2D)"
        self._log(f"Detected {mode_label}.")

    def _update_mode_specific_ui(self):
        is_map_mode = self.mode_combo.currentIndex() == 0
        for widget in self.map_mode_widgets:
            widget.setEnabled(is_map_mode)
        for widget in self.snapshot_mode_widgets:
            widget.setEnabled(not is_map_mode)
        if is_map_mode:
            self.snapshot_index_spin.setEnabled(False)
            self.snapshot_prev_button.setEnabled(False)
            self.snapshot_next_button.setEnabled(False)
        else:
            self._update_snapshot_navigation_enabled()

    def _map_intensity_image(self) -> Optional[np.ndarray]:
        if self.loaded is None or self.mode_combo.currentIndex() != 0:
            return None
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return None
        e0 = max(0, min(self.energy_start_spin.value(), data.shape[2] - 1))
        e1 = max(e0 + 1, min(self.energy_stop_spin.value(), data.shape[2]))
        return data[:, :, e0:e1].sum(axis=2)

    def _default_map_polygon(self, width: int, height: int) -> list[tuple[float, float]]:
        return [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

    def _current_polygon_mask(self) -> Optional[np.ndarray]:
        image = self._map_intensity_image()
        if image is None:
            return None
        height, width = image.shape
        if not self.map_polygon_vertices or len(self.map_polygon_vertices) < 3:
            self.map_polygon_vertices = self._default_map_polygon(width, height)
        yy, xx = np.mgrid[0:height, 0:width]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        path = MplPath(self.map_polygon_vertices)
        mask = path.contains_points(points, radius=0.5).reshape((height, width))
        return mask

    def _map_intensity_range_values(self) -> tuple[float, float]:
        min_pos = self.map_mask_min_slider.value()
        max_pos = self.map_mask_max_slider.value()
        if min_pos > max_pos:
            min_pos, max_pos = max_pos, min_pos
        span = self._map_intensity_max - self._map_intensity_min
        if span <= 0:
            return self._map_intensity_min, self._map_intensity_max
        z_min = self._map_intensity_min + (span * min_pos / 1000.0)
        z_max = self._map_intensity_min + (span * max_pos / 1000.0)
        return z_min, z_max

    def _reset_map_histogram_controls(self):
        image = self._map_intensity_image()
        polygon_mask = self._current_polygon_mask()
        if image is None or polygon_mask is None or not polygon_mask.any():
            return
        values = image[polygon_mask]
        self._map_intensity_min = float(np.min(values))
        self._map_intensity_max = float(np.max(values))
        self.map_mask_min_slider.blockSignals(True)
        self.map_mask_max_slider.blockSignals(True)
        self.map_mask_min_slider.setValue(0)
        self.map_mask_max_slider.setValue(1000)
        self.map_mask_min_slider.blockSignals(False)
        self.map_mask_max_slider.blockSignals(False)
        self._update_map_histogram()

    def _update_map_histogram(self):
        self.map_hist_canvas.figure.clear()
        ax = self.map_hist_canvas.figure.add_subplot(111)
        image = self._map_intensity_image()
        polygon_mask = self._current_polygon_mask()
        if image is None or polygon_mask is None or not polygon_mask.any():
            ax.text(0.5, 0.5, "Draw a polygon ROI to inspect intensity histogram", ha="center", va="center")
            ax.set_axis_off()
            self.map_hist_canvas.draw_idle()
            return
        values = image[polygon_mask]
        ax.hist(values, bins=64, color="0.5", edgecolor="0.3")
        z_min, z_max = self._map_intensity_range_values()
        ax.axvline(z_min, color="royalblue", linewidth=1.2)
        ax.axvline(z_max, color="crimson", linewidth=1.2)
        ax.set_title("Polygon intensity histogram")
        ax.set_xlabel("Integrated intensity")
        ax.set_ylabel("Pixels")
        self.map_mask_min_label.setText(f"Mask min: {z_min:.2f}")
        self.map_mask_max_label.setText(f"Mask max: {z_max:.2f}")
        self.map_hist_canvas.draw_idle()

    def _on_map_mask_slider_changed(self):
        self._update_map_histogram()
        if self.mode_combo.currentIndex() == 0:
            self._update_map_mask_preview()

    def _update_map_mask_preview(self):
        if self.mode_combo.currentIndex() != 0:
            return
        self.corrected_canvas.figure.clear()
        ax = self.corrected_canvas.figure.add_subplot(111)
        image = self._map_intensity_image()
        polygon_mask = self._current_polygon_mask()
        if image is None or polygon_mask is None:
            ax.text(0.5, 0.5, "Masked image", ha="center", va="center")
            ax.set_axis_off()
            self.corrected_canvas.draw_idle()
            return
        z_min, z_max = self._map_intensity_range_values()
        selection_mask = polygon_mask & (image >= z_min) & (image <= z_max)
        masked = np.ma.masked_where(~selection_mask, image)
        ax.imshow(masked, cmap="inferno", origin="upper")
        ax.set_title("Masked image")
        ax.set_xlabel("")
        ax.set_ylabel("")
        self.corrected_canvas.draw_idle()

    def _draw_initial_image(self):
        preserve_view = self.loaded is not None and self.mode_combo.currentIndex() == 1
        if preserve_view and self.image_canvas.figure.axes:
            current_ax = self.image_canvas.figure.axes[0]
            self._image_view_limits = (current_ax.get_xlim(), current_ax.get_ylim())

        self.image_canvas.figure.clear()
        ax = self.image_canvas.figure.add_subplot(111)
        if self.loaded is None:
            ax.text(0.5, 0.5, "Load an EELS dataset to start", ha="center", va="center")
            ax.set_axis_off()
            self.image_canvas.draw_idle()
            return

        mode_index = self.mode_combo.currentIndex()
        data = np.asarray(self.loaded.eels_signal.data)
        if mode_index == 0:
            image = self._map_intensity_image()
            im = ax.imshow(image, cmap="inferno", origin="upper")
            ax.set_title("Map intensity preview")
            ax.set_xlabel("")
            ax.set_ylabel("")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            self.image_canvas.figure.colorbar(im, cax=cax)
        else:
            frame_index = min(self.snapshot_index_spin.value(), data.shape[0] - 1)
            detector_image = data[frame_index]
            ax.imshow(detector_image, cmap="viridis", origin="upper", aspect="auto")
            ax.set_title(f"Snapshot {frame_index}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            self._apply_saved_image_view(ax, detector_image.shape[1], detector_image.shape[0])

        if self.loaded.image_signal is not None and np.asarray(self.loaded.image_signal.data).ndim >= 2:
            if mode_index != 0:
                self.image_canvas.draw_idle()
                self._attach_selector()
                self._update_selection_overlay()
                return
            try:
                ax.imshow(np.asarray(self.loaded.image_signal.data), cmap="gray", alpha=0.25, origin="upper")
            except Exception:
                pass

        self.image_canvas.draw_idle()
        self._attach_selector()
        self._update_selection_overlay()
        if mode_index == 0:
            self._update_map_mask_preview()

    def _attach_selector(self):
        axes = self.image_canvas.figure.axes
        if not axes:
            return
        if self.mode_combo.currentIndex() == 0:
            self.selector = PolygonSelector(
                axes[0],
                self._on_polygon_selected,
                useblit=False,
                props={"color": "cyan", "linewidth": 1.5, "alpha": 0.9},
            )
        else:
            self.selector = RectangleSelector(
                axes[0],
                self._on_rectangle_selected,
                useblit=False,
                button=[1],
                interactive=True,
                drag_from_anywhere=True,
            )
        self._sync_selector_state()

    def _sync_selector_state(self):
        if self.selector is not None:
            self.selector.set_active(self.enable_selector.isChecked())

    def _on_polygon_selected(self, verts):
        if self.mode_combo.currentIndex() != 0:
            return
        self.map_polygon_vertices = [(float(x), float(y)) for x, y in verts]
        self._reset_map_histogram_controls()
        self._draw_initial_image()

    def _on_rectangle_selected(self, eclick, erelease):
        if eclick.ydata is None or erelease.ydata is None:
            return
        y0, y1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
        if self.mode_combo.currentIndex() == 0:
            if eclick.xdata is None or erelease.xdata is None:
                return
            x0, x1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
            self.roi_x0.blockSignals(True)
            self.roi_x1.blockSignals(True)
            self.roi_y0.blockSignals(True)
            self.roi_y1.blockSignals(True)
            self.roi_x0.setValue(x0)
            self.roi_x1.setValue(max(x0 + 1, x1))
            self.roi_y0.setValue(y0)
            self.roi_y1.setValue(max(y0 + 1, y1))
            self.roi_x0.blockSignals(False)
            self.roi_x1.blockSignals(False)
            self.roi_y0.blockSignals(False)
            self.roi_y1.blockSignals(False)
        else:
            self.stack_y0.blockSignals(True)
            self.stack_y1.blockSignals(True)
            self.stack_y0.setValue(y0)
            self.stack_y1.setValue(max(y0 + 1, y1))
            self.stack_y0.blockSignals(False)
            self.stack_y1.blockSignals(False)
        self._update_selection_overlay()

    def _update_selection_overlay(self):
        axes = self.image_canvas.figure.axes
        if not axes:
            return
        ax = axes[0]
        if self.selection_rect is not None:
            try:
                self.selection_rect.remove()
            except Exception:
                pass
        if self.selection_polygon is not None:
            try:
                self.selection_polygon.remove()
            except Exception:
                pass
        if self.mode_combo.currentIndex() == 0:
            polygon_mask = self._current_polygon_mask()
            if self.map_polygon_vertices is None and polygon_mask is not None:
                height, width = polygon_mask.shape
                self.map_polygon_vertices = self._default_map_polygon(width, height)
            if self.map_polygon_vertices:
                self.selection_polygon = Polygon(
                    self.map_polygon_vertices,
                    closed=True,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                )
                ax.add_patch(self.selection_polygon)
            self.image_canvas.draw_idle()
            return
        else:
            x0 = 0
            if self.loaded is not None:
                x1 = int(self.loaded.eels_signal.data.shape[2])
            else:
                x1 = int(round(max(ax.get_xlim())))
            y0 = self.stack_y0.value()
            y1 = self.stack_y1.value()
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        self.selection_rect = Rectangle((x0, y0), width, height, fill=False, edgecolor="cyan", linewidth=1.5)
        ax.add_patch(self.selection_rect)
        if (
            self.mode_combo.currentIndex() == 1
            and self.loaded is not None
            and self._image_view_limits is None
        ):
            ax.set_xlim(0, self.loaded.eels_signal.data.shape[2] - 1)
            ax.set_ylim(self.loaded.eels_signal.data.shape[1] - 1, 0)
        self.image_canvas.draw_idle()

    def _on_mode_changed(self):
        if self.loaded is not None:
            try:
                detected = self._detect_mode_index(self.loaded.eels_signal)
            except ValueError:
                detected = self.mode_combo.currentIndex()
            if detected != self.mode_combo.currentIndex():
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(detected)
                self.mode_combo.blockSignals(False)
        self._update_mode_specific_ui()
        self._draw_initial_image()

    def _update_snapshot_navigation_enabled(self):
        enabled = self.loaded is not None and self.mode_combo.currentIndex() == 1
        self.snapshot_index_spin.setEnabled(enabled)
        self.snapshot_prev_button.setEnabled(enabled)
        self.snapshot_next_button.setEnabled(enabled)

    def _show_previous_snapshot(self):
        if not self.snapshot_index_spin.isEnabled():
            return
        self.snapshot_index_spin.setValue(max(0, self.snapshot_index_spin.value() - 1))

    def _show_next_snapshot(self):
        if not self.snapshot_index_spin.isEnabled():
            return
        self.snapshot_index_spin.setValue(
            min(self.snapshot_index_spin.maximum(), self.snapshot_index_spin.value() + 1)
        )

    def _suggest_export_dir(self) -> Path:
        if self.loaded is None or not self.loaded.eels_path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return Path.cwd() / "vibeels-analysis" / timestamp
        data_path = Path(self.loaded.eels_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return data_path.parent / "vibeels-analysis" / timestamp

    def _choose_export_dir(self) -> Optional[Path]:
        suggested = self._suggest_export_dir()
        suggested.parent.mkdir(parents=True, exist_ok=True)

        while True:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Choose result folder name",
                str(suggested),
                "Folder (*)",
            )
            if not selected:
                return None

            target_dir = Path(selected)
            if target_dir.exists():
                reply = QtWidgets.QMessageBox.warning(
                    self,
                    "Folder already exists",
                    (
                        f"The folder\n\n{target_dir}\n\nalready exists. "
                        "Saving may overwrite existing files."
                    ),
                    QtWidgets.QMessageBox.Retry
                    | QtWidgets.QMessageBox.Ok
                    | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Retry,
                )
                if reply == QtWidgets.QMessageBox.Retry:
                    suggested = target_dir
                    continue
                if reply == QtWidgets.QMessageBox.Cancel:
                    return None
            return target_dir

    def _collect_export_arrays(self) -> dict[str, np.ndarray]:
        if self.current_result is None:
            return {}

        result = self.current_result
        arrays: dict[str, np.ndarray] = {
            "spectrum_x_raw": np.asarray(result.energy_axis_raw, dtype=float),
            "zlp_x": np.asarray(result.zero_loss_fit.fit_x, dtype=float),
            "zlp_y_integrated": np.asarray(result.zero_loss_fit.fit_y, dtype=float),
            "zlp_fit": np.asarray(result.zero_loss_fit.best_fit, dtype=float),
            "zlp_center_ev": np.asarray([result.zero_loss_fit.center_ev], dtype=float),
        }

        if isinstance(result, StackProcessingResult):
            frame_index = min(self.snapshot_index_spin.value(), self.loaded.eels_signal.data.shape[0] - 1)
            detector_image = np.asarray(self.loaded.eels_signal.data[frame_index], dtype=float)
            corrected_image = self._current_snapshot_aligned_image()
            current_spectrum = self._current_snapshot_spectrum()
            fit_mask = (
                (result.energy_axis_raw >= self.fit_start_spin.value())
                & (result.energy_axis_raw <= self.fit_stop_spin.value())
            )
            snapshot_fit_y = np.asarray(current_spectrum[fit_mask], dtype=float) if current_spectrum is not None else np.array([])
            reference_max = float(np.max(result.zero_loss_fit.fit_y)) if result.zero_loss_fit.fit_y.size else 1.0
            snapshot_max = float(np.max(snapshot_fit_y)) if snapshot_fit_y.size else 1.0
            scale = reference_max / snapshot_max if snapshot_max > 0 else 1.0
            arrays.update(
                {
                    "snapshot_index": np.asarray([frame_index], dtype=int),
                    "detector_image_snapshot": detector_image,
                    "detector_image_aligned_snapshot": np.asarray(corrected_image, dtype=float)
                    if corrected_image is not None
                    else np.array([]),
                    "zlp_snapshot_x": np.asarray(result.energy_axis_raw[fit_mask], dtype=float),
                    "zlp_snapshot_y_scaled": snapshot_fit_y * scale,
                    "zlp_snapshot_scale": np.asarray([scale], dtype=float),
                }
            )
        else:
            arrays.update(
                {
                    "map_display_image": np.asarray(result.display_image, dtype=float),
                    "map_intensity_image": np.asarray(result.intensity_image, dtype=float),
                    "map_masked_image": np.asarray(np.nan_to_num(result.masked_image, nan=np.nan), dtype=float),
                    "map_selection_mask": np.asarray(result.selection_mask, dtype=int),
                }
            )

        return arrays

    def _collect_export_parameters(self) -> dict[str, object]:
        mode = "snapshot" if self.mode_combo.currentIndex() == 1 else "map"
        params: dict[str, object] = {
            "saved_at": datetime.now().isoformat(),
            "mode": mode,
            "eels_file": self.loaded.eels_path if self.loaded else None,
            "reference_image": self.loaded.image_path if self.loaded else None,
            "fit_window_ev": [self.fit_start_spin.value(), self.fit_stop_spin.value()],
            "guess_window_ev": [self.guess_start_spin.value(), self.guess_stop_spin.value()],
            "zero_loss_center_ev": float(self.current_result.zero_loss_fit.center_ev),
        }
        if mode == "map":
            params.update(
                {
                    "energy_range_pixels": [self.energy_start_spin.value(), self.energy_stop_spin.value()],
                    "intensity_range": list(self._map_intensity_range_values()),
                    "polygon_vertices": self.map_polygon_vertices,
                }
            )
        else:
            params.update(
                {
                    "detector_vertical_range": [self.stack_y0.value(), self.stack_y1.value()],
                    "snapshot_range": [self.frame_start.value(), self.frame_stop.value()],
                    "snapshot_index_for_preview": self.snapshot_index_spin.value(),
                }
            )
        return params

    def _build_repro_script(self, data_stem: str) -> str:
        return f'''from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np


bundle_dir = Path(__file__).resolve().parent
stem = "{data_stem}"
arrays = np.load(bundle_dir / f"{{stem}}_graph_data.npz")
params = json.loads((bundle_dir / f"{{stem}}.json").read_text())
spectrum_xy = np.loadtxt(bundle_dir / f"{{stem}}.xy", skiprows=1)

fig = plt.figure(figsize=(14, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

ax0 = fig.add_subplot(gs[0, 0])
if "detector_image_snapshot" in arrays:
    ax0.imshow(arrays["detector_image_snapshot"], cmap="viridis", origin="upper", aspect="auto")
    ax0.set_title(f"Snapshot {{int(arrays['snapshot_index'][0])}}")
    ax0.set_xlabel("spectral channel")
    ax0.set_ylabel("detector vertical")
else:
    ax0.imshow(arrays["map_display_image"], cmap="gray", origin="upper")
    ax0.contour(arrays["map_selection_mask"], levels=[0.5], colors=["cyan"], linewidths=1.0)
    ax0.set_title("Map ROI")

ax1 = fig.add_subplot(gs[0, 1])
if "detector_image_aligned_snapshot" in arrays and arrays["detector_image_aligned_snapshot"].size:
    ax1.imshow(arrays["detector_image_aligned_snapshot"], cmap="viridis", origin="upper", aspect="auto")
    ax1.set_title(f"Snapshot {{int(arrays['snapshot_index'][0])}} aligned")
    ax1.set_xlabel("spectral channel")
    ax1.set_ylabel("aligned detector rows")
else:
    ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(arrays["zlp_x"], arrays["zlp_y_integrated"], "k.", markersize=3, label="integrated")
ax2.plot(arrays["zlp_x"], arrays["zlp_fit"], color="crimson", linewidth=1.2, label="fit")
ax2.axvline(arrays["zlp_center_ev"][0], color="royalblue", linestyle="--", linewidth=1.0)
if "zlp_snapshot_x" in arrays and arrays["zlp_snapshot_x"].size:
    scale = arrays["zlp_snapshot_scale"][0]
    snap_index = int(arrays["snapshot_index"][0])
    ax2.plot(
        arrays["zlp_snapshot_x"],
        arrays["zlp_snapshot_y_scaled"],
        color="darkorange",
        linewidth=1.0,
        label=f"snapshot {{snap_index}} (scaled x{{scale:.2f}})",
    )
ax2.set_title("ZLP, aligned")
ax2.set_xlabel("Energy loss (eV)")
ax2.set_ylabel("ZLP region")
ax2.legend(loc="lower left", fontsize=8)

ax3 = fig.add_subplot(gs[1, :])
ax3.plot(spectrum_xy[:, 0], spectrum_xy[:, 1], color="black", linewidth=1.0)
ax3.set_xlabel("Energy loss (eV, ZLP corrected)")
ax3.set_ylabel("Intensity (a.u.)")
sec = ax3.secondary_xaxis("top", functions=(lambda x: x * 8065.54429, lambda x: x / 8065.54429))
sec.set_xlabel(r"Wavenumber (cm$^{{-1}}$)")

plt.show()
'''

    def _save_results(self):
        if self.loaded is None or self.current_result is None:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing to save",
                "Load data and run the processing before saving results.",
            )
            return

        export_dir = self._choose_export_dir()
        if export_dir is None:
            return

        export_dir.mkdir(parents=True, exist_ok=True)
        data_stem = Path(self.loaded.eels_path).stem
        arrays = self._collect_export_arrays()
        params = self._collect_export_parameters()

        xy_path = export_dir / f"{data_stem}.xy"
        json_path = export_dir / f"{data_stem}.json"
        npz_path = export_dir / f"{data_stem}_graph_data.npz"
        script_path = export_dir / f"{data_stem}_reproduce.py"

        spectrum_xy = np.column_stack(
            [
                np.asarray(self.current_result.energy_axis_calibrated, dtype=float),
                np.asarray(self.current_result.summed_spectrum, dtype=float),
            ]
        )
        np.savetxt(xy_path, spectrum_xy, header="x y", comments="")
        json_path.write_text(json.dumps(params, indent=2))
        np.savez(npz_path, **arrays)
        script_path.write_text(self._build_repro_script(data_stem))

        self._log(f"Saved results to {export_dir}")
        QtWidgets.QMessageBox.information(
            self,
            "Results saved",
            f"Saved analysis bundle to:\n\n{export_dir}",
        )

    def _apply_saved_image_view(self, ax, x_size: int, y_size: int):
        if self._image_view_limits is None:
            ax.set_xlim(0, x_size - 1)
            ax.set_ylim(y_size - 1, 0)
            return

        (x0, x1), (y0, y1) = self._image_view_limits
        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        # Ignore stale/default limits such as (0, 1) or limits fully outside the image.
        if (
            (x_max - x_min) < 2
            or (y_max - y_min) < 2
            or x_max < 0
            or x_min > (x_size - 1)
            or y_max < 0
            or y_min > (y_size - 1)
        ):
            self._image_view_limits = None
            ax.set_xlim(0, x_size - 1)
            ax.set_ylim(y_size - 1, 0)
            return

        x0 = min(max(x0, 0), x_size - 1)
        x1 = min(max(x1, 0), x_size - 1)
        y0 = min(max(y0, 0), y_size - 1)
        y1 = min(max(y1, 0), y_size - 1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    def _on_snapshot_index_changed(self):
        if self.mode_combo.currentIndex() == 1 and isinstance(self.current_result, StackProcessingResult):
            self._render_result(self.current_result)
            return
        self._draw_initial_image()

    def _current_snapshot_spectrum(self) -> Optional[np.ndarray]:
        if self.loaded is None or self.mode_combo.currentIndex() != 1:
            return None
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return None
        frame_index = min(self.snapshot_index_spin.value(), data.shape[0] - 1)
        y0 = max(0, min(self.stack_y0.value(), data.shape[1] - 1))
        y1 = max(y0 + 1, min(self.stack_y1.value(), data.shape[1]))
        aligned_rows = align_spectra_1d(data[frame_index, y0:y1, :].copy())
        return aligned_rows.sum(axis=0)

    def _current_snapshot_aligned_image(self) -> Optional[np.ndarray]:
        if self.loaded is None or self.mode_combo.currentIndex() != 1:
            return None
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return None
        frame_index = min(self.snapshot_index_spin.value(), data.shape[0] - 1)
        y0 = max(0, min(self.stack_y0.value(), data.shape[1] - 1))
        y1 = max(y0 + 1, min(self.stack_y1.value(), data.shape[1]))
        return align_spectra_1d(data[frame_index, y0:y1, :].copy())

    def _fit_window(self):
        return (self.fit_start_spin.value(), self.fit_stop_spin.value())

    def _guess_window(self):
        return (self.guess_start_spin.value(), self.guess_stop_spin.value())

    def _process_current(self):
        if self.loaded is None or not self.loaded.eels_path:
            self._log("Load an EELS dataset before processing.")
            return
        try:
            if self.mode_combo.currentIndex() == 0:
                polygon_mask = self._current_polygon_mask()
                if polygon_mask is None:
                    raise ValueError("Map polygon ROI is not available.")
                result = process_map_dataset(
                    self.loaded.eels_signal,
                    energy_range=(self.energy_start_spin.value(), self.energy_stop_spin.value()),
                    polygon_mask=polygon_mask,
                    intensity_range=self._map_intensity_range_values(),
                    display_image=self._current_display_image(),
                    fit_window=self._fit_window(),
                    guess_window=self._guess_window(),
                )
            else:
                result = process_snapshot_stack(
                    self.loaded.eels_signal,
                    vertical_range=(self.stack_y0.value(), self.stack_y1.value()),
                    frame_range=(self.frame_start.value(), self.frame_stop.value()),
                    fit_window=self._fit_window(),
                    guess_window=self._guess_window(),
                )
        except Exception as exc:
            self._log(f"Processing failed: {exc}")
            return

        self.current_result = result
        self._render_result(result)
        self._log("Processing complete.")

    def _current_display_image(self):
        if self.loaded is None or self.loaded.image_signal is None:
            return None
        data = np.asarray(self.loaded.image_signal.data)
        if data.ndim == 2:
            return data
        if data.ndim > 2:
            return np.squeeze(data)
        return None

    def _render_result(self, result):
        if isinstance(result, StackProcessingResult) and self.image_canvas.figure.axes:
            current_ax = self.image_canvas.figure.axes[0]
            self._image_view_limits = (current_ax.get_xlim(), current_ax.get_ylim())
        self.image_canvas.figure.clear()
        image_ax = self.image_canvas.figure.add_subplot(111)

        if isinstance(result, MapProcessingResult):
            im = image_ax.imshow(result.intensity_image, cmap="inferno", origin="upper")
            divider = make_axes_locatable(image_ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            self.image_canvas.figure.colorbar(im, cax=cax)
            image_ax.set_title("Map intensity preview")
            pixel_count_text = str(result.selected_pixel_count)
            cal_axis = result.energy_axis_calibrated
            image_ax.set_xlabel("")
            image_ax.set_ylabel("")
        else:
            frame_index = min(self.snapshot_index_spin.value(), self.loaded.eels_signal.data.shape[0] - 1)
            detector_image = self.loaded.eels_signal.data[frame_index]
            image_ax.imshow(detector_image, cmap="viridis", origin="upper", aspect="auto")
            band_height = max(1, self.stack_y1.value() - self.stack_y0.value())
            image_ax.add_patch(
                Rectangle(
                    (0, self.stack_y0.value()),
                    detector_image.shape[1],
                    band_height,
                    edgecolor="cyan",
                    fill=False,
                    linewidth=1.5,
                )
            )
            self._apply_saved_image_view(image_ax, detector_image.shape[1], detector_image.shape[0])
            image_ax.set_title(f"Snapshot {frame_index}")
            pixel_count_text = str(result.aligned_stack.shape[0] * result.aligned_stack.shape[1])
            cal_axis = result.energy_axis_calibrated
            image_ax.set_xlabel("")
            image_ax.set_ylabel("")

        self.image_canvas.draw_idle()
        self._image_view_limits = (image_ax.get_xlim(), image_ax.get_ylim())
        if isinstance(result, StackProcessingResult):
            image_ax.callbacks.connect("xlim_changed", self._sync_corrected_x_from_image)
        self._attach_selector()
        self._update_selection_overlay()

        self.corrected_canvas.figure.clear()
        if isinstance(result, StackProcessingResult) and self.corrected_canvas.figure.axes:
            current_ax = self.corrected_canvas.figure.axes[0]
            self._corrected_view_limits = (current_ax.get_xlim(), current_ax.get_ylim())
        corrected_ax = self.corrected_canvas.figure.add_subplot(111)
        if isinstance(result, StackProcessingResult):
            corrected_image = self._current_snapshot_aligned_image()
            if corrected_image is not None:
                corrected_ax.imshow(corrected_image, cmap="viridis", origin="upper", aspect="auto")
                corrected_ax.set_title(f"Snapshot {self.snapshot_index_spin.value()} aligned")
                corrected_ax.set_xlabel("")
                corrected_ax.set_ylabel("")
                self._apply_saved_corrected_view(
                    corrected_ax,
                    corrected_image.shape[1],
                    corrected_image.shape[0],
                )
                self._match_corrected_x_to_image(corrected_ax, corrected_image.shape[1])
            else:
                corrected_ax.text(0.5, 0.5, "No corrected image", ha="center", va="center")
                corrected_ax.set_axis_off()
        else:
            masked = np.ma.masked_invalid(result.masked_image)
            corrected_ax.imshow(masked, cmap="inferno", origin="upper")
            corrected_ax.set_title("Masked image")
            corrected_ax.set_xlabel("")
            corrected_ax.set_ylabel("")
        self.corrected_canvas.draw_idle()

        self.fit_canvas.figure.clear()
        fit_ax = self.fit_canvas.figure.add_subplot(111)
        if isinstance(result, StackProcessingResult):
            fit_ax.plot(result.zero_loss_fit.fit_x, result.zero_loss_fit.fit_y, "k.", markersize=3)
            fit_ax.plot(result.zero_loss_fit.fit_x, result.zero_loss_fit.best_fit, color="crimson", linewidth=1.2)
            fit_ax.axvline(
                result.zero_loss_fit.center_ev,
                color="royalblue",
                linewidth=1.0,
                linestyle="--",
                label=f"centroid = {result.zero_loss_fit.center_ev:.5f} eV",
            )
            current_spectrum = self._current_snapshot_spectrum()
            if current_spectrum is not None:
                fit_mask = (
                    (result.energy_axis_raw >= self.fit_start_spin.value())
                    & (result.energy_axis_raw <= self.fit_stop_spin.value())
                )
                snapshot_fit_y = current_spectrum[fit_mask]
                reference_max = float(np.max(result.zero_loss_fit.fit_y)) if result.zero_loss_fit.fit_y.size else 1.0
                snapshot_max = float(np.max(snapshot_fit_y)) if snapshot_fit_y.size else 1.0
                scale = reference_max / snapshot_max if snapshot_max > 0 else 1.0
                fit_ax.plot(
                    result.energy_axis_raw[fit_mask],
                    snapshot_fit_y * scale,
                    color="darkorange",
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"snapshot {self.snapshot_index_spin.value()} (scaled x{scale:.2f})",
                )
        else:
            fit_mask = (
                (result.energy_axis_raw >= self.fit_start_spin.value())
                & (result.energy_axis_raw <= self.fit_stop_spin.value())
            )
            zlp_stack = result.selected_spectra[:, fit_mask]
            for row in zlp_stack:
                fit_ax.plot(
                    result.energy_axis_raw[fit_mask],
                    row,
                    color="0.5",
                    linewidth=0.6,
                    alpha=0.25,
                )
            fit_ax.plot(result.zero_loss_fit.fit_x, result.zero_loss_fit.fit_y, "k.", markersize=3, label="integrated")
            fit_ax.plot(result.zero_loss_fit.fit_x, result.zero_loss_fit.best_fit, color="crimson", linewidth=1.2, label="fit")
            fit_ax.axvline(
                result.zero_loss_fit.center_ev,
                color="royalblue",
                linewidth=1.0,
                linestyle="--",
                label=f"centroid = {result.zero_loss_fit.center_ev:.5f} eV",
            )
        fit_ax.set_xlabel("Energy loss (eV)")
        fit_ax.set_ylabel("ZLP region")
        fit_ax.set_title("ZLP, aligned")
        fit_ax.legend(loc="lower left", fontsize=8)

        self.fit_canvas.draw_idle()

        self.spectrum_canvas.figure.clear()
        spectrum_ax = self.spectrum_canvas.figure.add_subplot(111)
        spectrum_ax.plot(cal_axis, result.summed_spectrum, color="black", linewidth=1.0)
        spectrum_ax.set_xlabel("Energy loss (eV, ZLP corrected)")
        spectrum_ax.set_ylabel("Intensity (a.u.)")
        spectrum_ax_top = spectrum_ax.secondary_xaxis(
            "top",
            functions=(lambda x: x * EV_TO_CMINV, lambda x: x / EV_TO_CMINV),
        )
        spectrum_ax_top.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        self.spectrum_canvas.draw_idle()

        self.zlp_center_label.setText(f"{result.zero_loss_fit.center_ev:.6f} eV")
        self.pixel_count_label.setText(pixel_count_text)
        self.axis_span_label.setText(f"{cal_axis.min():.4f} to {cal_axis.max():.4f} eV")

    def _apply_saved_corrected_view(self, ax, x_size: int, y_size: int):
        if self._corrected_view_limits is None:
            ax.set_xlim(0, x_size - 1)
            ax.set_ylim(y_size - 1, 0)
            return

        (x0, x1), (y0, y1) = self._corrected_view_limits
        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        if (
            (x_max - x_min) < 2
            or (y_max - y_min) < 2
            or x_max < 0
            or x_min > (x_size - 1)
            or y_max < 0
            or y_min > (y_size - 1)
        ):
            self._corrected_view_limits = None
            ax.set_xlim(0, x_size - 1)
            ax.set_ylim(y_size - 1, 0)
            return

        x0 = min(max(x0, 0), x_size - 1)
        x1 = min(max(x1, 0), x_size - 1)
        y0 = min(max(y0, 0), y_size - 1)
        y1 = min(max(y1, 0), y_size - 1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    def _match_corrected_x_to_image(self, corrected_ax, corrected_width: int):
        if not self.image_canvas.figure.axes:
            return
        image_ax = self.image_canvas.figure.axes[0]
        x0, x1 = image_ax.get_xlim()
        x0 = min(max(x0, 0), corrected_width - 1)
        x1 = min(max(x1, 0), corrected_width - 1)
        corrected_ax.set_xlim(x0, x1)

    def _sync_corrected_x_from_image(self, image_ax):
        if self._syncing_corrected_xlim or not self.corrected_canvas.figure.axes:
            return
        corrected_ax = self.corrected_canvas.figure.axes[0]
        x0, x1 = image_ax.get_xlim()
        corrected_width = None
        for image in corrected_ax.images:
            corrected_width = image.get_array().shape[1]
            break
        if corrected_width is None:
            return
        x0 = min(max(x0, 0), corrected_width - 1)
        x1 = min(max(x1, 0), corrected_width - 1)
        self._syncing_corrected_xlim = True
        try:
            corrected_ax.set_xlim(x0, x1)
            self._corrected_view_limits = (corrected_ax.get_xlim(), corrected_ax.get_ylim())
            self.corrected_canvas.draw_idle()
        finally:
            self._syncing_corrected_xlim = False


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VibeelsWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

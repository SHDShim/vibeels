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

matplotlib.use("QtAgg")

import numpy as np
from matplotlib import rcParams, ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt6 import QtCore, QtGui, QtWidgets

from .processing import (
    EV_TO_CMINV,
    MapProcessingResult,
    StackProcessingResult,
    describe_signal_layout,
    load_signal,
    load_eels_signal,
    process_map_dataset,
    process_snapshot_stack,
)
from .theme import (
    PLOT_CYCLE,
    PLOT_FIT,
    PLOT_MAIN,
    PLOT_PREVIEW,
    PLOT_WARNING,
    SPINE,
    apply_qt_theme,
    configure_matplotlib_defaults,
    style_colorbar,
    style_mpl_axes,
    style_secondary_axis,
)
from .version import __version__


QtOrientation = QtCore.Qt.Orientation
QtItemFlag = QtCore.Qt.ItemFlag
QtCheckState = QtCore.Qt.CheckState
HeaderResizeMode = QtWidgets.QHeaderView.ResizeMode
SelectionBehavior = QtWidgets.QAbstractItemView.SelectionBehavior
SelectionMode = QtWidgets.QAbstractItemView.SelectionMode
MessageBoxButton = QtWidgets.QMessageBox.StandardButton

configure_matplotlib_defaults()


def _application_icon_path() -> Optional[Path]:
    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root / "assets" / "icons" / "vibeels.png",
        package_root.parent / "asset" / "icons" / "vibeels.png",
        package_root.parent / "asset" / "icon.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _application_icon() -> Optional[QtGui.QIcon]:
    icon_path = _application_icon_path()
    if icon_path is None:
        return None
    icon = QtGui.QIcon(str(icon_path))
    return icon if not icon.isNull() else None


def _set_macos_dock_icon(icon_path: Optional[Path]) -> None:
    if sys.platform != "darwin" or icon_path is None:
        return
    try:
        from AppKit import NSApplication, NSImage
    except Exception:
        return
    image = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
    if image is not None:
        NSApplication.sharedApplication().setApplicationIconImage_(image)


@dataclass
class LoadedData:
    eels_signal: object
    image_signal: Optional[object]
    eels_path: str
    image_path: Optional[str]


@dataclass
class SavedMapEntry:
    mode: str
    show: bool
    locked: bool
    comment: str
    roi_text: str
    polygon_vertices: list
    selection_mask: np.ndarray
    display_image: np.ndarray
    masked_image: np.ndarray
    intensity_range: tuple[float, float]
    energy_axis_raw: np.ndarray
    fit_window: tuple[float, float]
    selected_spectra: np.ndarray
    energy_axis: np.ndarray
    spectrum: np.ndarray


@dataclass
class SavedStateRecord:
    folder_name: str
    saved_at: str
    comment: str
    directory: Path


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
        style_mpl_axes(self.figure, *self.axes)

    def reset_axis(self, subplot_spec=111):
        self.figure.clear()
        axis = self.figure.add_subplot(subplot_spec)
        self.axes = [axis]
        style_mpl_axes(self.figure, axis)
        return axis


class RangeSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, orientation=QtOrientation.Horizontal, parent=None):
        super().__init__(parent)
        self._orientation = orientation
        self._minimum = 0
        self._maximum = 100
        self._lower_value = 0
        self._upper_value = 100
        self._active_handle: Optional[str] = None
        self.setMouseTracking(True)
        self.setMinimumHeight(28)

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    def lowerValue(self) -> int:
        return self._lower_value

    def upperValue(self) -> int:
        return self._upper_value

    def setRange(self, minimum: int, maximum: int):
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self.setValues(self._lower_value, self._upper_value)

    def setValues(self, lower: int, upper: int):
        lower = self._clamp(int(lower))
        upper = self._clamp(int(upper))
        if lower > upper:
            lower, upper = upper, lower
        changed = lower != self._lower_value or upper != self._upper_value
        self._lower_value = lower
        self._upper_value = upper
        self.update()
        if changed:
            self.valueChanged.emit(self._lower_value, self._upper_value)

    def setLowerValue(self, value: int):
        self.setValues(value, self._upper_value)

    def setUpperValue(self, value: int):
        self.setValues(self._lower_value, value)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(240, 32)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        groove_rect = self._groove_rect()
        lower_center = self._handle_center_x(self._lower_value)
        upper_center = self._handle_center_x(self._upper_value)
        selected_rect = QtCore.QRectF(
            min(lower_center, upper_center),
            groove_rect.top(),
            abs(upper_center - lower_center),
            groove_rect.height(),
        )

        painter.setPen(QtGui.QPen(QtGui.QColor("#202227"), 1))
        painter.setBrush(QtGui.QColor("#3c4048"))
        painter.drawRoundedRect(groove_rect, 3, 3)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#6fb6ff"))
        painter.drawRoundedRect(selected_rect, 3, 3)

        for center_x in [lower_center, upper_center]:
            handle_rect = self._handle_rect(center_x)
            painter.setPen(QtGui.QPen(QtGui.QColor("#2a2d33"), 1))
            painter.setBrush(QtGui.QColor("#535862"))
            painter.drawRoundedRect(handle_rect, 4, 4)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        pos_x = event.position().x()
        lower_center = self._handle_center_x(self._lower_value)
        upper_center = self._handle_center_x(self._upper_value)
        if abs(pos_x - lower_center) <= abs(pos_x - upper_center):
            self._active_handle = "lower"
        else:
            self._active_handle = "upper"
        self._set_active_from_position(pos_x)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._active_handle is None:
            event.ignore()
            return
        self._set_active_from_position(event.position().x())
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._active_handle = None
        event.accept()

    def _clamp(self, value: int) -> int:
        return min(max(value, self._minimum), self._maximum)

    def _groove_rect(self) -> QtCore.QRectF:
        margin = 12.0
        groove_height = 8.0
        y = (self.height() - groove_height) / 2.0
        return QtCore.QRectF(margin, y, max(1.0, self.width() - (margin * 2.0)), groove_height)

    def _handle_rect(self, center_x: float) -> QtCore.QRectF:
        handle_width = 12.0
        handle_height = 22.0
        y = (self.height() - handle_height) / 2.0
        return QtCore.QRectF(center_x - handle_width / 2.0, y, handle_width, handle_height)

    def _handle_center_x(self, value: int) -> float:
        groove_rect = self._groove_rect()
        span = self._maximum - self._minimum
        if span <= 0:
            return groove_rect.left()
        ratio = (value - self._minimum) / span
        return groove_rect.left() + (ratio * groove_rect.width())

    def _value_from_position(self, pos_x: float) -> int:
        groove_rect = self._groove_rect()
        if groove_rect.width() <= 0:
            return self._minimum
        ratio = (pos_x - groove_rect.left()) / groove_rect.width()
        ratio = min(max(ratio, 0.0), 1.0)
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def _set_active_from_position(self, pos_x: float):
        value = self._value_from_position(pos_x)
        if self._active_handle == "lower":
            self.setLowerValue(min(value, self._upper_value))
        elif self._active_handle == "upper":
            self.setUpperValue(max(value, self._lower_value))


def set_spinbox_max_width_fraction(widget, fraction: float = 0.7) -> None:
    hint_width = max(1, widget.sizeHint().width())
    widget.setMaximumWidth(max(60, int(round(hint_width * float(fraction)))))


class ProcessingWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        mode_index: int,
        signal,
        map_kwargs: Optional[dict[str, object]] = None,
        snapshot_kwargs: Optional[dict[str, object]] = None,
    ):
        super().__init__()
        self.mode_index = mode_index
        self.signal = signal
        self.map_kwargs = map_kwargs or {}
        self.snapshot_kwargs = snapshot_kwargs or {}

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if self.mode_index == 0:
                result = process_map_dataset(
                    self.signal,
                    progress_callback=self._emit_progress,
                    **self.map_kwargs,
                )
            else:
                result = process_snapshot_stack(
                    self.signal,
                    progress_callback=self._emit_progress,
                    **self.snapshot_kwargs,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    def _emit_progress(self, current: int, total: int, message: str):
        self.progress.emit(int(current), int(total), str(message))


class VibeelsWindow(QtWidgets.QMainWindow):
    SETTINGS_GROUP = "paths"
    LAST_DATA_DIR_KEY = "last_data_dir"
    TITLE_FONT_SIZE_KEY = "title_font_size"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"vibeels {__version__}")
        icon = _application_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        self.resize(1440, 900)

        self.settings = QtCore.QSettings("vibeels", "vibeels")
        self.loaded: Optional[LoadedData] = None
        self.current_result: Optional[object] = None
        self.selector: Optional[object] = None
        self.selection_rect: Optional[Rectangle] = None
        self.selection_polygons: list[Polygon] = []
        self._image_view_limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._corrected_view_limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._syncing_corrected_xlim = False
        self.map_polygon_vertices: list[list[tuple[float, float]]] = []
        self._map_intensity_min = 0.0
        self._map_intensity_max = 1.0
        self.saved_map_entries: list[SavedMapEntry] = []
        self._updating_saved_map_table = False
        self.saved_state_records: list[SavedStateRecord] = []
        self._updating_saved_state_table = False
        self.processing_thread: Optional[QtCore.QThread] = None
        self.processing_worker: Optional[ProcessingWorker] = None
        self._last_progress_message: Optional[str] = None
        self._active_map_histogram_handle: Optional[str] = None
        self._snapshot_intensity_min = 0.0
        self._snapshot_intensity_max = 1.0
        self._active_snapshot_histogram_handle: Optional[str] = None
        self._plot_toolbars: dict[FigureCanvas, NavigationToolbar] = {}
        self._mouse_mode = "zoom"
        self._main_canvas_mouse_cids: list[tuple[FigureCanvas, int]] = []

        self._build_ui()
        self._apply_default_ranges()
        self._update_reference_image_preview()
        self._refresh_saved_state_records()

    def _make_button(
        self,
        text: str,
        slot=None,
        variant: str = "default",
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        if variant == "success":
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: #1f8f4d;
                    border: 1px solid #29b765;
                    border-radius: 4px;
                    color: #f7fff9;
                    min-height: 38px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #24a258;
                }
                QPushButton:pressed {
                    background-color: #18733e;
                }
                QPushButton:disabled {
                    background-color: #3d4a42;
                    border: 1px solid #56645b;
                    color: #aeb8b1;
                }
                """
            )
        elif variant == "danger":
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: #9f2436;
                    border: 1px solid #d24b62;
                    border-radius: 4px;
                    color: #fff7f8;
                    min-height: 38px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #b12a3d;
                }
                QPushButton:pressed {
                    background-color: #811d2c;
                }
                QPushButton:disabled {
                    background-color: #4d3c40;
                    border: 1px solid #655256;
                    color: #c3b4b7;
                }
                """
            )
        if slot is not None:
            button.clicked.connect(slot)
        return button

    def _make_plot_toolbar(self, canvas: FigureCanvas) -> NavigationToolbar:
        toolbar = NavigationToolbar(canvas, self)
        toolbar.hide()
        self._plot_toolbars[canvas] = toolbar
        return toolbar

    def _style_mouse_mode_buttons(self) -> None:
        self.zoom_button.setStyleSheet(
            """
            QPushButton {
                background-color: #eee8dc;
                border: 1px solid #7b7569;
                border-top-left-radius: 8px;
                border-bottom-left-radius: 8px;
                color: #2b2924;
                font-size: 13px;
                font-weight: 700;
                padding: 2px 14px;
            }
            QPushButton:checked {
                background-color: #d9a400;
                border-color: #8a6b00;
                color: #17130a;
            }
            """
        )
        self.roi_button.setStyleSheet(
            """
            QPushButton {
                background-color: #eee8dc;
                border: 1px solid #7b7569;
                border-left: 0;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                color: #2b2924;
                font-size: 13px;
                font-weight: 700;
                padding: 2px 14px;
            }
            QPushButton:checked {
                background-color: #d9a400;
                border-color: #8a6b00;
                color: #17130a;
            }
            """
        )

    def _set_mouse_mode(self, mode: str, warn: bool = False) -> None:
        if mode not in {"zoom", "roi"}:
            return
        if warn:
            QtWidgets.QMessageBox.warning(
                self,
                "ROI unavailable",
                "ROI editing is available only in the top-left graph. Switching to zoom.",
            )
            self._log("ROI editing is available only in the top-left graph. Switched to zoom.")

        self._mouse_mode = mode
        self.zoom_button.blockSignals(True)
        self.roi_button.blockSignals(True)
        self.zoom_button.setChecked(mode == "zoom")
        self.roi_button.setChecked(mode == "roi")
        self.zoom_button.blockSignals(False)
        self.roi_button.blockSignals(False)

        for toolbar in self._plot_toolbars.values():
            if toolbar.mode:
                toolbar.zoom()
        if mode == "zoom":
            for toolbar in self._plot_toolbars.values():
                toolbar.zoom()
        self._sync_selector_state()

    def _connect_main_canvas_mouse_events(self) -> None:
        for canvas in [self.image_canvas, self.corrected_canvas, self.fit_canvas, self.spectrum_canvas]:
            cid = canvas.mpl_connect("button_press_event", self._on_main_canvas_button_press)
            self._main_canvas_mouse_cids.append((canvas, cid))

    def _on_main_canvas_button_press(self, event) -> None:
        if event.button == 3 and self._mouse_mode == "zoom":
            self._reset_canvas_view(event.canvas)
            return

        if self._mouse_mode != "roi":
            return

        if event.canvas is not self.image_canvas:
            self._set_mouse_mode("zoom", warn=True)
            return

        if event.button != 3:
            return
        if self._current_mode_index() == 0 and self._point_inside_current_polygon(event.xdata, event.ydata):
            self._remove_map_roi_at_point(event.xdata, event.ydata)
        elif self._current_mode_index() == 1 and self._point_inside_current_spot_roi(event.xdata, event.ydata):
            self._clear_spot_roi()

    def _reset_canvas_view(self, canvas: FigureCanvas) -> None:
        if canvas is self.image_canvas:
            self._image_view_limits = None
        if canvas is self.corrected_canvas:
            self._corrected_view_limits = None
        if self._reset_snapshot_image_canvas_view(canvas):
            return
        toolbar = self._plot_toolbars.get(canvas)
        if toolbar is not None:
            toolbar.home()

    def _reset_all_canvas_views(self) -> None:
        self._image_view_limits = None
        self._corrected_view_limits = None
        for canvas, toolbar in self._plot_toolbars.items():
            if not self._reset_snapshot_image_canvas_view(canvas):
                toolbar.home()

    def _reset_snapshot_image_canvas_view(self, canvas: FigureCanvas) -> bool:
        if self._current_mode_index() != 1 or canvas not in {self.image_canvas, self.corrected_canvas}:
            return False
        if not canvas.figure.axes:
            return False
        axis = canvas.figure.axes[0]
        if not axis.images:
            return False
        image_shape = axis.images[0].get_array().shape
        if len(image_shape) < 2:
            return False
        height, width = int(image_shape[0]), int(image_shape[1])
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        if canvas is self.image_canvas:
            self._image_view_limits = (axis.get_xlim(), axis.get_ylim())
        else:
            self._corrected_view_limits = (axis.get_xlim(), axis.get_ylim())
        canvas.draw_idle()
        return True

    def _point_inside_current_polygon(self, x: Optional[float], y: Optional[float]) -> bool:
        if x is None or y is None:
            return False
        point = (float(x), float(y))
        return any(
            len(vertices) >= 3 and MplPath(vertices).contains_point(point, radius=0.5)
            for vertices in self.map_polygon_vertices
        )

    def _point_inside_current_spot_roi(self, x: Optional[float], y: Optional[float]) -> bool:
        if x is None or y is None or self.loaded is None or self._current_mode_index() != 1:
            return False
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return False
        y0, y1 = sorted((self.stack_y0.value(), self.stack_y1.value()))
        return 0 <= float(x) <= data.shape[2] and y0 <= float(y) <= y1

    def _clear_map_roi(self) -> None:
        if self._current_mode_index() != 0:
            return
        self.map_polygon_vertices = []
        self._reset_map_histogram_controls()
        self._draw_initial_image()
        self._update_clear_roi_button_enabled()

    def _remove_map_roi_at_point(self, x: Optional[float], y: Optional[float]) -> None:
        if x is None or y is None:
            return
        point = (float(x), float(y))
        for index in range(len(self.map_polygon_vertices) - 1, -1, -1):
            vertices = self.map_polygon_vertices[index]
            if len(vertices) >= 3 and MplPath(vertices).contains_point(point, radius=0.5):
                del self.map_polygon_vertices[index]
                self._reset_map_histogram_controls(preserve_mask_range=True)
                self._draw_initial_image()
                self._update_clear_roi_button_enabled()
                return

    def _clear_current_roi(self) -> None:
        if self._current_mode_index() == 0:
            self._clear_map_roi()
        else:
            self._clear_spot_roi()

    def _clear_spot_roi(self) -> None:
        if self.loaded is None or self._current_mode_index() != 1:
            return
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return
        self.stack_y0.blockSignals(True)
        self.stack_y1.blockSignals(True)
        self.stack_y0.setValue(0)
        self.stack_y1.setValue(data.shape[1])
        self.stack_y0.blockSignals(False)
        self.stack_y1.blockSignals(False)
        self._update_selection_overlay()
        self._update_clear_roi_button_enabled()

    def _spot_roi_is_full_detector(self) -> bool:
        if self.loaded is None or self._current_mode_index() != 1:
            return True
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return True
        return self.stack_y0.value() <= 0 and self.stack_y1.value() >= data.shape[1]

    def _update_clear_roi_button_enabled(self) -> None:
        if hasattr(self, "clear_roi_button"):
            if self._current_mode_index() == 0:
                self.clear_roi_button.setEnabled(bool(self.map_polygon_vertices))
            else:
                self.clear_roi_button.setEnabled(not self._spot_roi_is_full_detector())

    def _coerce_polygon_list(self, raw_vertices) -> list[list[tuple[float, float]]]:
        if not raw_vertices:
            return []
        first = raw_vertices[0]
        if isinstance(first, (list, tuple)) and len(first) == 2 and all(np.isscalar(value) for value in first):
            return [[(float(x), float(y)) for x, y in raw_vertices]]
        polygons: list[list[tuple[float, float]]] = []
        for polygon in raw_vertices:
            vertices = [(float(x), float(y)) for x, y in polygon]
            if len(vertices) >= 3:
                polygons.append(vertices)
        return polygons

    def _entry_polygon_list(self, entry: SavedMapEntry) -> list[list[tuple[float, float]]]:
        return self._coerce_polygon_list(entry.polygon_vertices)

    def _prepare_image_axis(self, axis):
        axis.grid(False)
        return axis

    def _capture_current_image_view(self) -> None:
        if self.loaded is None or self._current_mode_index() != 1 or not self.image_canvas.figure.axes:
            return
        axis = self.image_canvas.figure.axes[0]
        if not axis.images:
            return
        self._image_view_limits = (axis.get_xlim(), axis.get_ylim())

    def _capture_current_corrected_view(self) -> None:
        if self.loaded is None or self._current_mode_index() != 1 or not self.corrected_canvas.figure.axes:
            return
        axis = self.corrected_canvas.figure.axes[0]
        if not axis.images:
            return
        self._corrected_view_limits = (axis.get_xlim(), axis.get_ylim())

    def _on_image_view_changed(self, image_ax):
        if image_ax.images and self._current_mode_index() == 1:
            self._image_view_limits = (image_ax.get_xlim(), image_ax.get_ylim())

    def _set_empty_axis_message(self, axis, message: str) -> None:
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
            fontsize=max(8, self._current_title_font_size()),
        )
        axis.set_axis_off()

    def _current_title_font_size(self) -> int:
        return int(self.title_font_size_combo.currentData()) if hasattr(self, "title_font_size_combo") else 10

    def _apply_title_font_size_to_canvas(self, canvas) -> None:
        size = self._current_title_font_size()
        for axis in canvas.figure.axes:
            axis.title.set_fontsize(size)
        canvas.draw_idle()

    def _on_title_font_size_changed(self, _index: int):
        size = self._current_title_font_size()
        rcParams["axes.titlesize"] = size
        self.settings.setValue(f"{self.SETTINGS_GROUP}/{self.TITLE_FONT_SIZE_KEY}", size)
        for canvas_name in [
            "image_canvas",
            "corrected_canvas",
            "fit_canvas",
            "spectrum_canvas",
            "reference_image_canvas",
            "map_hist_canvas",
            "snapshot_hist_canvas",
        ]:
            canvas = getattr(self, canvas_name, None)
            if canvas is not None:
                self._apply_title_font_size_to_canvas(canvas)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtOrientation.Horizontal)
        outer.addWidget(splitter)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.mouse_mode_group = QtWidgets.QButtonGroup(self)
        self.mouse_mode_group.setExclusive(True)
        self.zoom_button = QtWidgets.QPushButton("Zoom")
        self.roi_button = QtWidgets.QPushButton("ROI")
        self.zoom_out_button = self._make_button("Zoom Out", self._reset_all_canvas_views)
        self.clear_roi_button = self._make_button("Clear ROI", self._clear_current_roi)
        action_button_height = self.zoom_out_button.sizeHint().height()
        for button in [self.zoom_button, self.roi_button]:
            button.setCheckable(True)
            button.setFixedHeight(action_button_height)
            button.setMinimumWidth(76)
            button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.zoom_button.setChecked(True)
        self.mouse_mode_group.addButton(self.zoom_button)
        self.mouse_mode_group.addButton(self.roi_button)
        self.zoom_button.clicked.connect(lambda: self._set_mouse_mode("zoom"))
        self.roi_button.clicked.connect(lambda: self._set_mouse_mode("roi"))
        self._style_mouse_mode_buttons()

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setContentsMargins(12, 8, 0, 0)
        mode_row.setSpacing(0)
        mode_row.addWidget(self.zoom_button)
        mode_row.addWidget(self.roi_button)
        mode_row.addSpacing(8)
        mode_row.addWidget(self.zoom_out_button)
        mode_row.addWidget(self.clear_roi_button)
        mode_row.addStretch(1)
        left_layout.addLayout(mode_row)

        top_row = QtWidgets.QHBoxLayout()

        image_panel = QtWidgets.QVBoxLayout()
        self.image_canvas = PlotCanvas((1, 1), self)
        self.image_toolbar = self._make_plot_toolbar(self.image_canvas)
        image_panel.addWidget(self.image_canvas, 1)

        corrected_panel = QtWidgets.QVBoxLayout()
        self.corrected_canvas = PlotCanvas((1, 1), self)
        self.corrected_toolbar = self._make_plot_toolbar(self.corrected_canvas)
        corrected_panel.addWidget(self.corrected_canvas, 1)

        fit_panel = QtWidgets.QVBoxLayout()
        self.fit_canvas = PlotCanvas((1, 1), self)
        self.fit_toolbar = self._make_plot_toolbar(self.fit_canvas)
        fit_panel.addWidget(self.fit_canvas, 1)

        top_row.addLayout(image_panel, 1)
        top_row.addLayout(corrected_panel, 1)
        top_row.addLayout(fit_panel, 1)
        left_layout.addLayout(top_row, 1)

        self.spectrum_canvas = PlotCanvas((1, 1), self)
        self.spectrum_toolbar = self._make_plot_toolbar(self.spectrum_canvas)
        left_layout.addWidget(self.spectrum_canvas, 1)
        self._connect_main_canvas_mouse_events()
        self._set_mouse_mode("zoom")
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs, 1)
        self.progress_label = QtWidgets.QLabel("Idle")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout = QtWidgets.QVBoxLayout()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        right_layout.addLayout(progress_layout)
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(160)
        right_layout.addWidget(self.status_box)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)

        self._build_data_tab()
        self._build_plot_tab()
        self._build_map_tab()
        self._build_spot_tab()
        self._build_calibration_tab()
        self._build_saved_maps_tab()
        self._build_image_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_data_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.eels_path_label = QtWidgets.QLabel("No EELS file loaded")
        self.eels_path_label.setWordWrap(True)

        load_eels = self._make_button("Load EELS DM3/DM4", self._load_eels, variant="success")
        refresh = self._make_button("Process Current Settings", self._process_current, variant="danger")
        save_state = self._make_button("Save status", self._save_session_state)
        load_state = self._make_button("Restore status", self._load_selected_session_state)

        self.shape_label = QtWidgets.QLabel("-")

        layout.addRow(load_eels)
        layout.addRow("EELS file", self.eels_path_label)
        layout.addRow("Loaded shape", self.shape_label)
        action_buttons = QtWidgets.QHBoxLayout()
        action_buttons.addWidget(refresh, 1)
        layout.addRow(action_buttons)
        state_buttons = QtWidgets.QHBoxLayout()
        state_buttons.addWidget(save_state, 1)
        state_buttons.addWidget(load_state, 1)
        layout.addRow("Saved states", state_buttons)

        self.saved_state_table = QtWidgets.QTableWidget(0, 3)
        self.saved_state_table.setHorizontalHeaderLabels(["Folder", "Timestamp", "Comment"])
        state_header = self.saved_state_table.horizontalHeader()
        state_header.setSectionResizeMode(0, HeaderResizeMode.ResizeToContents)
        state_header.setSectionResizeMode(1, HeaderResizeMode.ResizeToContents)
        state_header.setSectionResizeMode(2, HeaderResizeMode.Stretch)
        self.saved_state_table.verticalHeader().setVisible(False)
        self.saved_state_table.setSelectionBehavior(SelectionBehavior.SelectRows)
        self.saved_state_table.setSelectionMode(SelectionMode.SingleSelection)
        self.saved_state_table.setAlternatingRowColors(True)
        self.saved_state_table.itemChanged.connect(self._on_saved_state_item_changed)
        self.saved_state_table.setMinimumHeight(180)
        layout.addRow(self.saved_state_table)

        self.tabs.addTab(tab, "Data")

    def _build_image_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QFormLayout()
        load_image = self._make_button("Load Reference Image", self._load_image, variant="success")
        self.image_path_label = QtWidgets.QLabel("Optional image not loaded")
        self.image_path_label.setWordWrap(True)

        self.reference_image_slider = RangeSlider(QtOrientation.Horizontal)
        self.reference_image_slider.setRange(0, 1000)
        self.reference_image_slider.valueChanged.connect(self._update_reference_image_preview)
        self.reference_image_min_label = QtWidgets.QLabel("Z min: -")
        self.reference_image_max_label = QtWidgets.QLabel("Z max: -")
        self._reference_image_min = 0.0
        self._reference_image_max = 1.0

        reference_labels = QtWidgets.QHBoxLayout()
        reference_labels.addWidget(self.reference_image_min_label)
        reference_labels.addStretch(1)
        reference_labels.addWidget(self.reference_image_max_label)

        controls.addRow(load_image)
        controls.addRow("Reference image", self.image_path_label)
        controls.addRow(reference_labels)
        controls.addRow(self.reference_image_slider)
        layout.addLayout(controls)

        self.reference_image_canvas = PlotCanvas((1, 1), self)
        self.reference_image_toolbar = NavigationToolbar(self.reference_image_canvas, self)
        layout.addWidget(self.reference_image_toolbar)
        layout.addWidget(self.reference_image_canvas, 1)

        self.tabs.addTab(tab, "Image")

    def _build_plot_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.normalize_saved_spectra_checkbox = QtWidgets.QCheckBox("Normalize spectra intensity")
        self.normalize_saved_spectra_checkbox.toggled.connect(self._update_view_for_active_tab)
        export_button = self._make_button("Export NPY", self._save_results)
        self.title_font_size_combo = QtWidgets.QComboBox()
        for size in range(8, 31, 2):
            self.title_font_size_combo.addItem(str(size), size)
        current_size = self.settings.value(
            f"{self.SETTINGS_GROUP}/{self.TITLE_FONT_SIZE_KEY}",
            10,
            type=int,
        )
        current_index = self.title_font_size_combo.findData(int(current_size))
        if current_index < 0:
            current_index = self.title_font_size_combo.findData(10)
        self.title_font_size_combo.setCurrentIndex(current_index)
        self.title_font_size_combo.currentIndexChanged.connect(self._on_title_font_size_changed)
        set_spinbox_max_width_fraction(self.title_font_size_combo, 0.25)

        layout.addRow(self.normalize_saved_spectra_checkbox)
        layout.addRow(export_button)
        layout.addRow("Title font size", self.title_font_size_combo)

        self.tabs.addTab(tab, "Plot")

    def _build_map_tab(self):
        self.energy_start_spin = QtWidgets.QSpinBox()
        self.energy_stop_spin = QtWidgets.QSpinBox()
        self.energy_start_spin.setMaximum(100000)
        self.energy_stop_spin.setMaximum(100000)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setMaximum(1e12)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(18000.0)
        self.threshold_spin.hide()

        for widget in [
            self.energy_start_spin,
            self.energy_stop_spin,
        ]:
            set_spinbox_max_width_fraction(widget)

        self.map_mask_slider = RangeSlider(QtOrientation.Horizontal)
        self.map_mask_slider.setRange(0, 1000)
        self.map_mask_slider.valueChanged.connect(self._on_map_mask_slider_changed)
        self.map_mask_min_label = QtWidgets.QLabel("Mask min: -")
        self.map_mask_max_label = QtWidgets.QLabel("Mask max: -")
        self.map_hist_log_checkbox = QtWidgets.QCheckBox("Log y")
        self.map_hist_log_checkbox.setChecked(True)
        self.map_hist_log_checkbox.toggled.connect(self._update_map_histogram)
        self.map_hist_canvas = PlotCanvas((1, 1), self)
        self.map_hist_canvas.setMinimumHeight(126)
        self.map_hist_canvas.mpl_connect("button_press_event", self._on_map_histogram_press)
        self.map_hist_canvas.mpl_connect("motion_notify_event", self._on_map_histogram_move)
        self.map_hist_canvas.mpl_connect("button_release_event", self._on_map_histogram_release)

        process_button = self._make_button("Apply ROI / Alignment", self._process_current, variant="danger")

        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.map_mode_widgets = [
            self.energy_start_spin,
            self.energy_stop_spin,
            self.map_mask_slider,
            self.map_hist_canvas,
            self.map_mask_min_label,
            self.map_mask_max_label,
            self.map_hist_log_checkbox,
        ]

        map_grid = QtWidgets.QGridLayout()
        map_grid.addWidget(QtWidgets.QLabel("Map-z pixel start"), 0, 0)
        map_grid.addWidget(self.energy_start_spin, 0, 1)
        map_grid.addWidget(QtWidgets.QLabel("Map-z pixel stop"), 0, 2)
        map_grid.addWidget(self.energy_stop_spin, 0, 3)
        map_grid.setColumnStretch(0, 0)
        map_grid.setColumnStretch(1, 1)
        map_grid.setColumnStretch(2, 0)
        map_grid.setColumnStretch(3, 1)
        mask_label_row = QtWidgets.QHBoxLayout()
        mask_label_row.addWidget(self.map_mask_min_label)
        mask_label_row.addStretch(1)
        mask_label_row.addWidget(self.map_hist_log_checkbox)
        mask_label_row.addStretch(1)
        mask_label_row.addWidget(self.map_mask_max_label)
        layout.addRow(map_grid)
        layout.addRow(mask_label_row)
        layout.addRow(self.map_mask_slider)
        layout.addRow(self.map_hist_canvas)
        layout.addRow(process_button)

        self.map_tab_index = self.tabs.addTab(tab, "2D Map")

    def _build_spot_tab(self):
        self.stack_y0 = QtWidgets.QSpinBox()
        self.stack_y1 = QtWidgets.QSpinBox()
        self.frame_start = QtWidgets.QSpinBox()
        self.frame_stop = QtWidgets.QSpinBox()
        self.snapshot_index_spin = QtWidgets.QSpinBox()
        self.snapshot_index_spin.setMaximum(0)
        self.snapshot_index_spin.valueChanged.connect(self._on_snapshot_index_changed)
        self.snapshot_prev_button = self._make_button("Previous", self._show_previous_snapshot)
        self.snapshot_next_button = self._make_button("Next", self._show_next_snapshot)
        for widget in [self.stack_y0, self.stack_y1, self.frame_start, self.frame_stop]:
            widget.setMaximum(100000)
        for widget in [self.stack_y0, self.stack_y1, self.frame_start, self.frame_stop, self.snapshot_index_spin]:
            set_spinbox_max_width_fraction(widget)
        self.stack_y0.valueChanged.connect(self._update_selection_overlay)
        self.stack_y1.valueChanged.connect(self._update_selection_overlay)
        self.frame_start.valueChanged.connect(self._draw_initial_image)
        self.frame_stop.valueChanged.connect(self._draw_initial_image)
        self.snapshot_z_slider = RangeSlider(QtOrientation.Horizontal)
        self.snapshot_z_slider.setRange(0, 1000)
        self.snapshot_z_slider.valueChanged.connect(self._on_snapshot_z_slider_changed)
        self.snapshot_z_min_label = QtWidgets.QLabel("Z min: -")
        self.snapshot_z_max_label = QtWidgets.QLabel("Z max: -")
        self.snapshot_hist_log_checkbox = QtWidgets.QCheckBox("Log y")
        self.snapshot_hist_log_checkbox.setChecked(True)
        self.snapshot_hist_log_checkbox.toggled.connect(self._update_snapshot_histogram)
        self.snapshot_hist_canvas = PlotCanvas((1, 1), self)
        self.snapshot_hist_canvas.setMinimumHeight(126)
        self.snapshot_hist_canvas.mpl_connect("button_press_event", self._on_snapshot_histogram_press)
        self.snapshot_hist_canvas.mpl_connect("motion_notify_event", self._on_snapshot_histogram_move)
        self.snapshot_hist_canvas.mpl_connect("button_release_event", self._on_snapshot_histogram_release)

        process_button = self._make_button("Apply ROI / Alignment", self._process_current, variant="danger")

        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.snapshot_mode_widgets = [
            self.stack_y0,
            self.stack_y1,
            self.frame_start,
            self.frame_stop,
            self.snapshot_index_spin,
            self.snapshot_prev_button,
            self.snapshot_next_button,
            self.snapshot_z_slider,
            self.snapshot_hist_canvas,
            self.snapshot_z_min_label,
            self.snapshot_z_max_label,
            self.snapshot_hist_log_checkbox,
        ]

        snapshot_nav_row = QtWidgets.QHBoxLayout()
        snapshot_nav_row.addWidget(self.snapshot_index_spin, 1)
        snapshot_nav_row.addWidget(self.snapshot_prev_button, 1)
        snapshot_nav_row.addWidget(self.snapshot_next_button, 1)
        range_grid = QtWidgets.QGridLayout()
        range_grid.addWidget(QtWidgets.QLabel("Detector y start"), 0, 0)
        range_grid.addWidget(self.stack_y0, 0, 1)
        range_grid.addWidget(QtWidgets.QLabel("Detector y stop"), 0, 2)
        range_grid.addWidget(self.stack_y1, 0, 3)
        range_grid.addWidget(QtWidgets.QLabel("Snapshot start"), 1, 0)
        range_grid.addWidget(self.frame_start, 1, 1)
        range_grid.addWidget(QtWidgets.QLabel("Snapshot stop"), 1, 2)
        range_grid.addWidget(self.frame_stop, 1, 3)
        range_grid.setColumnStretch(0, 0)
        range_grid.setColumnStretch(1, 1)
        range_grid.setColumnStretch(2, 0)
        range_grid.setColumnStretch(3, 1)
        snapshot_hist_labels = QtWidgets.QHBoxLayout()
        snapshot_hist_labels.addWidget(self.snapshot_z_min_label)
        snapshot_hist_labels.addStretch(1)
        snapshot_hist_labels.addWidget(self.snapshot_hist_log_checkbox)
        snapshot_hist_labels.addStretch(1)
        snapshot_hist_labels.addWidget(self.snapshot_z_max_label)
        layout.addRow("Snapshot index", snapshot_nav_row)
        layout.addRow(range_grid)
        layout.addRow(snapshot_hist_labels)
        layout.addRow(self.snapshot_z_slider)
        layout.addRow(self.snapshot_hist_canvas)
        layout.addRow(process_button)

        self.spot_tab_index = self.tabs.addTab(tab, "1D Spot")

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
            set_spinbox_max_width_fraction(widget)

        self.zlp_center_label = QtWidgets.QLabel("-")
        self.pixel_count_label = QtWidgets.QLabel("-")
        self.axis_span_label = QtWidgets.QLabel("-")
        process_button = self._make_button("Run Zero-Loss Calibration", self._process_current, variant="success")

        calibration_grid = QtWidgets.QGridLayout()
        calibration_grid.addWidget(QtWidgets.QLabel("Fit start (eV)"), 0, 0)
        calibration_grid.addWidget(self.fit_start_spin, 0, 1)
        calibration_grid.addWidget(QtWidgets.QLabel("Fit stop (eV)"), 0, 2)
        calibration_grid.addWidget(self.fit_stop_spin, 0, 3)
        calibration_grid.addWidget(QtWidgets.QLabel("Guess start (eV)"), 1, 0)
        calibration_grid.addWidget(self.guess_start_spin, 1, 1)
        calibration_grid.addWidget(QtWidgets.QLabel("Guess stop (eV)"), 1, 2)
        calibration_grid.addWidget(self.guess_stop_spin, 1, 3)
        calibration_grid.setColumnStretch(0, 0)
        calibration_grid.setColumnStretch(1, 1)
        calibration_grid.setColumnStretch(2, 0)
        calibration_grid.setColumnStretch(3, 1)
        layout.addRow(calibration_grid)
        layout.addRow("Zero-loss center", self.zlp_center_label)
        layout.addRow("Selected spectra", self.pixel_count_label)
        layout.addRow("Calibrated axis", self.axis_span_label)
        layout.addRow(process_button)

        self.tabs.addTab(tab, "Calibration")

    def _build_saved_maps_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QHBoxLayout()
        self.saved_update_plot_button = self._make_button("Update plot", self._update_view_for_active_tab)
        self.saved_add_current_button = self._make_button("Add current", self._add_current_entry)
        self.saved_remove_selected_button = self._make_button("Remove selected", self._remove_selected_saved_entries)
        controls.addWidget(self.saved_update_plot_button)
        controls.addWidget(self.saved_add_current_button)
        controls.addWidget(self.saved_remove_selected_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.saved_map_table = QtWidgets.QTableWidget(0, 4)
        self.saved_map_table.setHorizontalHeaderLabels(["👁", "🔒", "Comment", "ROI coordinates"])
        header = self.saved_map_table.horizontalHeader()
        header.setSectionResizeMode(0, HeaderResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, HeaderResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, HeaderResizeMode.Stretch)
        header.setSectionResizeMode(3, HeaderResizeMode.Stretch)
        self.saved_map_table.verticalHeader().setVisible(False)
        self.saved_map_table.setSelectionBehavior(SelectionBehavior.SelectRows)
        self.saved_map_table.setSelectionMode(SelectionMode.ExtendedSelection)
        self.saved_map_table.setAlternatingRowColors(True)
        self.saved_map_table.itemChanged.connect(self._on_saved_map_item_changed)
        self._set_saved_map_header_icons()
        layout.addWidget(self.saved_map_table, 1)

        self.tabs.addTab(tab, "Saved")

    def _set_saved_map_header_icons(self):
        show_header = self.saved_map_table.horizontalHeaderItem(0)
        lock_header = self.saved_map_table.horizontalHeaderItem(1)
        show_header.setText("👁")
        lock_header.setText("🔒")

    def _apply_default_ranges(self):
        self.energy_start_spin.setValue(150)
        self.energy_stop_spin.setValue(550)
        self.stack_y0.setValue(220)
        self.stack_y1.setValue(320)
        self._update_snapshot_navigation_enabled()
        self._update_mode_specific_ui()

    def _log(self, message: str):
        self.status_box.appendPlainText(message)

    def _set_progress(self, current: int, total: int, message: str):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        percent = int(round((current / total) * 100))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _on_processing_progress(self, current: int, total: int, message: str):
        self._set_progress(current, total, message)
        if message != self._last_progress_message:
            self._last_progress_message = message
            self._log(message)

    def _cleanup_processing_worker(self):
        thread = self.processing_thread
        self.processing_thread = None
        self.processing_worker = None
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()

    def _on_processing_finished(self, result):
        self.current_result = result
        self._update_view_for_active_tab()
        self._set_progress(100, 100, "Processing complete")
        self._last_progress_message = None
        self._log("Processing complete.")
        self._cleanup_processing_worker()

    def _on_processing_failed(self, message: str):
        self._set_progress(0, 100, "Processing failed")
        self._last_progress_message = None
        self._log(f"Processing failed: {message}")
        self._cleanup_processing_worker()

    def _start_processing_worker(self, *, mode_index: int, map_kwargs=None, snapshot_kwargs=None):
        if self.processing_thread is not None:
            self._log("Processing is already running.")
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting processing")
        self._last_progress_message = None
        self._log("Processing started.")

        self.processing_thread = QtCore.QThread(self)
        self.processing_worker = ProcessingWorker(
            mode_index=mode_index,
            signal=self.loaded.eels_signal,
            map_kwargs=map_kwargs,
            snapshot_kwargs=snapshot_kwargs,
        )
        self.processing_worker.moveToThread(self.processing_thread)
        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.progress.connect(self._on_processing_progress)
        self.processing_worker.finished.connect(self._on_processing_finished)
        self.processing_worker.failed.connect(self._on_processing_failed)
        self.processing_worker.finished.connect(self.processing_thread.quit)
        self.processing_worker.failed.connect(self.processing_thread.quit)
        self.processing_thread.finished.connect(self.processing_worker.deleteLater)
        self.processing_thread.start()

    def _make_checkbox_table_item(self, checked: bool) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtItemFlag.ItemIsEnabled | QtItemFlag.ItemIsSelectable | QtItemFlag.ItemIsUserCheckable)
        item.setCheckState(QtCheckState.Checked if checked else QtCheckState.Unchecked)
        return item

    def _current_map_roi_text(self) -> str:
        if self.map_polygon_vertices:
            return " | ".join(
                f"ROI {index + 1}: " + "; ".join(f"({x:.1f}, {y:.1f})" for x, y in vertices)
                for index, vertices in enumerate(self.map_polygon_vertices)
            )
        return "Full map"

    def _current_spot_roi_text(self) -> str:
        return (
            f"y={self.stack_y0.value()}:{self.stack_y1.value()}, "
            f"frame={self.frame_start.value()}:{self.frame_stop.value()}, "
            f"snapshot={self.snapshot_index_spin.value()}"
        )

    def _current_spot_roi_polygon(self, width: int) -> list[tuple[float, float]]:
        x_max = max(0.0, float(width - 1))
        y0 = float(self.stack_y0.value())
        y1 = float(self.stack_y1.value())
        return [(0.0, y0), (x_max, y0), (x_max, y1), (0.0, y1)]

    def _saved_entry_masked_image_for_display(self, entry: SavedMapEntry) -> np.ndarray:
        masked = np.asarray(entry.masked_image, dtype=float)
        display = np.asarray(entry.display_image, dtype=float)
        if masked.shape == display.shape:
            return masked
        if entry.mode != "spot" or display.ndim != 2:
            return masked

        full = np.full(display.shape, np.nan, dtype=float)
        selection_mask = np.asarray(entry.selection_mask, dtype=bool)
        if selection_mask.shape != display.shape:
            return full

        active_rows = np.where(selection_mask.any(axis=1))[0]
        if active_rows.size == 0:
            return full
        row_start = int(active_rows.min())
        row_stop = min(row_start + masked.shape[0], full.shape[0])
        col_stop = min(masked.shape[1], full.shape[1]) if masked.ndim == 2 else 0
        if masked.ndim == 2 and row_stop > row_start and col_stop > 0:
            full[row_start:row_stop, :col_stop] = masked[: row_stop - row_start, :col_stop]
        return full

    def _saved_map_entries_for_display(self) -> list[tuple[int, SavedMapEntry]]:
        return [
            (row, entry)
            for row, entry in enumerate(self.saved_map_entries)
            if entry.show
        ]

    def _saved_map_entry_color(self, row: int):
        return PLOT_CYCLE[row % len(PLOT_CYCLE)]

    def _saved_maps_tab_index(self) -> int:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "Saved":
                return index
        return -1

    def _saved_maps_tab_active(self) -> bool:
        return self.tabs.currentIndex() == self._saved_maps_tab_index()

    def _analysis_root_dir(self) -> Path:
        if self.loaded is not None and self.loaded.eels_path:
            return Path(self.loaded.eels_path).expanduser().resolve().parent / "vibeels-analysis"
        return Path.cwd() / "vibeels-analysis"

    def _state_metadata_path(self, directory: Path) -> Path:
        return directory / "state.json"

    def _state_arrays_path(self, directory: Path) -> Path:
        return directory / "state_arrays.npz"

    def _current_filename_label(self) -> str:
        if self.loaded is not None and self.loaded.eels_path:
            return Path(self.loaded.eels_path).name
        return "Spectrum"

    def _current_reference_image_data(self) -> Optional[np.ndarray]:
        if self.loaded is None or self.loaded.image_signal is None:
            return None
        data = np.asarray(self.loaded.image_signal.data)
        if data.ndim == 2:
            return np.asarray(data, dtype=float)
        if data.ndim > 2:
            squeezed = np.squeeze(data)
            if squeezed.ndim == 2:
                return np.asarray(squeezed, dtype=float)
        return None

    def _reference_image_for_map_overlay(self) -> Optional[np.ndarray]:
        if self.loaded is None or self._current_mode_index() != 0:
            return None
        reference_image = self._current_reference_image_data()
        if reference_image is None or reference_image.ndim != 2:
            return None
        map_image = self._map_intensity_image()
        if map_image is None or reference_image.shape != map_image.shape:
            return None
        return reference_image

    def _reset_reference_image_controls(self):
        image = self._current_reference_image_data()
        if image is None or image.size == 0:
            self._reference_image_min = 0.0
            self._reference_image_max = 1.0
        else:
            self._reference_image_min = float(np.nanmin(image))
            self._reference_image_max = float(np.nanmax(image))
        self.reference_image_slider.blockSignals(True)
        self.reference_image_slider.setValues(0, 1000)
        self.reference_image_slider.blockSignals(False)
        self._update_reference_image_preview()

    def _reference_image_range_values(self) -> tuple[float, float]:
        min_pos = self.reference_image_slider.lowerValue()
        max_pos = self.reference_image_slider.upperValue()
        span = self._reference_image_max - self._reference_image_min
        if span <= 0:
            return self._reference_image_min, self._reference_image_max
        z_min = self._reference_image_min + (span * min_pos / 1000.0)
        z_max = self._reference_image_min + (span * max_pos / 1000.0)
        return z_min, z_max

    def _update_reference_image_preview(self):
        ax = self.reference_image_canvas.reset_axis()
        image = self._current_reference_image_data()
        if image is None:
            self._set_empty_axis_message(ax, "Load a reference image to preview it here")
            self.reference_image_min_label.setText("Z min: -")
            self.reference_image_max_label.setText("Z max: -")
            self.reference_image_canvas.draw_idle()
            return
        z_min, z_max = self._reference_image_range_values()
        ax.imshow(image, cmap="gray", origin="upper", vmin=z_min, vmax=z_max)
        ax.set_title("Reference image")
        ax.set_xlabel("")
        ax.set_ylabel("")
        self.reference_image_min_label.setText(f"Z min: {z_min:.2f}")
        self.reference_image_max_label.setText(f"Z max: {z_max:.2f}")
        self.reference_image_canvas.draw_idle()

    def _refresh_saved_state_records(self):
        root = self._analysis_root_dir()
        records: list[SavedStateRecord] = []
        if root.exists():
            for child in sorted(
                root.iterdir(),
                key=lambda path: int(path.name) if path.name.isdigit() else -1,
                reverse=True,
            ):
                if not child.is_dir() or not child.name.isdigit():
                    continue
                metadata_path = self._state_metadata_path(child)
                if not metadata_path.exists():
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text())
                except Exception:
                    continue
                records.append(
                    SavedStateRecord(
                        folder_name=child.name,
                        saved_at=str(metadata.get("saved_at", "")),
                        comment=str(metadata.get("comment", "")),
                        directory=child,
                    )
                )
        self.saved_state_records = records
        self._refresh_saved_state_table()

    def _refresh_saved_state_table(self):
        self._updating_saved_state_table = True
        try:
            self.saved_state_table.setRowCount(len(self.saved_state_records))
            for row, record in enumerate(self.saved_state_records):
                folder_item = QtWidgets.QTableWidgetItem(record.folder_name)
                folder_item.setFlags(QtItemFlag.ItemIsEnabled | QtItemFlag.ItemIsSelectable)
                self.saved_state_table.setItem(row, 0, folder_item)

                timestamp_item = QtWidgets.QTableWidgetItem(record.saved_at)
                timestamp_item.setFlags(QtItemFlag.ItemIsEnabled | QtItemFlag.ItemIsSelectable)
                self.saved_state_table.setItem(row, 1, timestamp_item)

                comment_item = QtWidgets.QTableWidgetItem(record.comment)
                comment_item.setFlags(
                    QtItemFlag.ItemIsEnabled | QtItemFlag.ItemIsSelectable | QtItemFlag.ItemIsEditable
                )
                self.saved_state_table.setItem(row, 2, comment_item)
            self.saved_state_table.resizeRowsToContents()
        finally:
            self._updating_saved_state_table = False

    def _on_saved_state_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._updating_saved_state_table or item.column() != 2:
            return
        row = item.row()
        if row < 0 or row >= len(self.saved_state_records):
            return
        record = self.saved_state_records[row]
        metadata_path = self._state_metadata_path(record.directory)
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}
        metadata["comment"] = item.text()
        metadata_path.write_text(json.dumps(metadata, indent=2))
        record.comment = item.text()

    def _next_saved_state_dir(self) -> Path:
        root = self._analysis_root_dir()
        root.mkdir(parents=True, exist_ok=True)
        existing = {int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()}
        index = 0
        while index in existing:
            index += 1
        return root / str(index)

    def _saved_entry_zlp_scale(self, entry: SavedMapEntry) -> float:
        fit_mask = (entry.energy_axis >= entry.fit_window[0]) & (entry.energy_axis <= entry.fit_window[1])
        if not np.any(fit_mask):
            fit_mask = np.isfinite(entry.spectrum)
        zlp_region = np.asarray(entry.spectrum[fit_mask], dtype=float)
        if zlp_region.size == 0:
            return 1.0
        scale = float(np.max(np.abs(zlp_region)))
        return scale if scale > 0 else 1.0

    def _scaled_saved_entry_spectrum(self, entry: SavedMapEntry) -> np.ndarray:
        spectrum = np.asarray(entry.spectrum, dtype=float)
        if not self.normalize_saved_spectra_checkbox.isChecked():
            return spectrum
        return spectrum / self._saved_entry_zlp_scale(entry)

    def _curve_fwhm(self, x_values: np.ndarray, y_values: np.ndarray) -> float:
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        if x_values.size < 2 or y_values.size < 2:
            return 0.0
        peak = float(np.max(y_values))
        if peak <= 0:
            return 0.0
        half_max = peak / 2.0
        mask = y_values >= half_max
        if not np.any(mask):
            return 0.0
        return float(x_values[mask][-1] - x_values[mask][0])

    def _format_scientific(self, value: float, significant_digits: int = 3) -> str:
        precision = max(0, int(significant_digits) - 1)
        return f"{float(value):.{precision}e}"

    def _zlp_title(self, x_values: np.ndarray, y_values: np.ndarray, *, aligned: bool = False) -> str:
        prefix = "ZLP, aligned" if aligned else "ZLP"
        fwhm = self._curve_fwhm(x_values, y_values)
        return f"{prefix} (FWHM = {self._format_scientific(fwhm)} eV)"

    def _format_colorbar_scientific(self, colorbar):
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        colorbar.formatter = formatter
        colorbar.update_ticks()
        style_colorbar(colorbar)

    def _serialize_saved_map_entries(self) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        metadata_entries: list[dict[str, object]] = []
        arrays: dict[str, np.ndarray] = {}
        for index, entry in enumerate(self.saved_map_entries):
            prefix = f"saved_map_{index}"
            metadata_entries.append(
                {
                    "mode": entry.mode,
                    "show": entry.show,
                    "locked": entry.locked,
                    "comment": entry.comment,
                    "roi_text": entry.roi_text,
                    "polygon_vertices": entry.polygon_vertices,
                    "intensity_range": list(entry.intensity_range),
                    "fit_window": list(entry.fit_window),
                }
            )
            arrays[f"{prefix}_selection_mask"] = np.asarray(entry.selection_mask, dtype=bool)
            arrays[f"{prefix}_display_image"] = np.asarray(entry.display_image, dtype=float)
            arrays[f"{prefix}_masked_image"] = np.asarray(entry.masked_image, dtype=float)
            arrays[f"{prefix}_energy_axis_raw"] = np.asarray(entry.energy_axis_raw, dtype=float)
            arrays[f"{prefix}_selected_spectra"] = np.asarray(entry.selected_spectra, dtype=float)
            arrays[f"{prefix}_energy_axis"] = np.asarray(entry.energy_axis, dtype=float)
            arrays[f"{prefix}_spectrum"] = np.asarray(entry.spectrum, dtype=float)
        return metadata_entries, arrays

    def _deserialize_saved_map_entries(self, metadata_entries: list[dict[str, object]], arrays) -> list[SavedMapEntry]:
        entries: list[SavedMapEntry] = []
        for index, metadata in enumerate(metadata_entries):
            prefix = f"saved_map_{index}"
            entries.append(
                SavedMapEntry(
                    mode=str(metadata.get("mode", "map")),
                    show=bool(metadata.get("show", True)),
                    locked=bool(metadata.get("locked", False)),
                    comment=str(metadata.get("comment", "")),
                    roi_text=str(metadata.get("roi_text", "-")),
                    polygon_vertices=self._coerce_polygon_list(metadata.get("polygon_vertices", []))
                    if str(metadata.get("mode", "map")) == "map"
                    else [
                        (float(x), float(y))
                        for x, y in metadata.get("polygon_vertices", [])
                    ],
                    selection_mask=np.asarray(arrays[f"{prefix}_selection_mask"], dtype=bool),
                    display_image=np.asarray(arrays[f"{prefix}_display_image"], dtype=float),
                    masked_image=np.asarray(arrays[f"{prefix}_masked_image"], dtype=float),
                    intensity_range=tuple(float(v) for v in metadata.get("intensity_range", [0.0, 0.0])),
                    energy_axis_raw=np.asarray(arrays[f"{prefix}_energy_axis_raw"], dtype=float),
                    fit_window=tuple(float(v) for v in metadata.get("fit_window", [-0.05, 0.05])),
                    selected_spectra=np.asarray(arrays[f"{prefix}_selected_spectra"], dtype=float),
                    energy_axis=np.asarray(arrays[f"{prefix}_energy_axis"], dtype=float),
                    spectrum=np.asarray(arrays[f"{prefix}_spectrum"], dtype=float),
                )
            )
        return entries

    def _current_state_metadata(self) -> dict[str, object]:
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "comment": "",
            "mode_index": self._current_mode_index(),
            "eels_path": self.loaded.eels_path if self.loaded else "",
            "image_path": self.loaded.image_path if self.loaded else "",
            "energy_range": [self.energy_start_spin.value(), self.energy_stop_spin.value()],
            "map_mask_slider": [self.map_mask_slider.lowerValue(), self.map_mask_slider.upperValue()],
            "map_hist_log": self.map_hist_log_checkbox.isChecked(),
            "snapshot_z_slider": [self.snapshot_z_slider.lowerValue(), self.snapshot_z_slider.upperValue()],
            "snapshot_hist_log": self.snapshot_hist_log_checkbox.isChecked(),
            "snapshot_vertical_range": [self.stack_y0.value(), self.stack_y1.value()],
            "snapshot_frame_range": [self.frame_start.value(), self.frame_stop.value()],
            "snapshot_index": self.snapshot_index_spin.value(),
            "fit_window": [self.fit_start_spin.value(), self.fit_stop_spin.value()],
            "guess_window": [self.guess_start_spin.value(), self.guess_stop_spin.value()],
            "map_polygon_vertices": self.map_polygon_vertices,
            "normalize_saved_spectra": self.normalize_saved_spectra_checkbox.isChecked(),
            "title_font_size": int(self.title_font_size_combo.currentData()),
        }

    def _build_state_repro_script(self) -> str:
        return '''from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


state_dir = Path(__file__).resolve().parent
metadata = json.loads((state_dir / "state.json").read_text())
arrays = np.load(state_dir / "state_arrays.npz")

saved_maps = metadata.get("saved_map_entries", [])
shown = [(index, entry) for index, entry in enumerate(saved_maps) if entry.get("show", True)]

fig = plt.figure(figsize=(14, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

ax_roi = fig.add_subplot(gs[0, 0])
ax_mask = fig.add_subplot(gs[0, 1])
ax_zlp = fig.add_subplot(gs[0, 2])
ax_spec = fig.add_subplot(gs[1, :])

if shown:
    def curve_fwhm(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        if x_values.size < 2 or y_values.size < 2:
            return 0.0
        peak = float(np.max(y_values))
        if peak <= 0:
            return 0.0
        half_max = peak / 2.0
        mask = y_values >= half_max
        if not np.any(mask):
            return 0.0
        return float(x_values[mask][-1] - x_values[mask][0])

    def format_scientific(value, significant_digits=3):
        precision = max(0, int(significant_digits) - 1)
        return f"{float(value):.{precision}e}"

    first_prefix = f"saved_map_{shown[0][0]}"
    first_selected = np.asarray(arrays[f"{first_prefix}_selected_spectra"], dtype=float)
    if first_selected.ndim == 1:
        first_selected = first_selected[np.newaxis, :]
    base_image = np.asarray(arrays[f"{first_prefix}_display_image"], dtype=float)
    ax_roi.imshow(base_image, cmap="inferno", origin="upper")
    ax_roi.set_title("ROI")

    combined_mask = np.full_like(np.asarray(arrays[f"{first_prefix}_masked_image"], dtype=float), np.nan, dtype=float)
    for index, entry in shown:
        prefix = f"saved_map_{index}"
        color = plt.cm.tab10(index % 10)
        raw_vertices = entry.get("polygon_vertices", [])
        if raw_vertices and len(raw_vertices[0]) == 2 and not isinstance(raw_vertices[0][0], (list, tuple)):
            polygons = [raw_vertices]
        else:
            polygons = raw_vertices
        for vertices in polygons:
            vertices = [(float(x), float(y)) for x, y in vertices]
            if vertices:
                ax_roi.add_patch(Polygon(vertices, closed=True, fill=False, edgecolor=color, linewidth=1.8))
        masked = np.asarray(arrays[f"{prefix}_masked_image"], dtype=float)
        combined_mask = np.where(np.isfinite(masked), masked, combined_mask)

        energy_axis = np.asarray(arrays[f"{prefix}_energy_axis"], dtype=float)
        spectrum = np.asarray(arrays[f"{prefix}_spectrum"], dtype=float)
        ax_spec.plot(energy_axis, spectrum, color=color, linewidth=1.2, label=entry.get("comment") or f"Map {index + 1}")

        raw_axis = np.asarray(arrays[f"{prefix}_energy_axis_raw"], dtype=float)
        selected = np.asarray(arrays[f"{prefix}_selected_spectra"], dtype=float)
        fit_window = entry.get("fit_window", [-0.05, 0.05])
        fit_mask = (raw_axis >= fit_window[0]) & (raw_axis <= fit_window[1])
        for row_idx, row in enumerate(selected[:, fit_mask]):
            ax_zlp.plot(
                raw_axis[fit_mask],
                row,
                color=color,
                alpha=0.15,
                linewidth=0.7,
                label=(entry.get("comment") or f"Map {index + 1}") if row_idx == 0 else None,
            )

    ax_mask.imshow(np.ma.masked_invalid(combined_mask), cmap="inferno", origin="upper")
    ax_mask.set_title("Masked ROI")
    ax_zlp.axvline(0.0, color="#3b82f6", linestyle=":", linewidth=1.0, label="alignment target")
    ax_zlp.set_title(
        f"ZLP (FWHM = {format_scientific(curve_fwhm(arrays[f'{first_prefix}_energy_axis_raw'], np.sum(first_selected, axis=0)))} eV)"
    )
    ax_zlp.set_xlabel("Energy loss (eV)")
    ax_zlp.set_ylabel("ZLP region")
    ax_zlp.legend(loc="best", fontsize=8)
    ax_spec.set_title(Path(metadata.get("eels_path", "")).name or "Spectrum")
    ax_spec.set_xlabel("Energy loss (eV, ZLP corrected)")
    ax_spec.set_ylabel("Intensity (a.u.)")
    ax_spec.legend(loc="best", fontsize=8)
else:
    for ax, title, text in [
        (ax_roi, "ROI", "No saved entries selected for display"),
        (ax_mask, "Masked ROI", "No masked ROI preview"),
        (ax_zlp, "ZLP", "No ZLP preview"),
        (ax_spec, "Spectrum", "No saved spectra selected for display"),
    ]:
        ax.set_title(title)
        ax.text(0.5, 0.5, text, ha="center", va="center")
        ax.set_axis_off()

plt.show()
'''

    def _write_saved_state_xy_files(self, state_dir: Path):
        data_stem = Path(self.loaded.eels_path).stem if self.loaded is not None and self.loaded.eels_path else "current_spectrum"

        if self.current_result is not None:
            energy_axis_ev = np.asarray(self.current_result.energy_axis_calibrated, dtype=float)
            current_xy = np.column_stack(
                [
                    energy_axis_ev,
                    energy_axis_ev * EV_TO_CMINV,
                    np.asarray(self.current_result.summed_spectrum, dtype=float),
                ]
            )
            np.savetxt(state_dir / f"{data_stem}.vxy", current_xy, header="eV cm-1 y", comments="")

        for index, entry in enumerate(self.saved_map_entries):
            spectrum_xy = np.column_stack(
                [
                    np.asarray(entry.energy_axis, dtype=float),
                    np.asarray(entry.energy_axis, dtype=float) * EV_TO_CMINV,
                    np.asarray(entry.spectrum, dtype=float),
                ]
            )
            np.savetxt(state_dir / f"{data_stem}_map_{index}.vxy", spectrum_xy, header="eV cm-1 y", comments="")
        (state_dir / "reproduce_state.py").write_text(self._build_state_repro_script())

    def _save_session_state(self):
        state_dir = self._next_saved_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._current_state_metadata()
        saved_map_metadata, saved_map_arrays = self._serialize_saved_map_entries()
        metadata["saved_map_entries"] = saved_map_metadata

        self._state_metadata_path(state_dir).write_text(json.dumps(metadata, indent=2))
        np.savez(self._state_arrays_path(state_dir), **saved_map_arrays)
        self._write_saved_state_xy_files(state_dir)
        self._refresh_saved_state_records()

        for row, record in enumerate(self.saved_state_records):
            if record.directory == state_dir:
                self.saved_state_table.selectRow(row)
                break
        self._log(f"Saved session state to {state_dir}")

    def _load_eels_from_path(self, path: str) -> object:
        signal = load_eels_signal(path)
        image_signal = self.loaded.image_signal if self.loaded else None
        image_path = self.loaded.image_path if self.loaded else None
        self.loaded = LoadedData(signal, image_signal, path, image_path)
        self.current_result = None
        self.saved_map_entries = []
        self._image_view_limits = None
        self._corrected_view_limits = None
        self.map_polygon_vertices = []
        self.eels_path_label.setText(path)
        self.shape_label.setText(str(signal.data.shape))
        self.zlp_center_label.setText("-")
        self.pixel_count_label.setText("-")
        self.axis_span_label.setText("-")
        self._set_mode_from_signal(signal)
        self._reset_analysis_canvases(self._current_mode_index())
        self._sync_ranges_to_loaded_data()
        self._update_mode_specific_ui()
        self._refresh_saved_map_table()
        if self._current_mode_index() == 0:
            self._reset_map_histogram_controls()
        self._remember_data_dir(path)
        if image_signal is None:
            self._update_reference_image_preview()
        self._refresh_saved_state_records()
        self._update_snapshot_histogram()
        return signal

    def _load_image_from_path(self, path: str):
        image_signal = load_signal(path)
        if self.loaded is None:
            self.loaded = LoadedData(image_signal, image_signal, "", path)
        else:
            self.loaded.image_signal = image_signal
            self.loaded.image_path = path
        self.image_path_label.setText(path)
        self._remember_data_dir(path)
        self._reset_reference_image_controls()

    def _load_selected_session_state(self):
        selected_rows = self.saved_state_table.selectionModel().selectedRows()
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Select saved state", "Select a saved state to load.")
            return
        row = selected_rows[0].row()
        if row < 0 or row >= len(self.saved_state_records):
            return
        record = self.saved_state_records[row]
        metadata_path = self._state_metadata_path(record.directory)
        arrays_path = self._state_arrays_path(record.directory)
        try:
            metadata = json.loads(metadata_path.read_text())
            arrays = np.load(arrays_path, allow_pickle=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Failed to load state", f"Could not load saved state:\n\n{exc}")
            return

        try:
            eels_path = str(metadata.get("eels_path", ""))
            image_path = str(metadata.get("image_path", ""))
            if eels_path:
                self._load_eels_from_path(eels_path)
            if image_path and Path(image_path).exists():
                self._load_image_from_path(image_path)

            self._update_mode_specific_ui()

            energy_range = metadata.get("energy_range", [self.energy_start_spin.value(), self.energy_stop_spin.value()])
            self.energy_start_spin.setValue(int(energy_range[0]))
            self.energy_stop_spin.setValue(int(energy_range[1]))

            snapshot_vertical = metadata.get("snapshot_vertical_range", [self.stack_y0.value(), self.stack_y1.value()])
            self.stack_y0.setValue(int(snapshot_vertical[0]))
            self.stack_y1.setValue(int(snapshot_vertical[1]))

            snapshot_frames = metadata.get("snapshot_frame_range", [self.frame_start.value(), self.frame_stop.value()])
            self.frame_start.setValue(int(snapshot_frames[0]))
            self.frame_stop.setValue(int(snapshot_frames[1]))
            self.snapshot_index_spin.setValue(int(metadata.get("snapshot_index", self.snapshot_index_spin.value())))

            fit_window = metadata.get("fit_window", [self.fit_start_spin.value(), self.fit_stop_spin.value()])
            self.fit_start_spin.setValue(float(fit_window[0]))
            self.fit_stop_spin.setValue(float(fit_window[1]))
            guess_window = metadata.get("guess_window", [self.guess_start_spin.value(), self.guess_stop_spin.value()])
            self.guess_start_spin.setValue(float(guess_window[0]))
            self.guess_stop_spin.setValue(float(guess_window[1]))

            self.map_polygon_vertices = self._coerce_polygon_list(metadata.get("map_polygon_vertices", []))

            slider_values = metadata.get(
                "map_mask_slider",
                [self.map_mask_slider.lowerValue(), self.map_mask_slider.upperValue()],
            )
            self._draw_initial_image()
            self.map_mask_slider.setValues(int(slider_values[0]), int(slider_values[1]))
            self.map_hist_log_checkbox.setChecked(bool(metadata.get("map_hist_log", True)))
            snapshot_slider_values = metadata.get(
                "snapshot_z_slider",
                [self.snapshot_z_slider.lowerValue(), self.snapshot_z_slider.upperValue()],
            )
            self.snapshot_z_slider.setValues(int(snapshot_slider_values[0]), int(snapshot_slider_values[1]))
            self.snapshot_hist_log_checkbox.setChecked(bool(metadata.get("snapshot_hist_log", True)))

            self.normalize_saved_spectra_checkbox.setChecked(bool(metadata.get("normalize_saved_spectra", False)))
            title_font_size = int(metadata.get("title_font_size", self.title_font_size_combo.currentData()))
            title_index = self.title_font_size_combo.findData(title_font_size)
            if title_index >= 0:
                self.title_font_size_combo.setCurrentIndex(title_index)
            self.saved_map_entries = self._deserialize_saved_map_entries(
                list(metadata.get("saved_map_entries", [])),
                arrays,
            )
            self._refresh_saved_map_table()
            if self.loaded is not None and self.loaded.eels_path:
                self._process_current()
            else:
                self._update_view_for_active_tab()
            self._log(f"Loaded session state from {record.directory}")
        finally:
            arrays.close()

    def _refresh_saved_map_table(self):
        self._updating_saved_map_table = True
        try:
            self.saved_map_table.setRowCount(len(self.saved_map_entries))
            for row, entry in enumerate(self.saved_map_entries):
                self.saved_map_table.setItem(row, 0, self._make_checkbox_table_item(entry.show))
                self.saved_map_table.setItem(row, 1, self._make_checkbox_table_item(entry.locked))

                comment_item = QtWidgets.QTableWidgetItem(entry.comment)
                comment_flags = QtItemFlag.ItemIsEnabled | QtItemFlag.ItemIsSelectable
                if not entry.locked:
                    comment_flags |= QtItemFlag.ItemIsEditable
                comment_item.setFlags(comment_flags)
                self.saved_map_table.setItem(row, 2, comment_item)

                roi_item = QtWidgets.QTableWidgetItem(entry.roi_text)
                roi_item.setFlags(QtItemFlag.ItemIsEnabled | QtItemFlag.ItemIsSelectable)
                self.saved_map_table.setItem(row, 3, roi_item)
            self.saved_map_table.resizeRowsToContents()
        finally:
            self._updating_saved_map_table = False

    def _on_saved_map_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._updating_saved_map_table:
            return
        row = item.row()
        if row < 0 or row >= len(self.saved_map_entries):
            return
        entry = self.saved_map_entries[row]
        if item.column() == 0:
            entry.show = item.checkState() == QtCheckState.Checked
        elif item.column() == 1:
            entry.locked = item.checkState() == QtCheckState.Checked
            self._refresh_saved_map_table()
        elif item.column() == 2:
            entry.comment = item.text()
        self._update_view_for_active_tab()

    def _remove_selected_saved_entries(self):
        selected_rows = sorted(
            {index.row() for index in self.saved_map_table.selectionModel().selectedRows()},
            reverse=True,
        )
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Select saved entry", "Select one or more saved entries to remove.")
            return
        for row in selected_rows:
            if 0 <= row < len(self.saved_map_entries):
                del self.saved_map_entries[row]
        self._refresh_saved_map_table()
        self._update_view_for_active_tab()

    def _add_current_entry(self):
        if self.current_result is None:
            QtWidgets.QMessageBox.information(
                self,
                "No processed result",
                "Run processing before adding the current spectrum to Saved.",
            )
            return
        if isinstance(self.current_result, MapProcessingResult):
            entry = SavedMapEntry(
                mode="map",
                show=True,
                locked=False,
                comment=f"Map {len(self.saved_map_entries) + 1}",
                roi_text=self._current_map_roi_text(),
                polygon_vertices=[list(vertices) for vertices in self.map_polygon_vertices],
                selection_mask=np.asarray(self.current_result.selection_mask, dtype=bool).copy(),
                display_image=np.asarray(self.current_result.display_image, dtype=float).copy(),
                masked_image=np.asarray(self.current_result.masked_image, dtype=float).copy(),
                intensity_range=self._map_intensity_range_values(),
                energy_axis_raw=np.asarray(self.current_result.energy_axis_raw, dtype=float).copy(),
                fit_window=self._fit_window(),
                selected_spectra=np.asarray(self.current_result.selected_spectra, dtype=float).copy(),
                energy_axis=np.asarray(self.current_result.energy_axis_calibrated, dtype=float).copy(),
                spectrum=np.asarray(self.current_result.summed_spectrum, dtype=float).copy(),
            )
        elif isinstance(self.current_result, StackProcessingResult):
            frame_index = min(self.snapshot_index_spin.value(), self.loaded.eels_signal.data.shape[0] - 1)
            detector_image = np.asarray(self.loaded.eels_signal.data[frame_index], dtype=float)
            y0 = self.stack_y0.value()
            y1 = self.stack_y1.value()
            aligned_image = self._current_snapshot_aligned_image()
            if aligned_image is None:
                aligned_image = np.zeros((max(1, y1 - y0), detector_image.shape[1]), dtype=float)
            masked_image = np.full(detector_image.shape, np.nan, dtype=float)
            row_stop = min(y0 + aligned_image.shape[0], masked_image.shape[0])
            col_stop = min(aligned_image.shape[1], masked_image.shape[1])
            if row_stop > y0 and col_stop > 0:
                masked_image[y0:row_stop, :col_stop] = aligned_image[: row_stop - y0, :col_stop]
            entry = SavedMapEntry(
                mode="spot",
                show=True,
                locked=False,
                comment=f"Spot {len(self.saved_map_entries) + 1}",
                roi_text=self._current_spot_roi_text(),
                polygon_vertices=self._current_spot_roi_polygon(detector_image.shape[1]),
                selection_mask=np.zeros(detector_image.shape, dtype=bool),
                display_image=detector_image.copy(),
                masked_image=masked_image,
                intensity_range=(0.0, 0.0),
                energy_axis_raw=np.asarray(self.current_result.energy_axis_raw, dtype=float).copy(),
                fit_window=self._fit_window(),
                selected_spectra=np.asarray(aligned_image, dtype=float).copy(),
                energy_axis=np.asarray(self.current_result.energy_axis_calibrated, dtype=float).copy(),
                spectrum=np.asarray(self.current_result.summed_spectrum, dtype=float).copy(),
            )
            entry.selection_mask[y0:y1, :] = True
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Unsupported result",
                "The current result cannot be added to Saved.",
            )
            return
        self.saved_map_entries.append(entry)
        self._refresh_saved_map_table()
        self._log(f"Added saved spectrum #{len(self.saved_map_entries)}.")
        self._update_view_for_active_tab()

    def _update_saved_spectra_plot(self):
        spectrum_ax = self.spectrum_canvas.reset_axis()

        visible_entries = self._saved_map_entries_for_display()
        if not visible_entries:
            self._set_empty_axis_message(spectrum_ax, "No saved spectra selected for display")
            self.spectrum_canvas.draw_idle()
            return

        normalize = self.normalize_saved_spectra_checkbox.isChecked()
        for row, entry in visible_entries:
            spectrum = self._scaled_saved_entry_spectrum(entry)
            label = entry.comment.strip() or f"Map {row + 1}"
            linewidth = 1.8 if entry.locked else 1.0
            alpha = 1.0 if entry.locked else 0.85
            spectrum_ax.plot(
                entry.energy_axis,
                spectrum,
                linewidth=linewidth,
                alpha=alpha,
                color=self._saved_map_entry_color(row),
                label=label,
            )

        spectrum_ax.set_xlabel("Energy loss (eV, ZLP corrected)")
        spectrum_ax.set_ylabel("Normalized intensity" if normalize else "Intensity (a.u.)")
        spectrum_ax.set_title(self._current_filename_label())
        spectrum_ax_top = spectrum_ax.secondary_xaxis(
            "top",
            functions=(lambda x: x * EV_TO_CMINV, lambda x: x / EV_TO_CMINV),
        )
        spectrum_ax_top.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        style_secondary_axis(spectrum_ax_top)
        spectrum_ax.legend(loc="best", fontsize=8)
        self.spectrum_canvas.draw_idle()

    def _has_visible_saved_entries(self) -> bool:
        return bool(self._saved_map_entries_for_display())

    def _visible_saved_preview_entries(self) -> list[tuple[int, SavedMapEntry]]:
        entries = self._saved_map_entries_for_display()
        if not entries:
            return []
        preview_mode = entries[0][1].mode
        return [(row, entry) for row, entry in entries if entry.mode == preview_mode]

    def _clear_saved_maps_preview(self):
        image_ax = self._prepare_image_axis(self.image_canvas.reset_axis())
        self._set_empty_axis_message(image_ax, "No saved entries selected for display")
        self.image_canvas.draw_idle()

        corrected_ax = self._prepare_image_axis(self.corrected_canvas.reset_axis())
        self._set_empty_axis_message(corrected_ax, "No masked map preview")
        self.corrected_canvas.draw_idle()

        fit_ax = self.fit_canvas.reset_axis()
        self._set_empty_axis_message(fit_ax, "No ZLP alignment preview")
        self.fit_canvas.draw_idle()

        spectrum_ax = self.spectrum_canvas.reset_axis()
        self._set_empty_axis_message(spectrum_ax, "No saved spectra selected for display")
        self.spectrum_canvas.draw_idle()

    def _reset_analysis_canvases(self, mode_index: Optional[int] = None):
        if mode_index is None:
            mode_index = self._current_mode_index() if self.loaded is not None else 0

        self._detach_selector()
        self.selection_rect = None
        self.selection_polygons = []
        self._image_view_limits = None
        self._corrected_view_limits = None

        if mode_index == 0:
            image_message = "Map intensity preview will appear here"
            corrected_message = "Masked map image will appear here after processing"
        else:
            image_message = "Snapshot preview will appear here"
            corrected_message = "Aligned snapshot image will appear here after processing"

        image_ax = self._prepare_image_axis(self.image_canvas.reset_axis())
        self._set_empty_axis_message(image_ax, image_message)
        self.image_canvas.draw_idle()

        corrected_ax = self._prepare_image_axis(self.corrected_canvas.reset_axis())
        self._set_empty_axis_message(corrected_ax, corrected_message)
        self.corrected_canvas.draw_idle()

        fit_ax = self.fit_canvas.reset_axis()
        self._set_empty_axis_message(fit_ax, "Zero-loss fit preview will appear here after processing")
        self.fit_canvas.draw_idle()

        spectrum_ax = self.spectrum_canvas.reset_axis()
        self._set_empty_axis_message(spectrum_ax, "Summed spectrum will appear here after processing")
        self.spectrum_canvas.draw_idle()

    def _preview_selected_saved_maps(self):
        preview_entries = self._visible_saved_preview_entries()
        if not preview_entries:
            self._clear_saved_maps_preview()
            return
        preview_mode = preview_entries[0][1].mode

        self._detach_selector()
        image_ax = self._prepare_image_axis(self.image_canvas.reset_axis())
        base_image = np.asarray(preview_entries[0][1].display_image, dtype=float)
        image_ax.imshow(base_image, cmap="inferno" if preview_mode == "map" else "viridis", origin="upper", aspect="auto")
        image_ax.set_title("ROI")
        image_ax.set_xlabel("")
        image_ax.set_ylabel("")
        for row, entry in preview_entries:
            color = self._saved_map_entry_color(row)
            for vertices in self._entry_polygon_list(entry):
                image_ax.add_patch(
                    Polygon(
                        vertices,
                        closed=True,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.8,
                    )
                )
        self.image_canvas.draw_idle()

        corrected_ax = self._prepare_image_axis(self.corrected_canvas.reset_axis())
        base_masked = self._saved_entry_masked_image_for_display(preview_entries[0][1])
        combined_masked = np.full_like(base_masked, np.nan, dtype=float)
        for _, entry in preview_entries:
            entry_masked = self._saved_entry_masked_image_for_display(entry)
            if entry_masked.shape != combined_masked.shape:
                continue
            combined_masked = np.where(np.isfinite(entry_masked), entry_masked, combined_masked)
        masked = np.ma.masked_invalid(combined_masked)
        corrected_ax.imshow(masked, cmap="inferno" if preview_mode == "map" else "viridis", origin="upper", aspect="auto")
        corrected_ax.set_title("Masked ROI")
        corrected_ax.set_xlabel("")
        corrected_ax.set_ylabel("")
        self.corrected_canvas.draw_idle()

        fit_ax = self.fit_canvas.reset_axis()
        active_entry = preview_entries[0][1]
        active_fit_mask = (
            (active_entry.energy_axis_raw >= active_entry.fit_window[0])
            & (active_entry.energy_axis_raw <= active_entry.fit_window[1])
        )
        active_spectra_stack = np.asarray(active_entry.selected_spectra, dtype=float)
        if active_spectra_stack.ndim == 1:
            active_spectra_stack = active_spectra_stack[np.newaxis, :]
        active_zlp_profile = np.sum(active_spectra_stack[:, active_fit_mask], axis=0)
        spectra_label = "Intensity"
        for row, entry in preview_entries:
            fit_mask = (
                (entry.energy_axis_raw >= entry.fit_window[0])
                & (entry.energy_axis_raw <= entry.fit_window[1])
            )
            spectra_stack = np.asarray(entry.selected_spectra, dtype=float)
            if spectra_stack.ndim == 1:
                spectra_stack = spectra_stack[np.newaxis, :]
            for index, spectrum in enumerate(spectra_stack[:, fit_mask]):
                fit_ax.plot(
                    entry.energy_axis_raw[fit_mask],
                    spectrum,
                    color=self._saved_map_entry_color(row),
                    alpha=0.15,
                    linewidth=0.7,
                    label=(entry.comment.strip() or f"Map {row + 1}") if index == 0 else None,
                )
        fit_ax.axvline(
            0.0,
            color=PLOT_PREVIEW,
            linewidth=1.0,
            linestyle=":",
            label="alignment target" if len(preview_entries) == 1 else "_nolegend_",
        )
        fit_ax.set_title(
            self._zlp_title(active_entry.energy_axis_raw[active_fit_mask], active_zlp_profile)
        )
        fit_ax.set_xlabel("Energy loss (eV)")
        fit_ax.set_ylabel(spectra_label)
        fit_ax.legend(loc="best", fontsize=8)
        self.fit_canvas.draw_idle()

        self._update_saved_spectra_plot()

    def _update_view_for_active_tab(self):
        if self._has_visible_saved_entries():
            self._preview_selected_saved_maps()
            return
        if self.current_result is not None:
            self._render_result(self.current_result)
            return
        self._draw_initial_image()

    def _on_tab_changed(self, _index: int):
        self._update_view_for_active_tab()

    def _initial_data_dir(self) -> str:
        saved_dir = self.settings.value(f"{self.SETTINGS_GROUP}/{self.LAST_DATA_DIR_KEY}", "", type=str)
        if saved_dir and Path(saved_dir).is_dir():
            return saved_dir
        return str(Path.home())

    def _remember_data_dir(self, path: str | Path):
        selected_path = Path(path).expanduser()
        directory = selected_path if selected_path.is_dir() else selected_path.parent
        if not str(directory):
            return
        self.settings.setValue(f"{self.SETTINGS_GROUP}/{self.LAST_DATA_DIR_KEY}", str(directory))

    def _load_eels(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load EELS signal",
            self._initial_data_dir(),
            "DigitalMicrograph (*.dm3 *.dm4);;All files (*)",
        )
        if not path:
            return
        try:
            self._load_eels_from_path(path)
        except Exception as exc:
            self._log(f"Failed to load EELS file: {exc}")
            return
        self._update_view_for_active_tab()
        self._log(f"Loaded EELS signal: {path}")

    def _load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load reference image",
            self._initial_data_dir(),
            "DigitalMicrograph (*.dm3 *.dm4);;All files (*)",
        )
        if not path:
            return
        try:
            self._load_image_from_path(path)
        except Exception as exc:
            self._log(f"Failed to load image file: {exc}")
            return
        self._update_view_for_active_tab()
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

        if self._current_mode_index() != 0:
            self.stack_y0.setMaximum(shape[1] - 1)
            self.stack_y1.setMaximum(shape[1])
            self.stack_y1.setValue(min(max(self.stack_y1.value(), self.stack_y0.value() + 1), shape[1]))
            self.frame_start.setMaximum(shape[0] - 1)
            self.frame_stop.setMaximum(shape[0])
            self.frame_stop.setValue(shape[0])
            self.snapshot_index_spin.setMaximum(shape[0] - 1)
            self.snapshot_index_spin.setValue(min(self.snapshot_index_spin.value(), shape[0] - 1))
            self._reset_snapshot_histogram_controls()
        self._update_snapshot_navigation_enabled()

    def _detect_mode_index(self, signal) -> int:
        data = getattr(signal, "data", None)
        if data is None or np.asarray(data).ndim != 3:
            raise ValueError(
                "Unsupported EELS dataset: expected a 3D Signal1D map `(y, x, energy)` or "
                "3D Signal2D snapshot stack `(frame, y, energy)`. "
                f"Got {describe_signal_layout(signal)}."
            )
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

    def _current_mode_index(self) -> int:
        if self.loaded is None:
            return 0
        return self._detect_mode_index(self.loaded.eels_signal)

    def _set_mode_from_signal(self, signal):
        mode_index = self._detect_mode_index(signal)
        mode_label = "map workflow (Signal1D)" if mode_index == 0 else "snapshot workflow (Signal2D)"
        self._log(f"Detected {mode_label}.")

    def _update_mode_specific_ui(self):
        is_map_mode = self._current_mode_index() == 0
        for widget in self.map_mode_widgets:
            widget.setEnabled(is_map_mode)
        for widget in self.snapshot_mode_widgets:
            widget.setEnabled(not is_map_mode)
        self.tabs.setTabEnabled(self.map_tab_index, is_map_mode)
        self.tabs.setTabEnabled(self.spot_tab_index, not is_map_mode)
        if is_map_mode and self.tabs.currentIndex() == self.spot_tab_index:
            self.tabs.setCurrentIndex(self.map_tab_index)
        if not is_map_mode and self.tabs.currentIndex() == self.map_tab_index:
            self.tabs.setCurrentIndex(self.spot_tab_index)
        if is_map_mode:
            self.snapshot_index_spin.setEnabled(False)
            self.snapshot_prev_button.setEnabled(False)
            self.snapshot_next_button.setEnabled(False)
        else:
            self._update_snapshot_navigation_enabled()
        self._update_clear_roi_button_enabled()

    def _map_intensity_image(self) -> Optional[np.ndarray]:
        if self.loaded is None or self._current_mode_index() != 0:
            return None
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3:
            return None
        e0 = max(0, min(self.energy_start_spin.value(), data.shape[2] - 1))
        e1 = max(e0 + 1, min(self.energy_stop_spin.value(), data.shape[2]))
        return data[:, :, e0:e1].sum(axis=2)

    def _current_polygon_mask(self) -> Optional[np.ndarray]:
        image = self._map_intensity_image()
        if image is None:
            return None
        height, width = image.shape
        if not self.map_polygon_vertices:
            return np.ones((height, width), dtype=bool)
        yy, xx = np.mgrid[0:height, 0:width]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = np.zeros((height, width), dtype=bool)
        for vertices in self.map_polygon_vertices:
            if len(vertices) < 3:
                continue
            path = MplPath(vertices)
            mask |= path.contains_points(points, radius=0.5).reshape((height, width))
        return mask

    def _map_intensity_range_values(self) -> tuple[float, float]:
        min_pos = self.map_mask_slider.lowerValue()
        max_pos = self.map_mask_slider.upperValue()
        span = self._map_intensity_max - self._map_intensity_min
        if span <= 0:
            return self._map_intensity_min, self._map_intensity_max
        z_min = self._map_intensity_min + (span * min_pos / 1000.0)
        z_max = self._map_intensity_min + (span * max_pos / 1000.0)
        return z_min, z_max

    def _map_mask_slider_value_from_intensity(self, intensity: float) -> int:
        span = self._map_intensity_max - self._map_intensity_min
        if span <= 0:
            return 0
        clamped = min(max(float(intensity), self._map_intensity_min), self._map_intensity_max)
        return int(round((clamped - self._map_intensity_min) / span * 1000.0))

    def _set_map_mask_slider_values(self, z_min: float, z_max: float):
        span = self._map_intensity_max - self._map_intensity_min
        if span <= 0:
            min_pos = 0
            max_pos = 1000
        else:
            clamped_min = min(max(z_min, self._map_intensity_min), self._map_intensity_max)
            clamped_max = min(max(z_max, self._map_intensity_min), self._map_intensity_max)
            min_pos = int(round((clamped_min - self._map_intensity_min) / span * 1000.0))
            max_pos = int(round((clamped_max - self._map_intensity_min) / span * 1000.0))
        self.map_mask_slider.blockSignals(True)
        self.map_mask_slider.setValues(min_pos, max_pos)
        self.map_mask_slider.blockSignals(False)

    def _map_image_display_limits(self) -> tuple[Optional[float], Optional[float]]:
        if self._current_mode_index() != 0:
            return None, None
        return self._map_intensity_range_values()

    def _current_snapshot_display_image(self) -> Optional[np.ndarray]:
        if self.loaded is None or self._current_mode_index() != 1:
            return None
        data = np.asarray(self.loaded.eels_signal.data)
        if data.ndim != 3 or data.shape[0] == 0:
            return None
        frame_index = min(self.snapshot_index_spin.value(), data.shape[0] - 1)
        frame_index = max(0, frame_index)
        return np.asarray(data[frame_index], dtype=float)

    def _snapshot_intensity_range_values(self) -> tuple[float, float]:
        min_pos = self.snapshot_z_slider.lowerValue()
        max_pos = self.snapshot_z_slider.upperValue()
        span = self._snapshot_intensity_max - self._snapshot_intensity_min
        if span <= 0:
            return self._snapshot_intensity_min, self._snapshot_intensity_max
        z_min = self._snapshot_intensity_min + (span * min_pos / 1000.0)
        z_max = self._snapshot_intensity_min + (span * max_pos / 1000.0)
        return z_min, z_max

    def _snapshot_z_slider_value_from_intensity(self, intensity: float) -> int:
        span = self._snapshot_intensity_max - self._snapshot_intensity_min
        if span <= 0:
            return 0
        clamped = min(max(float(intensity), self._snapshot_intensity_min), self._snapshot_intensity_max)
        return int(round((clamped - self._snapshot_intensity_min) / span * 1000.0))

    def _set_snapshot_z_slider_values(self, z_min: float, z_max: float):
        span = self._snapshot_intensity_max - self._snapshot_intensity_min
        if span <= 0:
            min_pos = 0
            max_pos = 1000
        else:
            clamped_min = min(max(z_min, self._snapshot_intensity_min), self._snapshot_intensity_max)
            clamped_max = min(max(z_max, self._snapshot_intensity_min), self._snapshot_intensity_max)
            min_pos = int(round((clamped_min - self._snapshot_intensity_min) / span * 1000.0))
            max_pos = int(round((clamped_max - self._snapshot_intensity_min) / span * 1000.0))
        self.snapshot_z_slider.blockSignals(True)
        self.snapshot_z_slider.setValues(min_pos, max_pos)
        self.snapshot_z_slider.blockSignals(False)

    def _snapshot_image_display_limits(self) -> tuple[Optional[float], Optional[float]]:
        if self._current_mode_index() != 1:
            return None, None
        return self._snapshot_intensity_range_values()

    def _reset_snapshot_histogram_controls(self, preserve_range: bool = False):
        image = self._current_snapshot_display_image()
        if image is None or image.size == 0:
            return
        previous_min, previous_max = self._snapshot_intensity_range_values()
        self._snapshot_intensity_min = float(np.min(image))
        self._snapshot_intensity_max = float(np.max(image))
        if preserve_range:
            self._set_snapshot_z_slider_values(previous_min, previous_max)
        else:
            self._set_snapshot_z_slider_values(self._snapshot_intensity_min, self._snapshot_intensity_max)
        self._update_snapshot_histogram()

    def _update_snapshot_histogram(self):
        ax = self.snapshot_hist_canvas.reset_axis()
        image = self._current_snapshot_display_image()
        if image is None or image.size == 0:
            self._set_empty_axis_message(ax, "Load a snapshot stack to inspect intensity histogram")
            self.snapshot_z_min_label.setText("Z min: -")
            self.snapshot_z_max_label.setText("Z max: -")
            self.snapshot_hist_canvas.draw_idle()
            return
        ax.hist(
            np.ravel(image),
            bins=64,
            color=PLOT_MAIN,
            edgecolor=SPINE,
            log=self.snapshot_hist_log_checkbox.isChecked(),
        )
        z_min, z_max = self._snapshot_intensity_range_values()
        ax.axvline(z_min, color=PLOT_PREVIEW, linewidth=1.2)
        ax.axvline(z_max, color=PLOT_FIT, linewidth=1.2)
        ax.set_title("Snapshot intensity histogram")
        ax.set_xlabel("Detector intensity")
        ax.set_ylabel("Pixels")
        self.snapshot_z_min_label.setText(f"Z min: {z_min:.2f}")
        self.snapshot_z_max_label.setText(f"Z max: {z_max:.2f}")
        self.snapshot_hist_canvas.draw_idle()

    def _snapshot_histogram_handle_at(self, x_value: Optional[float]) -> Optional[str]:
        if x_value is None:
            return None
        z_min, z_max = self._snapshot_intensity_range_values()
        axis_span = self._snapshot_intensity_max - self._snapshot_intensity_min
        tolerance = max(axis_span * 0.03, 1e-12)
        if abs(float(x_value) - z_min) <= abs(float(x_value) - z_max):
            return "min" if abs(float(x_value) - z_min) <= tolerance else None
        return "max" if abs(float(x_value) - z_max) <= tolerance else None

    def _set_snapshot_histogram_handle_from_x(self, handle: str, x_value: Optional[float]):
        if x_value is None:
            return
        slider_value = self._snapshot_z_slider_value_from_intensity(float(x_value))
        if handle == "min":
            self.snapshot_z_slider.setLowerValue(min(slider_value, self.snapshot_z_slider.upperValue()))
        elif handle == "max":
            self.snapshot_z_slider.setUpperValue(max(slider_value, self.snapshot_z_slider.lowerValue()))

    def _on_snapshot_histogram_press(self, event):
        if event.inaxes is None or event.inaxes not in self.snapshot_hist_canvas.figure.axes:
            return
        if event.button != 1:
            return
        handle = self._snapshot_histogram_handle_at(event.xdata)
        if handle is None:
            return
        self._active_snapshot_histogram_handle = handle
        self._set_snapshot_histogram_handle_from_x(handle, event.xdata)

    def _on_snapshot_histogram_move(self, event):
        if self._active_snapshot_histogram_handle is None:
            return
        if event.inaxes is None or event.inaxes not in self.snapshot_hist_canvas.figure.axes:
            return
        self._set_snapshot_histogram_handle_from_x(self._active_snapshot_histogram_handle, event.xdata)

    def _on_snapshot_histogram_release(self, _event):
        self._active_snapshot_histogram_handle = None

    def _reset_map_histogram_controls(self, preserve_mask_range: bool = False):
        image = self._map_intensity_image()
        if image is None:
            return
        previous_min, previous_max = self._map_intensity_range_values()
        values = np.asarray(image, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self._map_intensity_min = float(np.min(values))
        self._map_intensity_max = float(np.max(values))
        if preserve_mask_range:
            self._set_map_mask_slider_values(previous_min, previous_max)
        else:
            self._set_map_mask_slider_values(self._map_intensity_min, self._map_intensity_max)
        self._update_map_histogram()

    def _update_map_histogram(self):
        ax = self.map_hist_canvas.reset_axis()
        image = self._map_intensity_image()
        if image is None:
            self._set_empty_axis_message(ax, "Load a 2D map to inspect intensity histogram")
            self.map_hist_canvas.draw_idle()
            return
        values = np.asarray(image, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            self._set_empty_axis_message(ax, "No finite map intensities available")
            self.map_hist_canvas.draw_idle()
            return
        ax.hist(
            values,
            bins=64,
            color=PLOT_MAIN,
            edgecolor=SPINE,
            log=self.map_hist_log_checkbox.isChecked(),
        )
        z_min, z_max = self._map_intensity_range_values()
        ax.axvline(z_min, color=PLOT_PREVIEW, linewidth=1.2)
        ax.axvline(z_max, color=PLOT_FIT, linewidth=1.2)
        ax.set_title("Map intensity histogram")
        ax.set_xlabel("Integrated intensity")
        ax.set_ylabel("Pixels")
        self.map_mask_min_label.setText(f"Mask min: {z_min:.2f}")
        self.map_mask_max_label.setText(f"Mask max: {z_max:.2f}")
        self.map_hist_canvas.draw_idle()

    def _map_histogram_handle_at(self, x_value: Optional[float]) -> Optional[str]:
        if x_value is None:
            return None
        z_min, z_max = self._map_intensity_range_values()
        axis_span = self._map_intensity_max - self._map_intensity_min
        tolerance = max(axis_span * 0.03, 1e-12)
        if abs(float(x_value) - z_min) <= abs(float(x_value) - z_max):
            return "min" if abs(float(x_value) - z_min) <= tolerance else None
        return "max" if abs(float(x_value) - z_max) <= tolerance else None

    def _set_map_histogram_handle_from_x(self, handle: str, x_value: Optional[float]):
        if x_value is None:
            return
        slider_value = self._map_mask_slider_value_from_intensity(float(x_value))
        if handle == "min":
            self.map_mask_slider.setLowerValue(min(slider_value, self.map_mask_slider.upperValue()))
        elif handle == "max":
            self.map_mask_slider.setUpperValue(max(slider_value, self.map_mask_slider.lowerValue()))

    def _on_map_histogram_press(self, event):
        if event.inaxes is None or event.inaxes not in self.map_hist_canvas.figure.axes:
            return
        if event.button != 1:
            return
        handle = self._map_histogram_handle_at(event.xdata)
        if handle is None:
            return
        self._active_map_histogram_handle = handle
        self._set_map_histogram_handle_from_x(handle, event.xdata)

    def _on_map_histogram_move(self, event):
        if self._active_map_histogram_handle is None:
            return
        if event.inaxes is None or event.inaxes not in self.map_hist_canvas.figure.axes:
            return
        self._set_map_histogram_handle_from_x(self._active_map_histogram_handle, event.xdata)

    def _on_map_histogram_release(self, _event):
        self._active_map_histogram_handle = None

    def _on_map_mask_slider_changed(self, *_args):
        self._update_map_histogram()
        if self._current_mode_index() == 0:
            self._update_view_for_active_tab()
            self._update_map_mask_preview()

    def _update_map_mask_preview(self):
        if self._current_mode_index() != 0:
            return
        ax = self._prepare_image_axis(self.corrected_canvas.reset_axis())
        image = self._map_intensity_image()
        polygon_mask = self._current_polygon_mask()
        if image is None or polygon_mask is None:
            self._set_empty_axis_message(ax, "Masked image")
            self.corrected_canvas.draw_idle()
            return
        z_min, z_max = self._map_intensity_range_values()
        selection_mask = polygon_mask & (image >= z_min) & (image <= z_max)
        masked = np.ma.masked_where(~selection_mask, image)
        ax.imshow(masked, cmap="inferno", origin="upper", vmin=z_min, vmax=z_max)
        ax.set_title("Masked image")
        ax.set_xlabel("")
        ax.set_ylabel("")
        self.corrected_canvas.draw_idle()

    def _detach_selector(self):
        if self.selector is None:
            return
        try:
            self.selector.set_active(False)
        except Exception:
            pass
        disconnect = getattr(self.selector, "disconnect_events", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass
        self.selector = None

    def _clear_corrected_preview(self, message: str):
        ax = self._prepare_image_axis(self.corrected_canvas.reset_axis())
        self._set_empty_axis_message(ax, message)
        self.corrected_canvas.draw_idle()

    def _draw_initial_image(self):
        self._capture_current_image_view()

        self._detach_selector()
        ax = self._prepare_image_axis(self.image_canvas.reset_axis())
        if self.loaded is None:
            self._set_empty_axis_message(ax, "Load an EELS dataset to start")
            self.image_canvas.draw_idle()
            self._clear_corrected_preview("Load data and run alignment to preview corrected output")
            return

        mode_index = self._current_mode_index()
        data = np.asarray(self.loaded.eels_signal.data)
        if mode_index == 0:
            image = self._map_intensity_image()
            z_min, z_max = self._map_image_display_limits()
            im = ax.imshow(image, cmap="inferno", origin="upper", vmin=z_min, vmax=z_max)
            ax.set_title("Map intensity preview")
            ax.set_xlabel("")
            ax.set_ylabel("")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            colorbar = self.image_canvas.figure.colorbar(im, cax=cax)
            self._format_colorbar_scientific(colorbar)
        else:
            frame_index = min(self.snapshot_index_spin.value(), data.shape[0] - 1)
            detector_image = data[frame_index]
            z_min, z_max = self._snapshot_image_display_limits()
            ax.imshow(detector_image, cmap="viridis", origin="upper", aspect="auto", vmin=z_min, vmax=z_max)
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

        self.image_canvas.draw_idle()
        self._attach_selector()
        self._update_selection_overlay()
        if mode_index == 0:
            self._update_map_mask_preview()
        else:
            self._clear_corrected_preview("Run alignment to preview the corrected snapshot")
            self._update_snapshot_histogram()

    def _attach_selector(self):
        axes = self.image_canvas.figure.axes
        if not axes:
            return
        if self._current_mode_index() == 0:
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
            self.selector.set_active(self._mouse_mode == "roi")

    def _on_polygon_selected(self, verts):
        if self._current_mode_index() != 0:
            return
        vertices = [(float(x), float(y)) for x, y in verts]
        if len(vertices) >= 3:
            self.map_polygon_vertices.append(vertices)
        self._reset_map_histogram_controls(preserve_mask_range=True)
        self._draw_initial_image()

    def _on_rectangle_selected(self, eclick, erelease):
        if self._current_mode_index() == 0:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return
        y0, y1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
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
        if self._current_mode_index() == 1:
            self._capture_current_image_view()
        ax = axes[0]
        if self.selection_rect is not None:
            try:
                self.selection_rect.remove()
            except Exception:
                pass
        for selection_polygon in self.selection_polygons:
            try:
                selection_polygon.remove()
            except Exception:
                pass
        self.selection_polygons = []
        if self._current_mode_index() == 0:
            for vertices in self.map_polygon_vertices:
                selection_polygon = Polygon(
                    vertices,
                    closed=True,
                    fill=False,
                    edgecolor=PLOT_PREVIEW,
                    linewidth=1.5,
                )
                ax.add_patch(selection_polygon)
                self.selection_polygons.append(selection_polygon)
            self.image_canvas.draw_idle()
            self._update_clear_roi_button_enabled()
            return
        else:
            x0, x1 = ax.get_xlim()
            y0 = self.stack_y0.value()
            y1 = self.stack_y1.value()
        width = x1 - x0
        height = max(1, y1 - y0)
        self.selection_rect = Rectangle((x0, y0), width, height, fill=False, edgecolor=PLOT_PREVIEW, linewidth=1.5)
        ax.add_patch(self.selection_rect)
        if (
            self._current_mode_index() == 1
            and self.loaded is not None
            and self._image_view_limits is None
        ):
            ax.set_xlim(0, self.loaded.eels_signal.data.shape[2] - 1)
            ax.set_ylim(self.loaded.eels_signal.data.shape[1] - 1, 0)
        self.image_canvas.draw_idle()

    def _update_snapshot_navigation_enabled(self):
        enabled = self.loaded is not None and self._current_mode_index() == 1
        self.snapshot_index_spin.setEnabled(enabled)
        self.snapshot_prev_button.setEnabled(enabled)
        self.snapshot_next_button.setEnabled(enabled)

    def _show_previous_snapshot(self):
        if not self.snapshot_index_spin.isEnabled():
            return
        self._capture_current_image_view()
        self.snapshot_index_spin.setValue(max(0, self.snapshot_index_spin.value() - 1))

    def _show_next_snapshot(self):
        if not self.snapshot_index_spin.isEnabled():
            return
        self._capture_current_image_view()
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
                    MessageBoxButton.Retry
                    | MessageBoxButton.Ok
                    | MessageBoxButton.Cancel,
                    MessageBoxButton.Retry,
                )
                if reply == MessageBoxButton.Retry:
                    suggested = target_dir
                    continue
                if reply == MessageBoxButton.Cancel:
                    return None
            self._remember_data_dir(target_dir)
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
        mode = "snapshot" if self._current_mode_index() == 1 else "map"
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
spectrum_vxy = np.loadtxt(bundle_dir / f"{{stem}}.vxy", skiprows=1)

def curve_fwhm(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    peak = float(np.max(y_values))
    if peak <= 0:
        return 0.0
    half_max = peak / 2.0
    mask = y_values >= half_max
    if not np.any(mask):
        return 0.0
    return float(x_values[mask][-1] - x_values[mask][0])

def format_scientific(value, significant_digits=3):
    precision = max(0, int(significant_digits) - 1)
    return f"{{float(value):.{{precision}}e}}"

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
ax2.plot(arrays["zlp_x"], arrays["zlp_y_integrated"], ".", color="#f2f2f2", markersize=3, label="integrated")
ax2.plot(arrays["zlp_x"], arrays["zlp_fit"], color="#ff4d6d", linewidth=1.2, label="fit")
ax2.axvline(arrays["zlp_center_ev"][0], color="#3b82f6", linestyle="--", linewidth=1.0)
if "zlp_snapshot_x" in arrays and arrays["zlp_snapshot_x"].size:
    scale = arrays["zlp_snapshot_scale"][0]
    snap_index = int(arrays["snapshot_index"][0])
    ax2.plot(
        arrays["zlp_snapshot_x"],
        arrays["zlp_snapshot_y_scaled"],
        color="#f59e0b",
        linewidth=1.0,
        label=f"snapshot {{snap_index}} (scaled x{{scale:.2f}})",
    )
ax2.set_title(f"ZLP, aligned (FWHM = {{format_scientific(curve_fwhm(arrays['zlp_x'], arrays['zlp_fit']))}} eV)")
ax2.set_xlabel("Energy loss (eV)")
ax2.set_ylabel("ZLP region")
ax2.legend(loc="lower left", fontsize=8)

ax3 = fig.add_subplot(gs[1, :])
ax3.plot(spectrum_vxy[:, 0], spectrum_vxy[:, 2], color="#f2f2f2", linewidth=1.0)
ax3.set_xlabel("Energy loss (eV, ZLP corrected)")
ax3.set_ylabel("Intensity (a.u.)")
sec = ax3.secondary_xaxis("top", functions=(lambda x: x * 8065.54429, lambda x: x / 8065.54429))
sec.set_xlabel(r"Wavenumber (cm$^{{-1}}$)")

plt.show()
'''

    def _build_export_vxy(self, energy_axis_ev: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        energy_axis_ev = np.asarray(energy_axis_ev, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        return np.column_stack([energy_axis_ev, energy_axis_ev * EV_TO_CMINV, intensity])

    def _write_saved_export_vxy_files(self, export_dir: Path, data_stem: str):
        for index, entry in enumerate(self.saved_map_entries):
            spectrum_vxy = self._build_export_vxy(entry.energy_axis, entry.spectrum)
            np.savetxt(
                export_dir / f"{data_stem}_saved_{index}.vxy",
                spectrum_vxy,
                header="eV cm-1 y",
                comments="",
            )

    def _build_export_figure(self, arrays: dict[str, np.ndarray], spectrum_vxy: np.ndarray) -> Figure:
        fig = Figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])
        style_mpl_axes(fig, ax0, ax1, ax2, ax3)

        preview_entries = self._visible_saved_preview_entries()
        if preview_entries:
            preview_mode = preview_entries[0][1].mode
            base_image = np.asarray(preview_entries[0][1].display_image, dtype=float)
            ax0.imshow(base_image, cmap="inferno" if preview_mode == "map" else "viridis", origin="upper", aspect="auto")
            ax0.set_title("ROI")
            ax0.set_xlabel("")
            ax0.set_ylabel("")
            for row, entry in preview_entries:
                color = self._saved_map_entry_color(row)
                for vertices in self._entry_polygon_list(entry):
                    ax0.add_patch(
                        Polygon(
                            vertices,
                            closed=True,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.8,
                        )
                    )

            base_masked = self._saved_entry_masked_image_for_display(preview_entries[0][1])
            combined_masked = np.full_like(base_masked, np.nan, dtype=float)
            for _, entry in preview_entries:
                entry_masked = self._saved_entry_masked_image_for_display(entry)
                if entry_masked.shape != combined_masked.shape:
                    continue
                combined_masked = np.where(np.isfinite(entry_masked), entry_masked, combined_masked)
            masked = np.ma.masked_invalid(combined_masked)
            ax1.imshow(masked, cmap="inferno" if preview_mode == "map" else "viridis", origin="upper", aspect="auto")
            ax1.set_title("Masked ROI")
            ax1.set_xlabel("")
            ax1.set_ylabel("")

            active_entry = preview_entries[0][1]
            active_fit_mask = (
                (active_entry.energy_axis_raw >= active_entry.fit_window[0])
                & (active_entry.energy_axis_raw <= active_entry.fit_window[1])
            )
            active_spectra_stack = np.asarray(active_entry.selected_spectra, dtype=float)
            if active_spectra_stack.ndim == 1:
                active_spectra_stack = active_spectra_stack[np.newaxis, :]
            active_zlp_profile = np.sum(active_spectra_stack[:, active_fit_mask], axis=0)
            for row, entry in preview_entries:
                fit_mask = (
                    (entry.energy_axis_raw >= entry.fit_window[0])
                    & (entry.energy_axis_raw <= entry.fit_window[1])
                )
                spectra_stack = np.asarray(entry.selected_spectra, dtype=float)
                if spectra_stack.ndim == 1:
                    spectra_stack = spectra_stack[np.newaxis, :]
                for index, spectrum in enumerate(spectra_stack[:, fit_mask]):
                    ax2.plot(
                        entry.energy_axis_raw[fit_mask],
                        spectrum,
                        color=self._saved_map_entry_color(row),
                        alpha=0.15,
                        linewidth=0.7,
                        label=(entry.comment.strip() or f"Map {row + 1}") if index == 0 else None,
                    )
            ax2.axvline(
                0.0,
                color=PLOT_PREVIEW,
                linewidth=1.0,
                linestyle=":",
                label="alignment target" if len(preview_entries) == 1 else "_nolegend_",
            )
            ax2.set_title(self._zlp_title(active_entry.energy_axis_raw[active_fit_mask], active_zlp_profile))
            ax2.set_xlabel("Energy loss (eV)")
            ax2.set_ylabel("Intensity")
            ax2.legend(loc="best", fontsize=8)

            normalize = self.normalize_saved_spectra_checkbox.isChecked()
            for row, entry in preview_entries:
                spectrum = self._scaled_saved_entry_spectrum(entry)
                label = entry.comment.strip() or f"Map {row + 1}"
                linewidth = 1.8 if entry.locked else 1.0
                alpha = 1.0 if entry.locked else 0.85
                ax3.plot(
                    entry.energy_axis,
                    spectrum,
                    linewidth=linewidth,
                    alpha=alpha,
                    color=self._saved_map_entry_color(row),
                    label=label,
                )
            ax3.set_xlabel("Energy loss (eV, ZLP corrected)")
            ax3.set_ylabel("Normalized intensity" if normalize else "Intensity (a.u.)")
            sec = ax3.secondary_xaxis("top", functions=(lambda x: x * EV_TO_CMINV, lambda x: x / EV_TO_CMINV))
            sec.set_xlabel(r"Wavenumber (cm$^{-1}$)")
            style_secondary_axis(sec)
            ax3.legend(loc="best", fontsize=8)
            return fig

        if "detector_image_snapshot" in arrays:
            ax0.imshow(arrays["detector_image_snapshot"], cmap="viridis", origin="upper", aspect="auto")
            ax0.set_title(f"Snapshot {int(arrays['snapshot_index'][0])}")
            ax0.set_xlabel("spectral channel")
            ax0.set_ylabel("detector vertical")
        else:
            ax0.imshow(arrays["map_display_image"], cmap="gray", origin="upper")
            ax0.contour(arrays["map_selection_mask"], levels=[0.5], colors=["cyan"], linewidths=1.0)
            ax0.set_title("Map ROI")
            ax0.set_xlabel("")
            ax0.set_ylabel("")

        if "detector_image_aligned_snapshot" in arrays and arrays["detector_image_aligned_snapshot"].size:
            ax1.imshow(arrays["detector_image_aligned_snapshot"], cmap="viridis", origin="upper", aspect="auto")
            ax1.set_title(f"Snapshot {int(arrays['snapshot_index'][0])} aligned")
            ax1.set_xlabel("spectral channel")
            ax1.set_ylabel("aligned detector rows")
        elif "map_masked_image" in arrays:
            masked = np.ma.masked_invalid(np.asarray(arrays["map_masked_image"], dtype=float))
            ax1.imshow(masked, cmap="inferno", origin="upper")
            ax1.set_title("Masked ROI")
            ax1.set_xlabel("")
            ax1.set_ylabel("")
        else:
            ax1.set_axis_off()

        ax2.plot(arrays["zlp_x"], arrays["zlp_y_integrated"], ".", color="#f2f2f2", markersize=3, label="integrated")
        ax2.plot(arrays["zlp_x"], arrays["zlp_fit"], color="#ff4d6d", linewidth=1.2, label="fit")
        ax2.axvline(arrays["zlp_center_ev"][0], color="#3b82f6", linestyle="--", linewidth=1.0)
        if "zlp_snapshot_x" in arrays and arrays["zlp_snapshot_x"].size:
            scale = arrays["zlp_snapshot_scale"][0]
            snap_index = int(arrays["snapshot_index"][0])
            ax2.plot(
                arrays["zlp_snapshot_x"],
                arrays["zlp_snapshot_y_scaled"],
                color="#f59e0b",
                linewidth=1.0,
                label=f"snapshot {snap_index} (scaled x{scale:.2f})",
            )
        ax2.set_title(self._zlp_title(arrays["zlp_x"], arrays["zlp_fit"], aligned=True))
        ax2.set_xlabel("Energy loss (eV)")
        ax2.set_ylabel("ZLP region")
        ax2.legend(loc="lower left", fontsize=8)

        ax3.plot(spectrum_vxy[:, 0], spectrum_vxy[:, 2], color="#f2f2f2", linewidth=1.0)
        ax3.set_xlabel("Energy loss (eV, ZLP corrected)")
        ax3.set_ylabel("Intensity (a.u.)")
        sec = ax3.secondary_xaxis("top", functions=(lambda x: x * EV_TO_CMINV, lambda x: x / EV_TO_CMINV))
        sec.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        style_secondary_axis(sec)

        return fig

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

        vxy_path = export_dir / f"{data_stem}.vxy"
        json_path = export_dir / f"{data_stem}.json"
        npz_path = export_dir / f"{data_stem}_graph_data.npz"
        script_path = export_dir / f"{data_stem}_reproduce.py"
        pdf_path = export_dir / f"{data_stem}.pdf"
        png_path = export_dir / f"{data_stem}.png"

        spectrum_vxy = self._build_export_vxy(
            self.current_result.energy_axis_calibrated,
            self.current_result.summed_spectrum,
        )
        np.savetxt(vxy_path, spectrum_vxy, header="eV cm-1 y", comments="")
        json_path.write_text(json.dumps(params, indent=2))
        np.savez(npz_path, **arrays)
        script_path.write_text(self._build_repro_script(data_stem))
        self._write_saved_export_vxy_files(export_dir, data_stem)
        export_figure = self._build_export_figure(arrays, spectrum_vxy)
        try:
            export_figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
            export_figure.savefig(png_path, dpi=300, bbox_inches="tight")
        finally:
            export_figure.clear()

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

        # Ignore limits fully outside the image. Tight zoom windows are valid user state.
        if (
            x_max < 0
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
        if self._current_mode_index() == 1:
            self._capture_current_image_view()
            self._capture_current_corrected_view()
            self._reset_snapshot_histogram_controls(preserve_range=True)
        self._update_view_for_active_tab()

    def _on_snapshot_z_slider_changed(self, *_args):
        self._update_snapshot_histogram()
        if self._current_mode_index() == 1:
            self._update_view_for_active_tab()

    def _current_snapshot_spectrum(self) -> Optional[np.ndarray]:
        if (
            self.loaded is None
            or self._current_mode_index() != 1
            or not isinstance(self.current_result, StackProcessingResult)
        ):
            return None
        aligned_stack = np.asarray(self.current_result.aligned_stack, dtype=float)
        if aligned_stack.ndim != 3 or aligned_stack.shape[0] == 0:
            return None
        frame_offset = min(self.snapshot_index_spin.value() - self.frame_start.value(), aligned_stack.shape[0] - 1)
        frame_offset = max(0, frame_offset)
        return aligned_stack[frame_offset].sum(axis=0)

    def _current_snapshot_aligned_image(self, full_detector: bool = False) -> Optional[np.ndarray]:
        if (
            self.loaded is None
            or self._current_mode_index() != 1
            or not isinstance(self.current_result, StackProcessingResult)
        ):
            return None
        aligned_stack = np.asarray(self.current_result.aligned_stack, dtype=float)
        if aligned_stack.ndim != 3 or aligned_stack.shape[0] == 0:
            return None
        frame_offset = min(self.snapshot_index_spin.value() - self.frame_start.value(), aligned_stack.shape[0] - 1)
        frame_offset = max(0, frame_offset)
        aligned_image = aligned_stack[frame_offset]
        if not full_detector:
            return aligned_image
        detector_shape = np.asarray(self.loaded.eels_signal.data[self.snapshot_index_spin.value()]).shape
        full_image = np.full(detector_shape, np.nan, dtype=float)
        y0, y1 = self.current_result.vertical_range
        row_stop = min(y1, detector_shape[0], y0 + aligned_image.shape[0])
        col_stop = min(detector_shape[1], aligned_image.shape[1])
        if row_stop > y0 and col_stop > 0:
            full_image[y0:row_stop, :col_stop] = aligned_image[: row_stop - y0, :col_stop]
        return full_image

    def _fit_window(self):
        return (self.fit_start_spin.value(), self.fit_stop_spin.value())

    def _guess_window(self):
        return (self.guess_start_spin.value(), self.guess_stop_spin.value())

    def _process_current(self):
        if self.loaded is None or not self.loaded.eels_path:
            self._log("Load an EELS dataset before processing.")
            return
        mode_index = self._current_mode_index()
        if mode_index == 0:
            polygon_mask = self._current_polygon_mask()
            if polygon_mask is None:
                self._log("Processing failed: Map polygon ROI is not available.")
                return
            self._start_processing_worker(
                mode_index=mode_index,
                map_kwargs={
                    "energy_range": (self.energy_start_spin.value(), self.energy_stop_spin.value()),
                    "polygon_mask": polygon_mask,
                    "intensity_range": self._map_intensity_range_values(),
                    "display_image": self._current_display_image(),
                    "fit_window": self._fit_window(),
                    "guess_window": self._guess_window(),
                },
            )
            return

        self._start_processing_worker(
            mode_index=mode_index,
            snapshot_kwargs={
                "vertical_range": (self.stack_y0.value(), self.stack_y1.value()),
                "frame_range": (self.frame_start.value(), self.frame_stop.value()),
                "fit_window": self._fit_window(),
                "guess_window": self._guess_window(),
            },
        )

    def _current_display_image(self):
        image = self._map_intensity_image()
        if image is not None:
            return np.asarray(image, dtype=float)
        return None

    def _render_result(self, result):
        if isinstance(result, StackProcessingResult):
            self._capture_current_image_view()

        if isinstance(result, MapProcessingResult):
            self._detach_selector()
            image_ax = self._prepare_image_axis(self.image_canvas.reset_axis())
            z_min, z_max = self._map_image_display_limits()
            im = image_ax.imshow(result.intensity_image, cmap="inferno", origin="upper", vmin=z_min, vmax=z_max)
            divider = make_axes_locatable(image_ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            colorbar = self.image_canvas.figure.colorbar(im, cax=cax)
            self._format_colorbar_scientific(colorbar)
            image_ax.set_title("ROI")
            pixel_count_text = str(result.selected_pixel_count)
            cal_axis = result.energy_axis_calibrated
            image_ax.set_xlabel("")
            image_ax.set_ylabel("")
            self.image_canvas.draw_idle()
            self._attach_selector()
            self._update_selection_overlay()
        else:
            pixel_count_text = str(result.aligned_stack.shape[0] * result.aligned_stack.shape[1])
            cal_axis = result.energy_axis_calibrated

        if isinstance(result, StackProcessingResult) and self.corrected_canvas.figure.axes:
            self._capture_current_corrected_view()
        corrected_ax = self._prepare_image_axis(self.corrected_canvas.reset_axis())
        if isinstance(result, StackProcessingResult):
            corrected_image = self._current_snapshot_aligned_image(full_detector=True)
            if corrected_image is not None:
                z_min, z_max = self._snapshot_image_display_limits()
                corrected_ax.imshow(corrected_image, cmap="viridis", origin="upper", aspect="auto", vmin=z_min, vmax=z_max)
                corrected_ax.set_title(f"Snapshot {self.snapshot_index_spin.value()} aligned")
                corrected_ax.set_xlabel("")
                corrected_ax.set_ylabel("")
                self._apply_saved_corrected_view(
                    corrected_ax,
                    corrected_image.shape[1],
                    corrected_image.shape[0],
                )
                corrected_ax.callbacks.connect("xlim_changed", self._on_corrected_view_changed)
                corrected_ax.callbacks.connect("ylim_changed", self._on_corrected_view_changed)
            else:
                self._set_empty_axis_message(corrected_ax, "No corrected image")
        else:
            masked = np.ma.masked_invalid(result.masked_image)
            z_min, z_max = self._map_image_display_limits()
            corrected_ax.imshow(masked, cmap="inferno", origin="upper", vmin=z_min, vmax=z_max)
            corrected_ax.set_title(f"No. spectra: {result.selected_pixel_count}")
            corrected_ax.set_xlabel("")
            corrected_ax.set_ylabel("")
        self.corrected_canvas.draw_idle()

        fit_ax = self.fit_canvas.reset_axis()
        if isinstance(result, StackProcessingResult):
            fit_ax.plot(result.zero_loss_fit.fit_x, result.zero_loss_fit.fit_y, ".", color=PLOT_MAIN, markersize=3)
            fit_ax.plot(result.zero_loss_fit.fit_x, result.zero_loss_fit.best_fit, color=PLOT_FIT, linewidth=1.2)
            fit_ax.axvline(
                result.zero_loss_fit.center_ev,
                color=PLOT_PREVIEW,
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
                    color=PLOT_WARNING,
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
            for index, row in enumerate(zlp_stack):
                fit_ax.plot(
                    result.energy_axis_raw[fit_mask],
                    row,
                    color=PLOT_MAIN,
                    linewidth=0.6,
                    alpha=0.25,
                    label="individually aligned spectra" if index == 0 else None,
                )
            fit_ax.axvline(
                0.0,
                color=PLOT_PREVIEW,
                linewidth=1.0,
                linestyle=":",
                label="_nolegend_",
            )
            fit_ax.plot(
                result.zero_loss_fit.fit_x,
                result.zero_loss_fit.fit_y,
                ".",
                color=PLOT_MAIN,
                markersize=3,
                label="summed after alignment",
            )
            fit_ax.plot(
                result.zero_loss_fit.fit_x,
                result.zero_loss_fit.best_fit,
                color=PLOT_FIT,
                linewidth=1.2,
                label="fit of aligned sum",
            )
            fit_ax.axvline(
                result.zero_loss_fit.center_ev,
                color=PLOT_WARNING,
                linewidth=1.0,
                linestyle="--",
                label=f"residual centroid = {result.zero_loss_fit.center_ev:.5f} eV",
            )
        fit_ax.set_xlabel("Energy loss (eV)")
        fit_ax.set_ylabel("Intensity")
        fit_ax.set_title(
            self._zlp_title(
                result.zero_loss_fit.fit_x,
                result.zero_loss_fit.best_fit,
                aligned=isinstance(result, StackProcessingResult),
            )
        )
        fit_ax.legend(loc="lower left", fontsize=8)

        self.fit_canvas.draw_idle()

        spectrum_ax = self.spectrum_canvas.reset_axis()
        spectrum_ax.plot(cal_axis, result.summed_spectrum, color=PLOT_MAIN, linewidth=1.0)
        spectrum_ax.set_xlabel("Energy loss (eV, ZLP corrected)")
        spectrum_ax.set_ylabel("Intensity (a.u.)")
        spectrum_ax.set_title(self._current_filename_label())
        spectrum_ax_top = spectrum_ax.secondary_xaxis(
            "top",
            functions=(lambda x: x * EV_TO_CMINV, lambda x: x / EV_TO_CMINV),
        )
        spectrum_ax_top.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        style_secondary_axis(spectrum_ax_top)
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
            x_max < 0
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

    def _on_corrected_view_changed(self, corrected_ax):
        if self._syncing_corrected_xlim:
            return
        if corrected_ax.images:
            self._corrected_view_limits = (corrected_ax.get_xlim(), corrected_ax.get_ylim())

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
    app.setApplicationName("vibeels")
    app.setDesktopFileName("vibeels")
    icon_path = _application_icon_path()
    icon = _application_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    _set_macos_dock_icon(icon_path)
    apply_qt_theme(app)
    window = VibeelsWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

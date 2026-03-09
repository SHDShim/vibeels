from __future__ import annotations

from matplotlib import rcParams
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette


FIGURE_BG = "#2b2d31"
AXES_BG = "#1e1f22"
GRID = "#43464d"
TEXT = "#f0f0f0"
SPINE = "#c9c9c9"

PLOT_MAIN = "#f2f2f2"
PLOT_SECONDARY = "#60a5fa"
PLOT_FIT = "#ff4d6d"
PLOT_PREVIEW = "#3b82f6"
PLOT_WARNING = "#f59e0b"
PLOT_SUCCESS = "#22c55e"
PLOT_CYCLE = [
    PLOT_SECONDARY,
    PLOT_FIT,
    PLOT_WARNING,
    PLOT_SUCCESS,
    PLOT_MAIN,
    "#14b8a6",
    "#fb7185",
    "#eab308",
    "#38bdf8",
    "#4ade80",
]

def build_dark_palette() -> QPalette:
    palette = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    palette.setColor(role.Window, QColor(40, 42, 46))
    palette.setColor(role.WindowText, Qt.GlobalColor.white)
    palette.setColor(role.Base, QColor(26, 28, 31))
    palette.setColor(role.AlternateBase, QColor(45, 48, 53))
    palette.setColor(role.ToolTipBase, QColor(26, 28, 31))
    palette.setColor(role.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(role.Text, Qt.GlobalColor.white)
    palette.setColor(role.Button, QColor(52, 55, 61))
    palette.setColor(role.ButtonText, Qt.GlobalColor.white)
    palette.setColor(role.Link, QColor(88, 166, 255))
    palette.setColor(role.Highlight, QColor(88, 166, 255))
    palette.setColor(role.HighlightedText, QColor(20, 20, 20))
    palette.setColor(group.Disabled, role.Text, QColor(130, 130, 130))
    palette.setColor(group.Disabled, role.ButtonText, QColor(130, 130, 130))
    palette.setColor(group.Disabled, role.WindowText, QColor(130, 130, 130))
    return palette


def apply_qt_theme(app) -> None:
    app.setStyle("Fusion")
    app.setPalette(build_dark_palette())


def configure_matplotlib_defaults() -> None:
    rcParams.update(
        {
            "figure.facecolor": FIGURE_BG,
            "axes.facecolor": AXES_BG,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "savefig.facecolor": FIGURE_BG,
            "savefig.edgecolor": FIGURE_BG,
        }
    )

def style_mpl_axes(fig, *axes) -> None:
    fig.set_facecolor(FIGURE_BG)
    for ax in axes:
        ax.set_facecolor(AXES_BG)
        ax.grid(True, color=GRID, alpha=0.35, linewidth=0.6)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(SPINE)


def style_secondary_axis(axis) -> None:
    axis.tick_params(colors=TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(SPINE)


def style_colorbar(colorbar) -> None:
    colorbar.ax.tick_params(colors=TEXT)
    colorbar.outline.set_edgecolor(SPINE)
    if colorbar.ax.yaxis.label:
        colorbar.ax.yaxis.label.set_color(TEXT)

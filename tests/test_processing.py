import numpy as np

from vibeels.processing import ensure_supported_eels_signal, fit_zero_loss_peak, process_map_dataset


class FakeAxis:
    def __init__(self, axis):
        self.axis = axis


class FakeAxesManager:
    def __init__(self, axis, signal_dimension=1, navigation_dimension=None):
        self.signal_axes = [FakeAxis(axis)]
        self.signal_dimension = signal_dimension
        self.navigation_dimension = navigation_dimension


class FakeSignal:
    def __init__(self, data, axis, *, signal_dimension=1, navigation_dimension=None):
        self.data = data
        self.axes_manager = FakeAxesManager(axis, signal_dimension, navigation_dimension)


class Signal1D(FakeSignal):
    pass


class Signal2D(FakeSignal):
    pass


def test_fit_zero_loss_peak_recenters_axis():
    axis = np.linspace(-0.1, 0.1, 401)
    shifted_center = 0.012
    intensity = np.exp(-((axis - shifted_center) ** 2) / (2 * 0.004**2)) * 1000
    intensity += 25 + axis * 2

    result = fit_zero_loss_peak(axis, intensity)

    assert abs(result.center_ev - shifted_center) < 0.002
    assert abs(result.calibrated_axis[np.argmax(intensity)]) < 0.003


def test_process_map_dataset_selects_thresholded_roi():
    axis = np.linspace(-0.1, 0.2, 128)
    data = np.zeros((4, 5, axis.size), dtype=float)
    for y in range(4):
        for x in range(5):
            center = 0.008 + (x * 0.0002)
            base = np.exp(-((axis - center) ** 2) / (2 * 0.003**2)) * (100 + x + y)
            data[y, x] = base
    data[0, 0] *= 0.01

    signal = Signal1D(data, axis)
    result = process_map_dataset(
        signal,
        energy_range=(20, 80),
        polygon_mask=np.ones((4, 5), dtype=bool),
        intensity_range=(10.0, 1e9),
    )

    assert result.selected_pixel_count == 19
    assert result.summed_spectrum.shape == axis.shape
    assert result.selected_zlp_centers_ev.shape == (19,)
    assert np.max(np.abs(result.selected_zlp_centers_ev - np.mean(result.selected_zlp_centers_ev))) < 0.01
    assert abs(result.zero_loss_fit.center_ev) < 0.003
    assert result.masked_image.shape == (4, 5)


def test_ensure_supported_eels_signal_accepts_3d_map_signal():
    signal = Signal1D(
        np.zeros((4, 5, 128), dtype=float),
        np.linspace(-0.1, 0.2, 128),
        signal_dimension=1,
        navigation_dimension=2,
    )

    assert ensure_supported_eels_signal(signal) is signal


def test_ensure_supported_eels_signal_accepts_3d_snapshot_signal():
    signal = Signal2D(
        np.zeros((7, 12, 128), dtype=float),
        np.linspace(-0.1, 0.2, 128),
        signal_dimension=2,
        navigation_dimension=1,
    )

    assert ensure_supported_eels_signal(signal) is signal


def test_ensure_supported_eels_signal_rejects_non_3d_input():
    signal = Signal1D(
        np.zeros(128, dtype=float),
        np.linspace(-0.1, 0.2, 128),
        signal_dimension=1,
        navigation_dimension=0,
    )

    try:
        ensure_supported_eels_signal(signal)
    except ValueError as exc:
        assert "Unsupported EELS dataset" in str(exc)
        assert "shape (128,)" in str(exc)
    else:
        raise AssertionError("Expected non-3D input to be rejected.")


def test_ensure_supported_eels_signal_rejects_multiple_loaded_signals():
    signals = [
        Signal1D(np.zeros((4, 5, 128), dtype=float), np.linspace(-0.1, 0.2, 128)),
        Signal1D(np.zeros((4, 5, 128), dtype=float), np.linspace(-0.1, 0.2, 128)),
    ]

    try:
        ensure_supported_eels_signal(signals)
    except ValueError as exc:
        assert "multiple signals" in str(exc)
    else:
        raise AssertionError("Expected multi-signal input to be rejected.")

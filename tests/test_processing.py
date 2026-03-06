import numpy as np

from vibeels.processing import fit_zero_loss_peak, process_map_dataset


class FakeAxis:
    def __init__(self, axis):
        self.axis = axis


class FakeAxesManager:
    def __init__(self, axis):
        self.signal_axes = [FakeAxis(axis)]


class FakeSignal:
    def __init__(self, data, axis):
        self.data = data
        self.axes_manager = FakeAxesManager(axis)


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

    signal = FakeSignal(data, axis)
    result = process_map_dataset(
        signal,
        energy_range=(20, 80),
        polygon_mask=np.ones((4, 5), dtype=bool),
        intensity_range=(10.0, 1e9),
    )

    assert result.selected_pixel_count == 19
    assert result.summed_spectrum.shape == axis.shape
    assert abs(result.zero_loss_fit.center_ev - 0.0085) < 0.01
    assert result.masked_image.shape == (4, 5)

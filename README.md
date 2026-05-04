# vibeels

`vibeels` is a PyQt6 application for vibrational EELS spectrum extraction and zero-loss peak calibration using HyperSpy.

## Status

`vibeels` is under active development. Review results carefully before using them in published analysis.

## Features

- `Signal1D` map workflow with multiple polygon ROIs and intensity masking
- `Signal2D` snapshot workflow with detector-band extraction and alignment
- Zero-loss peak fitting and calibrated spectrum display
- Export of spectrum data, parameters, and reproducibility bundle
- Session saving and restoration for reproducible analysis workflows

## Installation

### From PyPI

```bash
python -m pip install vibeels
```

### From source

```bash
git clone https://github.com/danshim/vibeels.git
cd vibeels
python -m pip install -e .
```

For a conda-based workflow, create and activate an environment first:

```bash
conda create -n vibeels python=3.11
conda activate vibeels
python -m pip install -e .
```

## How to use

1. Launch the application and load an EELS dataset with `Load EELS DM3/DM4`.
2. If needed, load a reference image in the `Image` tab to help define the region of interest.
3. For `Signal1D` data, use the `2D Map` tab:
   - draw one or more polygon ROIs over the areas you want to analyze
   - adjust the map energy range and intensity mask as needed
   - click `Apply ROI / Alignment` to extract and align the spectra
4. For `Signal2D` data, use the `1D Spot` tab:
   - choose the detector row range that contains the zero-loss peak signal
   - set the snapshot range to include the frames you want to process
   - click `Apply ROI / Alignment` to align the snapshots and build the summed spectrum
5. Inspect the `Calibration` tab to review the ZLP fit, FWHM, calibrated spectrum, and preview plots. If the alignment looks off, adjust the ROI or fit window and process again.
6. Export the processed data when you are satisfied with the result, or save the current session state for later review.

## Run locally

```bash
python -m vibeels
```

If installed from the package entry point:

```bash
vibeels
```

## Build distribution

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Citation

If you use `vibeels` in research, cite the software using the metadata in [`CITATION.cff`](CITATION.cff).

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).

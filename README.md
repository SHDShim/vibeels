# vibeels

[![DOI](https://zenodo.org/badge/1174978017.svg)](https://doi.org/10.5281/zenodo.20018423)

`vibeels` is a PyQt6 application for extracting and calibrating vibrational electron energy-loss spectroscopy (EELS) spectra. It supports map-based and snapshot-stack workflows, with zero-loss peak (ZLP) alignment and calibrated spectrum export.

## Status

`vibeels` is under active development. Review results carefully before using them in published analysis.

## Features

- `Signal1D` 2D map workflow with multiple polygon ROIs.
- `Signal2D` snapshot-stack workflow with a single rectangular detector-band ROI.
- Shared plot controls for zoom, ROI editing, zoom-out, and ROI clearing.
- Intensity masking using a full-map histogram for 2D map data.
- Per-spectrum ZLP fitting, alignment, and calibrated summed spectra.
- Raw and aligned snapshot previews with independent zoom control.
- Export of spectra, masks, figures, parameters, and reproducibility bundles.
- Session saving and restoration.
- Runtime application icon support on macOS and Windows.

## Installation

### From PyPI

After the package is published to PyPI:

```bash
python -m pip install vibeels
```

### From source

```bash
git clone https://github.com/shdshim/vibeels.git
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
2. If needed, load a reference image in the `Image` tab.
3. Use the shared plot controls above the figures:
   - `Zoom`: mouse zoom behavior for all four main plots.
   - `ROI`: ROI editing on the top-left raw-data plot.
   - `Zoom Out`: reset zoom for all four main plots.
   - `Clear ROI`: remove the active ROI selection.
4. For `Signal1D` map data, use the `2D Map` tab:
   - draw one or more polygon ROIs on the top-left map preview.
   - adjust the map energy range.
   - use the full-map intensity histogram to set the mask range.
   - click `Apply ROI / Alignment` to extract, align, and sum all spectra inside any polygon and inside the intensity range.
5. For `Signal2D` snapshot-stack data, use the `1D Spot` tab:
   - draw or set one rectangular detector-y ROI.
   - set the snapshot range and snapshot index.
   - click `Apply ROI / Alignment` to align snapshots and build the summed spectrum.
6. Inspect the `Calibration` tab to review the ZLP fit, FWHM, calibrated spectrum, and previews.
7. Save selected spectra in the `Saved` tab, export the processed bundle, or save the session state for later review.

## ROI and Plot Controls

- 2D map mode allows multiple polygon ROIs. Pixels inside any polygon are included.
- 1D snapshot mode allows one rectangular detector-y ROI.
- In 2D map mode, the intensity histogram always represents the full integrated image, not only the ROI pixels.
- Right-click in `Zoom` mode resets only the clicked plot.
- Right-click inside an ROI in `ROI` mode clears that ROI.
- Zoom state is preserved during ROI changes and snapshot navigation unless `Zoom Out` or right-click zoom reset is used.

## Outputs

Exports include:

- calibrated spectra
- selected masks and preview images
- ZLP fit data
- processing parameters
- reproducibility scripts and figures
- saved session state

## Run

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

## Release Metadata

- Package name: `vibeels`
- Current version: `0.3.1`
- Repository: <https://github.com/shdshim/vibeels>
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

## Citation

If you use `vibeels` in research, cite the software using the metadata in [`CITATION.cff`](CITATION.cff).

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).

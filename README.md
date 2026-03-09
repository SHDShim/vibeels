# vibeels

`vibeels` is a PyQt6 application for vibrational EELS spectrum extraction and zero-loss peak calibration using HyperSpy.

## Features

- `Signal1D` map workflow with polygon ROI and intensity masking
- `Signal2D` snapshot workflow with detector-band extraction and alignment
- Zero-loss peak fitting and calibrated spectrum display
- Export of spectrum data, parameters, and reproducibility bundle

## How to use

1. Launch the application and load an EELS dataset with `Load EELS DM3/DM4`.
2. If needed, load a reference image in the `Image` tab to help define the region of interest.
3. For `Signal1D` data, use the `2D Map` tab:
   - draw a polygon ROI over the area you want to analyze
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

## Build distribution

```bash
python -m build
```

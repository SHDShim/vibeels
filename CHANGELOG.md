# Changelog

All notable changes to `vibeels` are documented here.

## 0.3.1 - 2026-05-05

### Changed

- Updated button labels for clearer application workflow guidance.

## 0.3.0 - 2026-05-04

### Added

- Shared plot interaction controls for zoom, ROI selection, zoom-out, and ROI clearing.
- Multiple polygon ROI support for 2D map workflows.
- Runtime application icon support and generated Windows/macOS icon assets.
- Session and export handling for multi-polygon map ROIs.

### Changed

- 2D map histogram now reflects the full integrated map image rather than ROI-only pixels.
- Snapshot alignment preview now updates only the aligned-image panel, leaving the raw snapshot panel unchanged.
- Snapshot alignment suppresses horizontal energy-axis displacement during 2D frame registration.
- Empty plot messages now wrap to avoid clipping in narrow panels.

### Fixed

- Preserved snapshot zoom state during ROI changes, tab changes, and snapshot navigation.
- Restored reliable zoom-out behavior for snapshot image panels.
- Avoided HyperSpy all-zero-shift alignment warnings by skipping no-op 2D alignment.

## 0.2.0

- Prototype release.

## 0.1.0

- Initial NumPy-based spectral alignment workflow.

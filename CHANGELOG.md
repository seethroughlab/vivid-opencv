# Changelog

All notable changes to vivid-opencv will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.2] - 2026-01-13

### Changed

- **CPU-only architecture**: Operators now output CPU pixels instead of GPU textures
  - Removed all WebGPU code from operators (~200 lines)
  - vivid-core handles GPU↔CPU conversion via new `OutputKind::CpuPixels`
  - Simpler, more maintainable operator implementations
- **No wgpu library linking**: Only WebGPU headers needed (for base class compatibility)
- Removed `texture_converter.cpp` and `texture_converter.h` (no longer needed)

### Fixed

- Build system simplified - no need to download wgpu separately on Windows

## [0.1.0-alpha.1] - 2026-01-14

Initial alpha release of the Vivid OpenCV addon.

### Added

- **Contours** operator for edge detection using Canny algorithm and contour extraction
- **OpticalFlow** operator for dense motion vector calculation using Farneback's algorithm
- **BlobTrack** operator for blob detection and tracking using SimpleBlobDetector
- Builds OpenCV 4.10.0 from source to avoid MSVC STL ABI issues on Windows
- Cross-platform CI builds (macOS arm64/x64, Linux x64, Windows x64)
- Automated release workflow triggered by version tags
- Examples: contours-webcam, contours-video, optical-flow, blob-tracking

### Notes

- Extracted from main vivid repository to enable independent CI and releases
- Previously disabled on Windows due to opencv-mobile prebuilt ABI incompatibility
- Now builds OpenCV from source, enabling full Windows support

# vivid-opencv

OpenCV computer vision module for Vivid. Builds OpenCV from source to avoid MSVC STL ABI issues on Windows.

## Build Commands

```bash
# Build with vivid SDK
cmake -B build -DVIVID_ROOT=/path/to/vivid-sdk
cmake --build build

# Build with tests
cmake -B build -DVIVID_ROOT=/path/to/vivid-sdk -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Project Structure

```
vivid-opencv/
├── include/vivid/opencv/     # Public headers
│   ├── opencv.h              # Main include (all operators)
│   ├── contours.h            # Contour detection
│   ├── optical_flow.h        # Motion detection
│   ├── blob_track.h          # Blob tracking
│   └── export.h              # DLL export macros
├── src/                      # Implementation
│   ├── opencv.cpp            # Module entry point
│   ├── contours.cpp          # Contours operator + REGISTER_OPERATOR
│   ├── optical_flow.cpp      # OpticalFlow operator + REGISTER_OPERATOR
│   └── blob_track.cpp        # BlobTrack operator + REGISTER_OPERATOR
├── examples/                 # Runnable demos
│   ├── contours-webcam/
│   ├── contours-video/
│   ├── optical-flow/
│   └── blob-tracking/
├── .github/workflows/        # CI/CD
│   ├── ci.yml                # Build & test on push
│   └── release.yml           # Multi-platform releases
├── CMakeLists.txt            # Build config (builds OpenCV from source)
├── module.json               # Vivid module metadata
└── README.md                 # User documentation
```

## Key Files

| Task | File |
|------|------|
| Add new operator | Create header in `include/`, impl in `src/`, add REGISTER_OPERATOR |
| Modify Contours | `include/vivid/opencv/contours.h`, `src/contours.cpp` |
| Fix build issues | `CMakeLists.txt` |
| Update CI | `.github/workflows/ci.yml` |

## Code Patterns

### Operator Registration
Each operator uses `REGISTER_OPERATOR` at the end of its .cpp file:
```cpp
using OpenCVContours = vivid::opencv::Contours;
REGISTER_OPERATOR(OpenCVContours, "OpenCV", "Description", true);
```

### PIMPL Pattern
OpenCV types are hidden from headers using PIMPL:
```cpp
// In header
struct Impl;
std::unique_ptr<Impl> m_impl;

// In cpp
struct Contours::Impl {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat lastInput;
};
```

### CPU Pixel Access
Operators use `cpuPixelView()` for zero-copy access to CPU pixels:
```cpp
auto cpuView = inputOp->cpuPixelView();
if (!cpuView.valid()) {
    didCook();
    return;  // Input doesn't provide CPU pixels
}
cv::Mat input(height, width, CV_8UC4, const_cast<uint8_t*>(cpuView.data));
```

## OpenCV Build Notes

This module builds OpenCV 4.10.0 from source with minimal modules:
- `core` - Basic data structures
- `imgproc` - Image processing (Canny, cvtColor, etc.)
- `video` - Video analysis (optical flow)
- `features2d` - Feature detection (blob detector)

All GUI, codec, and capture modules are disabled to minimize build size.

## Platform Notes

### Windows
- Builds OpenCV from source to avoid MSVC STL ABI incompatibility
- The original opencv-mobile prebuilts used MSVC 14.3x, causing linker errors with MSVC 14.43+
- Error was: `LNK2019: unresolved external symbol __std_find_first_of_trivial_pos_1`

### macOS
- Uses Accelerate framework for optimized BLAS operations
- Uses `-undefined dynamic_lookup` for symbol resolution at load time

### Linux
- Links against system zlib, pthread, dl
- Uses `--allow-shlib-undefined` for deferred symbol resolution

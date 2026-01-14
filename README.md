# vivid-opencv

OpenCV computer vision integration for [Vivid](https://github.com/seethroughlab/vivid).

## Features

- **Contours** - Edge detection and contour drawing using Canny algorithm
- **OpticalFlow** - Dense motion vector calculation using Farneback's algorithm
- **BlobTrack** - Blob detection and tracking using SimpleBlobDetector

## Installation

```bash
vivid modules install https://github.com/seethroughlab/vivid-opencv
```

### Supported Platforms

- macOS (arm64, x86_64)
- Linux (Ubuntu 22.04+)
- Windows (x64)

**Note:** This module builds OpenCV from source to ensure compatibility across all platforms, avoiding MSVC STL ABI issues that occur with prebuilt binaries.

## Usage

```cpp
#include <vivid/vivid.h>
#include <vivid/opencv/opencv.h>

void setup(Context& ctx) {
    auto& chain = ctx.chain();

    // Webcam input (provides CPU pixels for OpenCV)
    auto& cam = chain.add<vivid::video::Webcam>("cam");

    // Detect contours
    auto& contours = chain.add<vivid::opencv::Contours>("contours");
    contours.input("cam");
    contours.threshold1 = 50.0f;   // Canny threshold 1
    contours.threshold2 = 150.0f;  // Canny threshold 2
    contours.lineWidth = 2.0f;
    contours.colorG = 1.0f;        // Green contours

    chain.output("contours");
}
```

## Operators

### Contours

Detects edges using Canny algorithm and extracts contours.

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| threshold1 | float | 0-255 | 100 | Canny first threshold |
| threshold2 | float | 0-255 | 200 | Canny second threshold |
| mode | int | 0-3 | 0 | Retrieval mode (0=External, 1=List, 2=CComp, 3=Tree) |
| lineWidth | float | 1-20 | 2 | Contour line thickness |
| colorR/G/B/A | float | 0-1 | 0,1,0,1 | Contour color (green default) |

### OpticalFlow

Calculates motion vectors between consecutive frames.

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| scale | float | 0.05-1.0 | 0.15 | Processing resolution scale |
| pyrScale | float | 0.1-0.9 | 0.5 | Pyramid scale factor |
| levels | int | 1-5 | 1 | Number of pyramid levels |
| winSize | int | 3-25 | 9 | Averaging window size |
| iterations | int | 1-10 | 1 | Iterations per level |
| vizMode | int | 0-2 | 0 | Visualization (0=HSV, 1=Arrows, 2=Magnitude) |
| sensitivity | float | 0.1-10 | 1.0 | Motion sensitivity |

### BlobTrack

Detects circular blobs based on size, color, and shape.

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| minArea | float | 10-10000 | 100 | Minimum blob area |
| maxArea | float | 100-100000 | 50000 | Maximum blob area |
| minCircularity | float | 0-1 | 0.1 | Minimum circularity |
| minConvexity | float | 0-1 | 0.5 | Minimum convexity |
| detectBright | int | 0-1 | 1 | Detect bright blobs |
| detectDark | int | 0-1 | 1 | Detect dark blobs |
| threshold | float | 0-255 | 128 | Binarization threshold |

## Examples

### contours-webcam
Real-time contour detection from webcam with interactive controls.
```bash
vivid examples/contours-webcam
```

### contours-video
Contour detection on video files with overlay blending.
```bash
vivid examples/contours-video
```

### optical-flow
Dense optical flow motion detection.
```bash
vivid examples/optical-flow
```

### blob-tracking
Blob detection and tracking.
```bash
vivid examples/blob-tracking
```

## Architecture Notes

OpenCV operators require **CPU pixel data** from the input operator via the `cpuPixelView()` interface. This avoids expensive GPU→CPU readback.

**Compatible input sources:**
- `Webcam` - provides CPU pixels from camera capture
- `VideoPlayer` - provides CPU pixels from video decoding

**Incompatible sources:**
- GPU-only operators (shaders, effects) - these only have GPU textures

## Building from Source

```bash
# Clone the repository
git clone https://github.com/seethroughlab/vivid-opencv.git
cd vivid-opencv

# Build with vivid SDK
cmake -B build -DVIVID_ROOT=/path/to/vivid-sdk
cmake --build build

# Or build as part of main vivid repo (place in modules/)
```

## License

MIT License - See [LICENSE](LICENSE) for details.

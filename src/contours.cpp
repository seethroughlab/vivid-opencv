/**
 * @file contours.cpp
 * @brief Contour detection operator implementation
 */

#include <vivid/opencv/contours.h>
#include <vivid/context.h>
#include <vivid/chain.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vivid::opencv {

// PIMPL implementation - hides OpenCV types from header
struct Contours::Impl {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat lastInput;
};

Contours::Contours() : m_impl(std::make_unique<Impl>()) {
    registerParam(threshold1);
    registerParam(threshold2);
    registerParam(mode);
    registerParam(lineWidth);
    registerParam(colorR);
    registerParam(colorG);
    registerParam(colorB);
    registerParam(colorA);
}

Contours::~Contours() = default;

void Contours::cleanup() {
    m_outputPixels.clear();
    m_outputWidth = 0;
    m_outputHeight = 0;
}

void Contours::init(Context& ctx) {
    // Try to match input resolution
    matchInputResolution(0);
}

Operator::CpuPixelView Contours::cpuPixelView() const {
    if (m_outputPixels.empty() || m_outputWidth <= 0 || m_outputHeight <= 0) {
        return {};
    }
    return {m_outputPixels.data(), m_outputWidth, m_outputHeight, 4, 0};
}

void Contours::process(Context& ctx) {
    // Check if we need to process
    if (!needsCook()) {
        return;
    }

    // Get input operator
    Operator* inputOp = getInput(0);
    if (!inputOp) {
        didCook();
        return;
    }

    // Use zero-copy view for CPU pixels
    auto cpuView = inputOp->cpuPixelView();
    if (!cpuView.valid()) {
        // Input doesn't provide CPU pixels - skip processing
        // This is by design: OpenCV operators require CPU pixel sources
        didCook();
        return;
    }

    int width = cpuView.width;
    int height = cpuView.height;

    // Skip if too small to process
    if (width < 16 || height < 16) {
        didCook();
        return;
    }

    // Create cv::Mat from CPU pixel data (BGRA format from VideoPlayer/Webcam) - zero-copy
    cv::Mat input(height, width, CV_8UC4, const_cast<uint8_t*>(cpuView.data));

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);

    // Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(gray, edges,
              static_cast<double>(threshold1),
              static_cast<double>(threshold2));

    // Find contours
    m_impl->contours.clear();
    int cvMode = cv::RETR_EXTERNAL;
    switch (static_cast<int>(mode)) {
        case 0: cvMode = cv::RETR_EXTERNAL; break;
        case 1: cvMode = cv::RETR_LIST; break;
        case 2: cvMode = cv::RETR_CCOMP; break;
        case 3: cvMode = cv::RETR_TREE; break;
    }
    cv::findContours(edges, m_impl->contours, cvMode, cv::CHAIN_APPROX_SIMPLE);

    // Create output image with transparent background
    cv::Mat output(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // Draw contours
    // OpenCV uses BGR, but our Mat is BGRA, and color params are RGB
    cv::Scalar color(
        static_cast<int>(static_cast<float>(colorB) * 255),  // B
        static_cast<int>(static_cast<float>(colorG) * 255),  // G
        static_cast<int>(static_cast<float>(colorR) * 255),  // R
        static_cast<int>(static_cast<float>(colorA) * 255)   // A
    );

    int thickness = static_cast<int>(static_cast<float>(lineWidth));
    if (thickness < 1) thickness = 1;

    cv::drawContours(output, m_impl->contours, -1, color, thickness);

    // Store output in CPU pixel buffer
    m_outputWidth = width;
    m_outputHeight = height;
    size_t dataSize = output.total() * output.elemSize();
    m_outputPixels.assign(output.data, output.data + dataSize);

    didCook();
}

size_t Contours::contourCount() const {
    return m_impl->contours.size();
}

} // namespace vivid::opencv

// Alias for registration macro (must be outside namespace)
using OpenCVContours = vivid::opencv::Contours;

// Register operator
REGISTER_OPERATOR(OpenCVContours, "OpenCV", "Detect and draw contours using Canny edge detection", true);

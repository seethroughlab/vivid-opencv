/**
 * @file blob_track.cpp
 * @brief Blob detection operator implementation
 */

#include <vivid/opencv/blob_track.h>
#include <vivid/context.h>
#include <vivid/chain.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <cmath>

namespace vivid::opencv {

// PIMPL - hides OpenCV types from header
struct BlobTrack::Impl {
    cv::Ptr<cv::SimpleBlobDetector> detector;
    std::vector<cv::KeyPoint> keypoints;

    // Cached detector params to detect when we need to recreate
    float lastMinArea = -1;
    float lastMaxArea = -1;
    float lastMinCircularity = -1;
    float lastMinConvexity = -1;
    float lastMinInertia = -1;
    int lastDetectBright = -1;
    int lastDetectDark = -1;
    float lastThreshold = -1;
};

BlobTrack::BlobTrack() : m_impl(std::make_unique<Impl>()) {
    registerParam(minArea);
    registerParam(maxArea);
    registerParam(minCircularity);
    registerParam(minConvexity);
    registerParam(minInertia);
    registerParam(detectBright);
    registerParam(detectDark);
    registerParam(threshold);
}

BlobTrack::~BlobTrack() = default;

void BlobTrack::cleanup() {
    m_outputPixels.clear();
    m_outputWidth = 0;
    m_outputHeight = 0;
    m_impl->detector.release();
    m_impl->keypoints.clear();
}

void BlobTrack::init(Context& ctx) {
    matchInputResolution(0);
}

Operator::CpuPixelView BlobTrack::cpuPixelView() const {
    if (m_outputPixels.empty() || m_outputWidth <= 0 || m_outputHeight <= 0) {
        return {};
    }
    return {m_outputPixels.data(), m_outputWidth, m_outputHeight, 4, 0};
}

void BlobTrack::process(Context& ctx) {
    if (!needsCook()) {
        return;
    }

    Operator* inputOp = getInput(0);
    if (!inputOp) {
        didCook();
        return;
    }

    auto cpuView = inputOp->cpuPixelView();
    if (!cpuView.valid()) {
        didCook();
        return;
    }

    int width = cpuView.width;
    int height = cpuView.height;

    if (width < 16 || height < 16) {
        didCook();
        return;
    }

    // Check if detector params changed - recreate detector if needed
    bool paramsChanged =
        m_impl->lastMinArea != static_cast<float>(minArea) ||
        m_impl->lastMaxArea != static_cast<float>(maxArea) ||
        m_impl->lastMinCircularity != static_cast<float>(minCircularity) ||
        m_impl->lastMinConvexity != static_cast<float>(minConvexity) ||
        m_impl->lastMinInertia != static_cast<float>(minInertia) ||
        m_impl->lastDetectBright != static_cast<int>(detectBright) ||
        m_impl->lastDetectDark != static_cast<int>(detectDark) ||
        m_impl->lastThreshold != static_cast<float>(threshold);

    if (!m_impl->detector || paramsChanged) {
        cv::SimpleBlobDetector::Params params;

        // Threshold parameters
        params.minThreshold = static_cast<float>(threshold) - 50;
        params.maxThreshold = static_cast<float>(threshold) + 50;
        params.thresholdStep = 10;

        // Area filter
        params.filterByArea = true;
        params.minArea = static_cast<float>(minArea);
        params.maxArea = static_cast<float>(maxArea);

        // Circularity filter
        params.filterByCircularity = static_cast<float>(minCircularity) > 0.01f;
        params.minCircularity = static_cast<float>(minCircularity);

        // Convexity filter
        params.filterByConvexity = static_cast<float>(minConvexity) > 0.01f;
        params.minConvexity = static_cast<float>(minConvexity);

        // Inertia filter (elongation)
        params.filterByInertia = static_cast<float>(minInertia) > 0.01f;
        params.minInertiaRatio = static_cast<float>(minInertia);

        // Color filter
        params.filterByColor = true;
        if (static_cast<int>(detectBright) && !static_cast<int>(detectDark)) {
            params.blobColor = 255;  // Bright blobs only
        } else if (!static_cast<int>(detectBright) && static_cast<int>(detectDark)) {
            params.blobColor = 0;    // Dark blobs only
        } else {
            params.filterByColor = false;  // Both
        }

        m_impl->detector = cv::SimpleBlobDetector::create(params);

        // Cache params
        m_impl->lastMinArea = static_cast<float>(minArea);
        m_impl->lastMaxArea = static_cast<float>(maxArea);
        m_impl->lastMinCircularity = static_cast<float>(minCircularity);
        m_impl->lastMinConvexity = static_cast<float>(minConvexity);
        m_impl->lastMinInertia = static_cast<float>(minInertia);
        m_impl->lastDetectBright = static_cast<int>(detectBright);
        m_impl->lastDetectDark = static_cast<int>(detectDark);
        m_impl->lastThreshold = static_cast<float>(threshold);
    }

    // Create cv::Mat from CPU pixels (zero-copy)
    cv::Mat input(height, width, CV_8UC4, const_cast<uint8_t*>(cpuView.data));

    // Convert to grayscale for blob detection
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);

    // Detect blobs
    m_impl->keypoints.clear();
    m_impl->detector->detect(gray, m_impl->keypoints);

    // Create output with visualization
    cv::Mat output;
    input.copyTo(output);

    // Threshold image to find contours
    cv::Mat binary;
    float thresh = static_cast<float>(threshold);
    if (static_cast<int>(detectBright) && !static_cast<int>(detectDark)) {
        cv::threshold(gray, binary, thresh, 255, cv::THRESH_BINARY);
    } else if (!static_cast<int>(detectBright) && static_cast<int>(detectDark)) {
        cv::threshold(gray, binary, thresh, 255, cv::THRESH_BINARY_INV);
    } else {
        // For both, use regular threshold
        cv::threshold(gray, binary, thresh, 255, cv::THRESH_BINARY);
    }

    // Find contours for visualization
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours that match detected blob locations
    float minA = static_cast<float>(minArea);
    float maxA = static_cast<float>(maxArea);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= minA && area <= maxA) {
            // Draw the contour outline
            cv::drawContours(output, std::vector<std::vector<cv::Point>>{contour}, 0,
                           cv::Scalar(0, 255, 0, 255), 2, cv::LINE_AA);
        }
    }

    // Draw blob centers from keypoints
    for (const auto& kp : m_impl->keypoints) {
        int x = static_cast<int>(kp.pt.x);
        int y = static_cast<int>(kp.pt.y);
        int radius = static_cast<int>(kp.size / 2);

        // Draw bounding circle (yellow)
        cv::circle(output, cv::Point(x, y), radius, cv::Scalar(0, 255, 255, 200), 2, cv::LINE_AA);

        // Draw center crosshair (magenta)
        int cross = 8;
        cv::line(output, cv::Point(x - cross, y), cv::Point(x + cross, y),
                 cv::Scalar(255, 0, 255, 255), 2, cv::LINE_AA);
        cv::line(output, cv::Point(x, y - cross), cv::Point(x, y + cross),
                 cv::Scalar(255, 0, 255, 255), 2, cv::LINE_AA);
    }

    // Store output in CPU pixel buffer (BGRA format)
    m_outputWidth = width;
    m_outputHeight = height;
    size_t dataSize = output.total() * output.elemSize();
    m_outputPixels.assign(output.data, output.data + dataSize);

    didCook();
}

} // namespace vivid::opencv

using OpenCVBlobTrack = vivid::opencv::BlobTrack;
REGISTER_OPERATOR(OpenCVBlobTrack, "OpenCV", "Blob detection and tracking", true);

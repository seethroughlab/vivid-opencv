cmake_minimum_required(VERSION 3.16)
project(vivid-opencv VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vivid OpenCV Addon
# Computer vision operators using OpenCV built from source
# - Contours: Edge detection and contour extraction
# - OpticalFlow: Dense motion vector calculation
# - BlobTrack: Blob detection and tracking
#
# This module builds OpenCV from source to avoid MSVC STL ABI incompatibilities
# that occur with opencv-mobile prebuilt binaries on Windows.

include(FetchContent)

# -----------------------------------------------------------------------------
# Find vivid headers (required dependency)
# -----------------------------------------------------------------------------
# vivid-opencv is a standalone addon that only needs vivid headers at compile time.
# Symbols from vivid-core are resolved at load time when the addon is loaded
# by the vivid runtime.
#
# Options:
#   1. As subdirectory of main vivid repo (TARGET vivid-core exists)
#   2. Standalone build with VIVID_ROOT pointing to vivid SDK
#   3. Standalone build with find_package(vivid)

set(VIVID_BUILD_MODE "standalone")

if(TARGET vivid-core)
    # Building as part of the main vivid repo
    set(VIVID_BUILD_MODE "in-tree")
    message(STATUS "[vivid-opencv] Building in-tree (vivid-core target exists)")
elseif(DEFINED VIVID_ROOT)
    # Standalone build with SDK path or source tree
    message(STATUS "[vivid-opencv] Using VIVID_ROOT: ${VIVID_ROOT}")

    # Check for SDK layout (include/vivid) or source tree layout (modules/vivid-core/include/vivid)
    if(EXISTS "${VIVID_ROOT}/include/vivid/vivid.h")
        # SDK package layout - all headers bundled in include/
        set(VIVID_INCLUDE_DIR "${VIVID_ROOT}/include")
        set(VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/include")
        set(VIVID_SDK_BUILD TRUE)
    elseif(EXISTS "${VIVID_ROOT}/modules/vivid-core/include/vivid/vivid.h")
        # Source tree layout - headers spread across build/_deps
        set(VIVID_INCLUDE_DIR "${VIVID_ROOT}/modules/vivid-core/include")

        # Find dependency headers in build/_deps
        set(VIVID_DEP_INCLUDE_DIRS "")

        # WebGPU headers (needed by operator.h, but we don't link against wgpu)
        if(EXISTS "${VIVID_ROOT}/build/_deps/wgpu/include/webgpu/webgpu.h")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/wgpu/include")
        else()
            message(FATAL_ERROR "[vivid-opencv] WebGPU headers not found. Please build vivid first:\n"
                "  cd ${VIVID_ROOT} && cmake -B build && cmake --build build")
        endif()

        # GLM
        if(EXISTS "${VIVID_ROOT}/build/_deps/glm-src/glm/glm.hpp")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/glm-src")
        else()
            message(FATAL_ERROR "[vivid-opencv] GLM headers not found. Please build vivid first.")
        endif()

        # GLFW
        if(EXISTS "${VIVID_ROOT}/build/_deps/glfw-src/include/GLFW/glfw3.h")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/glfw-src/include")
        else()
            message(FATAL_ERROR "[vivid-opencv] GLFW headers not found. Please build vivid first.")
        endif()

        # nlohmann_json
        if(EXISTS "${VIVID_ROOT}/build/_deps/nlohmann_json-src/include/nlohmann/json.hpp")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/nlohmann_json-src/include")
        else()
            message(FATAL_ERROR "[vivid-opencv] nlohmann_json headers not found. Please build vivid first.")
        endif()

        # magic_enum
        if(EXISTS "${VIVID_ROOT}/build/_deps/magic_enum-src/include/magic_enum/magic_enum.hpp")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/magic_enum-src/include")
        else()
            message(FATAL_ERROR "[vivid-opencv] magic_enum headers not found. Please build vivid first.")
        endif()
    else()
        message(FATAL_ERROR "[vivid-opencv] VIVID_ROOT specified but headers not found.\n"
            "  Checked: ${VIVID_ROOT}/include/vivid/vivid.h\n"
            "  Checked: ${VIVID_ROOT}/modules/vivid-core/include/vivid/vivid.h")
    endif()
    message(STATUS "[vivid-opencv] Found vivid headers at: ${VIVID_INCLUDE_DIR}")
    message(STATUS "[vivid-opencv] Dependency include dirs: ${VIVID_DEP_INCLUDE_DIRS}")

    # On Windows, we need the import library for linking
    if(WIN32)
        find_library(VIVID_CORE_LIB vivid-core PATHS "${VIVID_ROOT}/lib" REQUIRED)
        message(STATUS "[vivid-opencv] Found vivid-core import library: ${VIVID_CORE_LIB}")
    endif()
else()
    # Try find_package
    find_package(vivid QUIET)
    if(vivid_FOUND)
        set(VIVID_BUILD_MODE "installed")
        message(STATUS "[vivid-opencv] Found vivid via find_package")
    else()
        message(FATAL_ERROR
            "[vivid-opencv] Could not find vivid headers. Options:\n"
            "  1. Set VIVID_ROOT to your vivid SDK installation\n"
            "  2. Install vivid and ensure it's in CMAKE_PREFIX_PATH\n"
            "  3. Build as part of the main vivid repo"
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# OpenCV dependency - BUILD FROM SOURCE
# -----------------------------------------------------------------------------
# We build OpenCV from source to avoid MSVC STL ABI incompatibilities
# with opencv-mobile prebuilt binaries on Windows.

set(OPENCV_VERSION "4.10.0")

message(STATUS "[vivid-opencv] Building OpenCV ${OPENCV_VERSION} from source...")

# Configure OpenCV build options for minimal size
set(BUILD_LIST "core,imgproc,video,features2d" CACHE STRING "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_DOCS OFF CACHE BOOL "")
set(BUILD_opencv_apps OFF CACHE BOOL "")
set(BUILD_opencv_python3 OFF CACHE BOOL "")
set(BUILD_opencv_python2 OFF CACHE BOOL "")
set(BUILD_opencv_java OFF CACHE BOOL "")
set(BUILD_opencv_js OFF CACHE BOOL "")
set(BUILD_JAVA OFF CACHE BOOL "")
set(BUILD_FAT_JAVA_LIB OFF CACHE BOOL "")
set(WITH_CUDA OFF CACHE BOOL "")
set(WITH_OPENCL OFF CACHE BOOL "")
set(WITH_OPENGL OFF CACHE BOOL "")
set(WITH_GTK OFF CACHE BOOL "")
set(WITH_QT OFF CACHE BOOL "")
set(WITH_VTK OFF CACHE BOOL "")
set(WITH_MATLAB OFF CACHE BOOL "")
set(WITH_GSTREAMER OFF CACHE BOOL "")
set(WITH_FFMPEG OFF CACHE BOOL "")
set(WITH_V4L OFF CACHE BOOL "")
set(WITH_DSHOW OFF CACHE BOOL "")
set(WITH_MSMF OFF CACHE BOOL "")
set(WITH_AVFOUNDATION OFF CACHE BOOL "")
set(WITH_1394 OFF CACHE BOOL "")
set(WITH_OPENEXR OFF CACHE BOOL "")
set(WITH_WEBP OFF CACHE BOOL "")
set(WITH_JASPER OFF CACHE BOOL "")
set(WITH_OPENJPEG OFF CACHE BOOL "")
set(WITH_TIFF OFF CACHE BOOL "")
set(WITH_PNG OFF CACHE BOOL "")
set(WITH_JPEG OFF CACHE BOOL "")
set(WITH_IMGCODEC_HDR OFF CACHE BOOL "")
set(WITH_IMGCODEC_SUNRASTER OFF CACHE BOOL "")
set(WITH_IMGCODEC_PXM OFF CACHE BOOL "")
set(WITH_IMGCODEC_PFM OFF CACHE BOOL "")
set(WITH_LAPACK OFF CACHE BOOL "")
set(WITH_EIGEN OFF CACHE BOOL "")
set(WITH_PROTOBUF OFF CACHE BOOL "")
set(WITH_QUIRC OFF CACHE BOOL "")
set(WITH_IPP OFF CACHE BOOL "")
set(WITH_ITT OFF CACHE BOOL "")
set(WITH_HALIDE OFF CACHE BOOL "")
set(WITH_VULKAN OFF CACHE BOOL "")
set(WITH_INF_ENGINE OFF CACHE BOOL "")
set(WITH_NGRAPH OFF CACHE BOOL "")
set(WITH_ONNX OFF CACHE BOOL "")
set(WITH_OBSENSOR OFF CACHE BOOL "")
set(BUILD_opencv_videoio OFF CACHE BOOL "")
set(ENABLE_CXX11 ON CACHE BOOL "")
set(OPENCV_GENERATE_PKGCONFIG OFF CACHE BOOL "")
set(CV_TRACE OFF CACHE BOOL "")

# Platform-specific optimizations
if(APPLE)
    set(WITH_ACCELERATE ON CACHE BOOL "")
    set(WITH_OPENCL_SVM OFF CACHE BOOL "")
    # Check if cross-compiling for x86_64 on ARM64
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        set(ENABLE_NEON OFF CACHE BOOL "")
        set(CPU_BASELINE "" CACHE STRING "")
        set(CPU_BASELINE_REQUIRE "" CACHE STRING "")
    else()
        set(ENABLE_NEON ON CACHE BOOL "")
    endif()
else()
    set(WITH_ACCELERATE OFF CACHE BOOL "")
endif()

if(WIN32)
    # Avoid Windows-specific video capture we don't need
    set(WITH_WIN32UI OFF CACHE BOOL "")
    # Use dynamic runtime (/MD) to match vivid-core
    set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "")
endif()

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG ${OPENCV_VERSION}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# Suppress OpenCV's verbose output
set(OPENCV_DISABLE_FILESYSTEM_SUPPORT ON CACHE BOOL "")
set(OPENCV_FORCE_3RDPARTY_BUILD ON CACHE BOOL "")

FetchContent_MakeAvailable(opencv)

message(STATUS "[vivid-opencv] OpenCV configured successfully")

# -----------------------------------------------------------------------------
# nlohmann/json (required by vivid operator_registry.h)
# -----------------------------------------------------------------------------
if(VIVID_SDK_BUILD)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
    set(NLOHMANN_JSON_INCLUDE_DIR "${nlohmann_json_SOURCE_DIR}/include")
    message(STATUS "[vivid-opencv] Fetched nlohmann/json: ${NLOHMANN_JSON_INCLUDE_DIR}")

    # magic_enum (required by vivid param.h)
    FetchContent_Declare(
        magic_enum
        GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
        GIT_TAG v0.9.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(magic_enum)
    set(MAGIC_ENUM_INCLUDE_DIR "${magic_enum_SOURCE_DIR}/include")
    message(STATUS "[vivid-opencv] Fetched magic_enum: ${MAGIC_ENUM_INCLUDE_DIR}")
endif()

# -----------------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------------
set(OPENCV_SOURCES
    src/opencv.cpp
    src/contours.cpp
    src/optical_flow.cpp
    src/blob_track.cpp
)

add_library(vivid-opencv SHARED ${OPENCV_SOURCES})

target_include_directories(vivid-opencv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${opencv_SOURCE_DIR}/include
        ${opencv_SOURCE_DIR}/modules/core/include
        ${opencv_SOURCE_DIR}/modules/imgproc/include
        ${opencv_SOURCE_DIR}/modules/video/include
        ${opencv_SOURCE_DIR}/modules/features2d/include
        ${CMAKE_BINARY_DIR}  # For opencv2/opencv_modules.hpp generated config
)

# Link dependencies based on build mode
if(VIVID_BUILD_MODE STREQUAL "in-tree")
    target_link_libraries(vivid-opencv
        PUBLIC vivid-core
        PRIVATE opencv_core opencv_imgproc opencv_video opencv_features2d
    )
elseif(VIVID_BUILD_MODE STREQUAL "standalone")
    target_include_directories(vivid-opencv PRIVATE ${VIVID_INCLUDE_DIR} ${VIVID_DEP_INCLUDE_DIRS})
    if(VIVID_SDK_BUILD)
        if(NLOHMANN_JSON_INCLUDE_DIR)
            target_include_directories(vivid-opencv PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
        endif()
        if(MAGIC_ENUM_INCLUDE_DIR)
            target_include_directories(vivid-opencv PRIVATE ${MAGIC_ENUM_INCLUDE_DIR})
        endif()
    endif()
    target_link_libraries(vivid-opencv PRIVATE opencv_core opencv_imgproc opencv_video opencv_features2d)

    if(WIN32)
        target_link_libraries(vivid-opencv PUBLIC ${VIVID_CORE_LIB})
    else()
        # macOS/Linux: symbols resolved at load time by vivid runtime
        if(APPLE)
            target_link_options(vivid-opencv PRIVATE -undefined dynamic_lookup)
        else()
            target_link_options(vivid-opencv PRIVATE -Wl,--allow-shlib-undefined)
        endif()
    endif()
else()
    # Installed vivid (via find_package)
    target_link_libraries(vivid-opencv
        PUBLIC vivid::core
        PRIVATE opencv_core opencv_imgproc opencv_video opencv_features2d
    )
endif()

# Platform-specific frameworks
if(APPLE)
    target_link_libraries(vivid-opencv PRIVATE
        "-framework Accelerate"
        "-framework CoreVideo"
    )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(vivid-opencv PRIVATE
        z
        pthread
        dl
    )
endif()

# -----------------------------------------------------------------------------
# Platform-specific configuration
# -----------------------------------------------------------------------------
if(APPLE)
    set_target_properties(vivid-opencv PROPERTIES
        INSTALL_RPATH "@executable_path;@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(WIN32)
    target_compile_definitions(vivid-opencv PRIVATE vivid_opencv_EXPORTS)
    set_target_properties(vivid-opencv PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS vivid-opencv
    EXPORT vivid-opencv-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for find_package
install(EXPORT vivid-opencv-targets
    FILE vivid-opencv-targets.cmake
    NAMESPACE vivid::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vivid-opencv
)

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
option(BUILD_TESTS "Build the test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)
endif()

message(STATUS "[vivid-opencv] Configuration complete")
message(STATUS "[vivid-opencv]   Build mode: ${VIVID_BUILD_MODE}")
message(STATUS "[vivid-opencv]   OpenCV version: ${OPENCV_VERSION} (from source)")

# CMakeLists.txt — VEX CV Override (AprilTag detector)
#
# Minimum CMake version required
cmake_minimum_required(VERSION 3.16)

project(vex_cv_override
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "VEX coprocessor AprilTag CV override"
)

# Use C++17 standard throughout
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# OpenCV — provides camera capture and image processing
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio calib3d)

# AprilTag C library — provides the tag detection and pose estimation
# Install on Raspberry Pi / Debian:  sudo apt install libapriltag-dev
find_package(PkgConfig REQUIRED)
pkg_check_modules(APRILTAG REQUIRED apriltag)

# ---------------------------------------------------------------------------
# Library target: the detector logic (reusable)
# ---------------------------------------------------------------------------
add_library(april_tag_detector
    april_tag_detector.cpp
)

target_include_directories(april_tag_detector
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # exposes april_tag_detector.h
        ${OpenCV_INCLUDE_DIRS}
        ${APRILTAG_INCLUDE_DIRS}
)

target_link_libraries(april_tag_detector
    PUBLIC
        ${OpenCV_LIBS}
        ${APRILTAG_LIBRARIES}
)

# ---------------------------------------------------------------------------
# Executable target: the VEX override entry point
# ---------------------------------------------------------------------------
add_executable(vex_cv_override
    vex_override.cpp
)

target_link_libraries(vex_cv_override
    PRIVATE
        april_tag_detector
)

# ---------------------------------------------------------------------------
# Install rules (optional — useful when deploying to the coprocessor)
# ---------------------------------------------------------------------------
install(TARGETS vex_cv_override
    RUNTIME DESTINATION bin
)

/**
 * april_tag_detector.h
 *
 * Header for the AprilTagDetector class used in the VEX coprocessor CV override.
 *
 * This module runs on a companion computer (e.g. Raspberry Pi / Jetson Nano)
 * connected to the VEX V5 Brain over UART/USB serial.  It captures frames from
 * a USB or CSI camera, locates AprilTag fiducial markers in the scene, and
 * reports each tag's ID, distance, and bearing angle back to the VEX brain so
 * the robot can react to field elements and game objects.
 *
 * AprilTag family used: tag36h11  (standard FRC / field-marking family)
 *
 * Dependencies:
 *   - OpenCV  >= 4.5   (core, imgproc, videoio, calib3d)
 *   - AprilTag C library (https://github.com/AprilRobotics/apriltag)
 */

#pragma once

#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

// AprilTag C library headers
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>

namespace vex_cv {

/**
 * @brief Information about a single detected AprilTag.
 */
struct TagDetection {
    int    id;          ///< Unique tag ID encoded in the tag pattern
    double distance_m;  ///< Estimated distance from the camera lens (metres)
    double bearing_deg; ///< Horizontal angle from the camera centre-line (degrees,
                        ///<   negative = left of centre, positive = right)
    double x_m;         ///< Estimated X offset in camera frame (metres)
    double y_m;         ///< Estimated Y offset in camera frame (metres)
    double z_m;         ///< Estimated Z (depth) offset in camera frame (metres)
};

/**
 * @brief Camera intrinsic parameters needed for pose estimation.
 *
 * Obtain these by running OpenCV's camera calibration tool on the
 * specific camera + lens you are using.  Defaults are rough estimates
 * for a typical 640×480 USB webcam.
 */
struct CameraParams {
    double fx = 600.0; ///< Focal length in pixels along X axis
    double fy = 600.0; ///< Focal length in pixels along Y axis
    double cx = 320.0; ///< Principal point (optical centre) X
    double cy = 240.0; ///< Principal point (optical centre) Y
    double tag_size_m = 0.165; ///< Physical side length of the tag (metres)
};

/**
 * @brief Detects AprilTags in a live camera stream using OpenCV and the
 *        AprilTag C library, then estimates the 3-D pose of each tag.
 *
 * Typical usage:
 * @code
 *   vex_cv::AprilTagDetector detector(0);          // open /dev/video0
 *   detector.set_camera_params(params);
 *   if (!detector.open()) { return -1; }
 *
 *   while (running) {
 *       auto tags = detector.detect();
 *       for (auto& tag : tags) {
 *           // send tag info to VEX brain via serial
 *       }
 *   }
 *   detector.close();
 * @endcode
 */
class AprilTagDetector {
public:
    /**
     * @brief Construct the detector.
     * @param camera_index  OpenCV VideoCapture device index (0 = first camera).
     */
    explicit AprilTagDetector(int camera_index = 0);

    /** Destructor — releases the camera and frees AprilTag resources. */
    ~AprilTagDetector();

    // Non-copyable (owns raw C resources)
    AprilTagDetector(const AprilTagDetector&)            = delete;
    AprilTagDetector& operator=(const AprilTagDetector&) = delete;

    /**
     * @brief Override camera intrinsic / tag-size parameters.
     * @param params  New camera parameters (see CameraParams).
     */
    void set_camera_params(const CameraParams& params);

    /**
     * @brief Open the camera and initialise the AprilTag detector.
     * @return true on success, false if the camera could not be opened.
     */
    bool open();

    /**
     * @brief Release the camera and free internal AprilTag resources.
     */
    void close();

    /**
     * @brief Capture one frame and return all detected tags with pose info.
     * @return Vector of TagDetection structs, one per detected tag.
     *         Empty if no tags are visible or the camera is not open.
     */
    std::vector<TagDetection> detect();

    /**
     * @brief Retrieve the last captured frame (BGR, full resolution).
     *
     * Useful for debugging — call after detect() to get the annotated image.
     */
    const cv::Mat& last_frame() const { return frame_; }

private:
    /**
     * @brief Convert raw AprilTag pose to a TagDetection struct.
     */
    TagDetection pose_to_detection(int tag_id,
                                   const apriltag_pose_t& pose) const;

    int          camera_index_;   ///< VideoCapture device index
    CameraParams cam_params_;     ///< Intrinsic + tag-size parameters
    cv::VideoCapture cap_;        ///< OpenCV camera handle
    cv::Mat          frame_;      ///< Most recent captured frame (BGR)

    // AprilTag C library objects
    apriltag_family_t*   tag_family_  = nullptr; ///< Tag36h11 family descriptor
    apriltag_detector_t* detector_   = nullptr;  ///< Core detector instance
};

} // namespace vex_cv

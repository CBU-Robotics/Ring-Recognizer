/**
 * april_tag_detector.cpp
 *
 * Implementation of AprilTagDetector for the VEX coprocessor CV override.
 *
 * Pipeline overview (runs on every call to detect()):
 *   1. Capture a colour frame from the USB/CSI camera via OpenCV.
 *   2. Convert the frame to greyscale — AprilTag works on luminance only.
 *   3. Wrap the grey buffer in an apriltag_image_u8_t (zero-copy).
 *   4. Run apriltag_detector_detect() to find all tags in the frame.
 *   5. For each detection, call estimate_tag_pose() to get the rotation and
 *      translation vectors (SE(3) rigid body transform, camera frame).
 *   6. Extract distance, X/Y/Z offsets, and bearing angle from the pose.
 *   7. Return a vector of TagDetection structs to the caller.
 *
 * All heavy lifting (image decoding, homography, pose estimation) is handled
 * by the AprilTag C library.  OpenCV is only used for camera capture and
 * colour→grey conversion.
 */

#include "april_tag_detector.h"

#include <cmath>
#include <stdexcept>

// OpenCV
#include <opencv2/imgproc.hpp>

namespace vex_cv {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

AprilTagDetector::AprilTagDetector(int camera_index)
    : camera_index_(camera_index) {}

AprilTagDetector::~AprilTagDetector() {
    close();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void AprilTagDetector::set_camera_params(const CameraParams& params) {
    cam_params_ = params;
}

bool AprilTagDetector::open() {
    // ---- Camera ----
    cap_.open(camera_index_);
    if (!cap_.isOpened()) {
        return false;
    }

    // Request a sensible resolution; the driver will choose the closest match.
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // ---- AprilTag detector ----
    // Create the tag36h11 family (most common; robust at distance).
    tag_family_ = tag36h11_create();

    detector_ = apriltag_detector_create();

    // Add the tag family to the detector.
    apriltag_detector_add_family(detector_, tag_family_);

    // Tuning parameters:
    //   quad_decimate — downsample factor before quad detection.
    //     2.0 is a good balance between speed and detection range.
    detector_->quad_decimate = 2.0f;

    //   quad_sigma — amount of Gaussian blur applied before quad detection.
    //     0 = no blur (sharpest, fastest), higher = smoother (helps with noise).
    detector_->quad_sigma = 0.0f;

    //   nthreads — number of threads for the detector.
    //     Set to 1 to keep CPU usage predictable on a Raspberry Pi.
    detector_->nthreads = 1;

    //   debug — set to 1 to write intermediate images to disk (slow).
    detector_->debug = 0;

    //   refine_edges — improves localisation accuracy at a small CPU cost.
    detector_->refine_edges = 1;

    return true;
}

void AprilTagDetector::close() {
    // Release the camera
    if (cap_.isOpened()) {
        cap_.release();
    }

    // Free AprilTag resources (must be done in this order)
    if (detector_) {
        apriltag_detector_destroy(detector_);
        detector_ = nullptr;
    }
    if (tag_family_) {
        tag36h11_destroy(tag_family_);
        tag_family_ = nullptr;
    }
}

std::vector<TagDetection> AprilTagDetector::detect() {
    std::vector<TagDetection> results;

    if (!cap_.isOpened() || !detector_) {
        return results; // Not initialised — return empty
    }

    // ---- 1. Capture frame ----
    cap_ >> frame_;
    if (frame_.empty()) {
        return results; // Camera read failed
    }

    // ---- 2. Convert to greyscale ----
    cv::Mat grey;
    cv::cvtColor(frame_, grey, cv::COLOR_BGR2GRAY);

    // ---- 3. Wrap in AprilTag image structure (zero-copy) ----
    // apriltag_image_u8_t stores row-major 8-bit pixel data.
    image_u8_t ap_img = {
        .width  = grey.cols,
        .height = grey.rows,
        .stride = grey.cols,  // OpenCV Mat is contiguous for a grey image
        .buf    = grey.data
    };

    // ---- 4. Detect tags ----
    zarray_t* detections = apriltag_detector_detect(detector_, &ap_img);

    // ---- 5. Estimate pose for each detection ----
    // Build the camera intrinsics info struct once per frame.
    apriltag_detection_info_t det_info;
    det_info.tagsize = cam_params_.tag_size_m;
    det_info.fx      = cam_params_.fx;
    det_info.fy      = cam_params_.fy;
    det_info.cx      = cam_params_.cx;
    det_info.cy      = cam_params_.cy;

    int num_detections = zarray_size(detections);
    results.reserve(static_cast<size_t>(num_detections));

    for (int i = 0; i < num_detections; ++i) {
        apriltag_detection_t* det = nullptr;
        zarray_get(detections, i, &det);

        det_info.det = det;

        // estimate_tag_pose() solves the Perspective-n-Point (PnP) problem
        // and fills a rigid body transform: rotation R and translation t.
        apriltag_pose_t pose;
        double err = estimate_tag_pose(&det_info, &pose);
        (void)err; // re-projection error — can be used for confidence filtering

        // ---- 6. Convert pose to our TagDetection struct ----
        results.push_back(pose_to_detection(det->id, pose));

        // Free the pose matrices allocated by estimate_tag_pose
        matd_destroy(pose.R);
        matd_destroy(pose.t);
    }

    // ---- 7. Free detection list (individual detections freed automatically) ----
    apriltag_detections_destroy(detections);

    return results;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

TagDetection AprilTagDetector::pose_to_detection(int tag_id,
                                                  const apriltag_pose_t& pose) const {
    TagDetection td;
    td.id = tag_id;

    // The translation vector t is a 3×1 matrix in the camera coordinate frame:
    //   t->data[0] = X  (positive = right of camera)
    //   t->data[1] = Y  (positive = down in the image)
    //   t->data[2] = Z  (positive = towards the camera / depth)
    double tx = pose.t->data[0]; // metres
    double ty = pose.t->data[1];
    double tz = pose.t->data[2];

    td.x_m = tx;
    td.y_m = ty;
    td.z_m = tz;

    // Euclidean distance from the camera origin to the tag centre.
    td.distance_m = std::sqrt(tx * tx + ty * ty + tz * tz);

    // Bearing angle: horizontal angle between the tag and the camera's
    // optical axis.  atan2(X, Z) gives the angle in the XZ plane.
    //   0°  = directly ahead
    //   +°  = tag is to the right
    //   -°  = tag is to the left
    td.bearing_deg = std::atan2(tx, tz) * (180.0 / M_PI);

    return td;
}

} // namespace vex_cv

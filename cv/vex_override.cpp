/**
 * vex_override.cpp
 *
 * VEX coprocessor CV override — main entry point.
 *
 * This program runs on a companion computer (e.g. Raspberry Pi Zero 2W or
 * Jetson Nano) that is physically connected to a VEX V5 Brain via a USB
 * serial link.  It continuously reads AprilTag detections from the camera
 * and streams structured data packets to the VEX Brain so the autonomous
 * and driver-assist code there can use precise field-relative positioning.
 *
 * Serial protocol (ASCII, one line per frame):
 *   Each detected tag produces one line:
 *     TAG <id> <distance_m> <bearing_deg> <x_m> <y_m> <z_m>\n
 *
 *   At the end of every frame (even if no tags were seen):
 *     FRAME_END\n
 *
 * The VEX Brain reads these lines with its serial port and parses the fields.
 *
 * Usage:
 *   ./vex_cv_override [camera_index] [serial_port]
 *
 *   Defaults:
 *     camera_index  = 0          (first USB camera)
 *     serial_port   = /dev/ttyACM0  (VEX V5 Brain over USB)
 *
 * Build:
 *   See CMakeLists.txt in this directory.
 */

#include "april_tag_detector.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <termios.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Signal handling — clean shutdown on Ctrl-C
// ---------------------------------------------------------------------------

static volatile bool g_running = true;

static void signal_handler(int /*sig*/) {
    g_running = false;
}

// ---------------------------------------------------------------------------
// Serial port helpers
// ---------------------------------------------------------------------------

/**
 * @brief Open and configure a serial port for communication with the VEX Brain.
 *
 * The VEX V5 Brain USB serial port runs at 115200 baud, 8N1, no flow control.
 *
 * @param port  Path to the serial device (e.g. "/dev/ttyACM0").
 * @return File descriptor on success, -1 on failure.
 */
static int open_serial(const std::string& port) {
    int fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        std::perror(("open_serial: " + port).c_str());
        return -1;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        std::perror("tcgetattr");
        ::close(fd);
        return -1;
    }

    // Baud rate: 115200
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);

    // 8-bit characters, no parity, one stop bit (8N1)
    tty.c_cflag &= ~static_cast<unsigned>(PARENB); // No parity
    tty.c_cflag &= ~static_cast<unsigned>(CSTOPB); // 1 stop bit
    tty.c_cflag &= ~static_cast<unsigned>(CSIZE);
    tty.c_cflag |=  CS8;                           // 8 data bits

    // Disable hardware flow control
    tty.c_cflag &= ~static_cast<unsigned>(CRTSCTS);

    // Enable receiver, ignore modem control lines
    tty.c_cflag |= (CREAD | CLOCAL);

    // Raw input/output — no special character processing
    tty.c_lflag &= ~static_cast<unsigned>(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~static_cast<unsigned>(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~static_cast<unsigned>(OPOST);

    // Non-blocking reads
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::perror("tcsetattr");
        ::close(fd);
        return -1;
    }

    return fd;
}

/**
 * @brief Write a string to the serial port.
 * @param fd   File descriptor returned by open_serial().
 * @param msg  Message to send (should end with '\n').
 */
static void serial_write(int fd, const std::string& msg) {
    if (fd < 0) return;
    ::write(fd, msg.c_str(), msg.size());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // ---- Parse arguments ----
    int         camera_index = 0;
    std::string serial_port  = "/dev/ttyACM0";

    if (argc > 1) camera_index = std::atoi(argv[1]);
    if (argc > 2) serial_port  = argv[2];

    // ---- Install signal handler for clean shutdown ----
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- Open serial link to VEX Brain ----
    int serial_fd = open_serial(serial_port);
    if (serial_fd < 0) {
        std::cerr << "Warning: could not open serial port " << serial_port
                  << " — running in console-only mode.\n";
    }

    // ---- Initialise the AprilTag detector ----
    vex_cv::AprilTagDetector detector(camera_index);

    // Camera intrinsics — update these values after calibrating your specific
    // camera with OpenCV's calibrateCamera().
    vex_cv::CameraParams cam;
    cam.fx         = 600.0;   // Focal length X (pixels)
    cam.fy         = 600.0;   // Focal length Y (pixels)
    cam.cx         = 320.0;   // Principal point X (pixels)
    cam.cy         = 240.0;   // Principal point Y (pixels)
    cam.tag_size_m = 0.165;   // Physical tag side length (metres) — measure yours!
    detector.set_camera_params(cam);

    if (!detector.open()) {
        std::cerr << "Error: could not open camera " << camera_index << "\n";
        if (serial_fd >= 0) ::close(serial_fd);
        return 1;
    }

    std::cout << "VEX CV override running.  Camera=" << camera_index
              << "  Serial=" << serial_port << "\n";
    std::cout << "Press Ctrl-C to stop.\n";

    // ---- Main detection loop ----
    while (g_running) {
        // Detect all AprilTags in the current frame
        auto tags = detector.detect();

        // Build and send a packet for each detected tag
        for (const auto& tag : tags) {
            // Format: TAG <id> <dist> <bearing> <x> <y> <z>
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "TAG %d %.4f %.2f %.4f %.4f %.4f\n",
                          tag.id,
                          tag.distance_m,
                          tag.bearing_deg,
                          tag.x_m,
                          tag.y_m,
                          tag.z_m);

            // Send to VEX Brain
            serial_write(serial_fd, buf);

            // Also echo to stdout for debugging
            std::cout << buf;
        }

        // Signal end of this frame so the VEX Brain knows all tags for the
        // current cycle have been sent.
        serial_write(serial_fd, "FRAME_END\n");
    }

    // ---- Cleanup ----
    std::cout << "\nShutting down.\n";
    detector.close();
    if (serial_fd >= 0) ::close(serial_fd);

    return 0;
}

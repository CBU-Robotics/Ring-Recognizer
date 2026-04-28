# cv — VEX Coprocessor CV Override (AprilTag Detection)

This folder contains a C++ computer-vision module that runs on a **companion
computer** (e.g. Raspberry Pi Zero 2 W, Jetson Nano, or any Linux SBC) connected
to a **VEX V5 Brain**.  It replaces (overrides) the limited built-in VEX Vision
Sensor with a full AprilTag detection pipeline, giving the robot accurate
**field-relative position and bearing** for each visible game element or field
marker.

---

## What Are AprilTags?

[AprilTags](https://april.eecs.umich.edu/software/apriltag) are 2-D fiducial
markers (similar to QR codes) printed on paper or rigid boards and placed on
field elements, game pieces, or walls.  Each tag encodes a unique numeric **ID**
and, when viewed through a calibrated camera, allows software to compute the
tag's **3-D position and orientation** relative to the camera — no depth sensor
needed.

The `tag36h11` family is used here (64-bit payload, very robust error detection).

---

## How It Works

```
┌──────────┐  USB/CSI  ┌────────────────────┐  USB Serial  ┌───────────────┐
│  Camera  │ ─────────▶│  Companion Computer  │ ───────────▶│  VEX V5 Brain │
└──────────┘           │  (this program)      │             └───────────────┘
                       │                      │
                       │  1. Capture frame     │
                       │  2. Greyscale convert │
                       │  3. AprilTag detect   │
                       │  4. Pose estimation   │
                       │  5. Serial TX         │
                       └────────────────────  ┘
```

### Step-by-step pipeline (per frame)

| Step | What happens |
|------|-------------|
| **1 — Capture** | OpenCV reads a frame from the USB or CSI camera (`/dev/video0` by default). |
| **2 — Greyscale** | The BGR frame is converted to a single-channel greyscale image — AprilTag only needs luminance. |
| **3 — Detect** | The AprilTag C library scans the image for quad (quadrilateral) candidates, decodes each one's ID, and returns a list of pixel-corner detections. |
| **4 — Pose estimation** | `estimate_tag_pose()` solves the **Perspective-n-Point (PnP)** problem using the four known corner positions of the tag and the camera's intrinsic parameters (focal length, principal point).  Output is a **rotation matrix R** and **translation vector t** in the camera coordinate frame. |
| **5 — Serial TX** | For each tag, one ASCII line is sent over the USB serial link to the VEX Brain: `TAG <id> <distance_m> <bearing_deg> <x_m> <y_m> <z_m>`. A `FRAME_END` line marks the end of each frame. |

---

## Serial Protocol

The VEX Brain reads lines from the companion computer.  Each line is one of:

```
TAG <id> <distance_m> <bearing_deg> <x_m> <y_m> <z_m>
FRAME_END
```

| Field | Units | Meaning |
|-------|-------|---------|
| `id` | integer | The tag's encoded numeric ID |
| `distance_m` | metres | Straight-line distance from the camera lens to the tag centre |
| `bearing_deg` | degrees | Horizontal angle from the camera's forward axis (negative = left, positive = right) |
| `x_m` | metres | Tag X offset in the camera frame (right = positive) |
| `y_m` | metres | Tag Y offset in the camera frame (down = positive) |
| `z_m` | metres | Tag Z (depth) offset in the camera frame (forward = positive) |

---

## File Structure

```
cv/
├── april_tag_detector.h    # Class header — AprilTagDetector + TagDetection struct
├── april_tag_detector.cpp  # Detection & pose-estimation implementation
├── vex_override.cpp        # Main entry point — serial bridge to VEX Brain
├── CMakeLists.txt          # CMake build configuration
└── README.md               # This file
```

---

## Dependencies

| Library | Version | Purpose | Install (Debian/Ubuntu/Raspberry Pi OS) |
|---------|---------|---------|----------------------------------------|
| OpenCV | ≥ 4.5 | Camera capture, colour conversion | `sudo apt install libopencv-dev` |
| AprilTag | ≥ 3.2 | Tag detection & pose estimation | `sudo apt install libapriltag-dev` |
| CMake | ≥ 3.16 | Build system | `sudo apt install cmake` |

---

## Building

```bash
# From the repo root
mkdir -p cv/build && cd cv/build
cmake ..
make -j$(nproc)
```

The build produces the `vex_cv_override` executable in `cv/build/`.

---

## Running

```bash
# Default: camera 0, serial port /dev/ttyACM0
./vex_cv_override

# Custom camera index and serial port
./vex_cv_override 1 /dev/ttyUSB0
```

### Auto-start on boot (systemd)

Create `/etc/systemd/system/vex-cv.service`:

```ini
[Unit]
Description=VEX CV Override — AprilTag detector
After=network.target

[Service]
ExecStart=/home/pi/Ring-Recognizer/cv/build/vex_cv_override 0 /dev/ttyACM0
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl enable vex-cv
sudo systemctl start  vex-cv
```

---

## Camera Calibration

The pose-estimation accuracy depends on **accurate camera intrinsic parameters**
(`fx`, `fy`, `cx`, `cy`).  The defaults in the code are rough estimates for a
640 × 480 USB webcam — they will give ballpark distances but not precise ones.

To calibrate your specific camera:

1. Print the OpenCV chessboard calibration pattern.
2. Run `opencv_interactive-calibration` or write a short calibration script
   (`calibrateCamera()`).
3. Update the `CameraParams` values in `vex_override.cpp`.

Also measure the **physical side length** of your printed AprilTags and update
`cam.tag_size_m` accordingly.

---

## Coordinate Frame

```
       Y (down)
       │
       │
       └─────── X (right)
      /
     /
    Z (forward, into the scene)
```

This is the standard OpenCV / camera coordinate frame.  `bearing_deg` is the
horizontal angle in the XZ plane:

- `0°`  → tag is directly in front of the camera
- `+30°` → tag is 30° to the right
- `-30°` → tag is 30° to the left

---

## Tuning Tips

| Parameter | Location | Effect |
|-----------|----------|--------|
| `quad_decimate` | `april_tag_detector.cpp` | Higher = faster but shorter detection range |
| `quad_sigma` | `april_tag_detector.cpp` | Higher = more blur, better for noisy images |
| `nthreads` | `april_tag_detector.cpp` | More threads = faster on multi-core SBCs |
| Camera resolution | `april_tag_detector.cpp` | Higher = longer range, lower = faster FPS |
| `tag_size_m` | `vex_override.cpp` | Must match actual printed tag size for correct distance |

#!/usr/bin/env python3
"""
Dataset processor for Ring-Recognizer project (improved)

- Supports multiple image extensions
- Uses robust findContours compatibility helper
- Adaptive morphological kernel and min_area based on image size
- Headless-safe (won't call cv2.imshow if DISPLAY not present)
- More logging and configurable behavior
- Uses shape_detector (if available) but has a local fallback detector

Usage:
  python dataset_processor.py --dataset dataset --output processed --show --sample 10
"""

from pathlib import Path
import argparse
import logging
import sys
import os
from typing import List, Tuple

import cv2
import numpy as np

from common_utils import find_contours_compat

# Attempt to import shape_detector functions if present in repository
try:
    # assumes shape_detector.py (snake_case) exists in same folder or installed
    from shape_detector import detect_shapes_contours, detect_circles_hough, detect_regular_polygons, detect_square_plus_regular_ngon  # type: ignore
    HAVE_SHAPE_MODULE = True
except Exception:
    HAVE_SHAPE_MODULE = False

# -------------------------
# Helpers
# -------------------------

def is_headless() -> bool:
    # On Linux DISPLAY is required for cv2.imshow; on Windows/mac it generally exists.
    if sys.platform.startswith("linux"):
        return os.environ.get("DISPLAY", "") == ""
    return False

def image_extensions() -> List[str]:
    return ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

# -------------------------
# Simple local detector (fallback)
# -------------------------
def detect_shapes_simple(image: np.ndarray,
                         min_area_px: int = None,
                         hsv_ranges: dict = None) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int,int,int,int], float]]]:
    """
    Simple color-based detector. Returns annotated image and list of detections:
      (color, (x,y,w,h), area)

    This is a fallback if the more advanced `shape_detector` module is not available.
    """
    if hsv_ranges is None:
        hsv_ranges = {
            "red1": (np.array([0, 60, 50]), np.array([10, 255, 255])),
            "red2": (np.array([170, 60, 50]), np.array([180, 255, 255])),
            "blue": (np.array([100, 60, 50]), np.array([140, 255, 255])),
        }

    h, w = image.shape[:2]
    # default min_area relative to image size if not specified
    if min_area_px is None:
        min_area_px = max(300, int((w * h) * 0.0004))

    # Blur and convert
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Masks
    r1_lo, r1_hi = hsv_ranges["red1"]
    r2_lo, r2_hi = hsv_ranges["red2"]
    b_lo, b_hi = hsv_ranges["blue"]

    red_mask1 = cv2.inRange(hsv, r1_lo, r1_hi)
    red_mask2 = cv2.inRange(hsv, r2_lo, r2_hi)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, b_lo, b_hi)

    # Morphology kernel size proportional to image size
    k = max(3, int(min(w, h) / 200))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    result = image.copy()
    detections = []

    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), ("blue", blue_mask, (255, 0, 0))]:
        contours, _ = find_contours_compat(mask)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            x, y, wbox, hbox = cv2.boundingRect(cnt)
            # size filter relative to image
            if wbox < 10 or hbox < 10:
                continue
            if wbox > image.shape[1] * 0.95 or hbox > image.shape[0] * 0.95:
                # likely background/edge; skip
                continue

            cv2.rectangle(result, (x, y), (x + wbox, y + hbox), bbox_color, 2)
            cv2.putText(result, f"{color} shape", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
            detections.append((color, (x, y, wbox, hbox), float(area)))

    return result, detections

# -------------------------
# Dataset processing
# -------------------------
def process_dataset(dataset_path: str,
                    output_path: str = None,
                    show_images: bool = False,
                    sample_size: int = None,
                    use_shape_module: bool = True):
    """
    Process images in dataset folder structure:
      dataset/
        Blue objects/
        Red objects/

    Returns: results dict
    """
    logger = logging.getLogger("dataset_processor")
    dataset_path = Path(dataset_path)
    if output_path:
        outp = Path(output_path)
        outp.mkdir(parents=True, exist_ok=True)

    results = {"blue_objects": [], "red_objects": []}

    folders = [("Blue objects", "blue_objects"), ("Red objects", "red_objects")]

    for folder_name, key in folders:
        folder = dataset_path / folder_name
        if not folder.exists():
            logger.info("Folder not found: %s - skipping", folder)
            continue

        # collect images with common extensions (case-insensitive)
        image_files = []
        for ext in image_extensions():
            image_files.extend(sorted(folder.glob(ext)))
            image_files.extend(sorted(folder.glob(ext.upper())))
        image_files = sorted(set(image_files))

        if sample_size:
            image_files = image_files[:sample_size]

        logger.info("Processing %d images in %s", len(image_files), folder)

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning("Unable to read image: %s", img_file)
                continue

            # Prefer advanced shape_detector if available and requested
            if use_shape_module and HAVE_SHAPE_MODULE:
                try:
                    processed, detections = detect_regular_polygons(img, n=18, debug=False)
                    # If composite detection desired, call detect_square_plus_regular_ngon instead.
                except Exception:
                    # fallback to simple detector if advanced fails
                    logger.exception("shape_detector failed for %s - falling back to simple detector", img_file)
                    processed, detections = detect_shapes_simple(img)
            else:
                processed, detections = detect_shapes_simple(img)

            blue_count = sum(1 for d in detections if d[0] == "blue" or (isinstance(d, dict) and d.get("n") is None and d.get("shape") == "blue"))
            red_count = sum(1 for d in detections if d[0] == "red" or (isinstance(d, dict) and d.get("shape") == "red"))

            record = {
                "filename": img_file.name,
                "blue_detected": int(blue_count),
                "red_detected": int(red_count),
                "total_shapes": int(len(detections))
            }
            results[key].append(record)

            if show_images and not is_headless():
                try:
                    cv2.imshow(f"{folder_name} - {img_file.name}", processed)
                    # show for minimal time - allow user to press a key to continue
                    cv2.waitKey(250)
                except Exception:
                    logger.exception("cv2.imshow failed (likely headless) - skipping GUI display")
                    show_images = False

            if output_path:
                out_file = Path(output_path) / f"processed_{key}_{img_file.name}"
                try:
                    cv2.imwrite(str(out_file), processed)
                except Exception:
                    logger.exception("Failed to write processed image %s", out_file)

        # close any open windows for folder loop
        if show_images and not is_headless():
            cv2.destroyAllWindows()

    return results

def print_summary(results):
    """
    Print a human-readable summary of detection results
    """
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)

    # Blue
    blue_results = results.get("blue_objects", [])
    if blue_results:
        print(f"\nBLUE OBJECTS FOLDER ({len(blue_results)} images):")
        detected_blue = sum(1 for r in blue_results if r["blue_detected"] > 0)
        missed_blue = len(blue_results) - detected_blue
        false_red = sum(1 for r in blue_results if r["red_detected"] > 0)
        no_detection = sum(1 for r in blue_results if r["total_shapes"] == 0)

        print(f"  ✓ Successfully detected blue objects: {detected_blue}/{len(blue_results)} ({detected_blue/len(blue_results)*100:.1f}%)")
        print(f"  ✗ Missed blue objects: {missed_blue}/{len(blue_results)} ({missed_blue/len(blue_results)*100:.1f}%)")
        print(f"  ⚠️ False positive (detected as red): {false_red}/{len(blue_results)} ({false_red/len(blue_results)*100:.1f}%)")
        print(f"  ❌ No detection at all: {no_detection}/{len(blue_results)} ({no_detection/len(blue_results)*100:.1f}%)")

        avg_blue = sum(r["blue_detected"] for r in blue_results) / len(blue_results)
        print(f"  📊 Average blue detections per image: {avg_blue:.1f}")

        missed_files = [r["filename"] for r in blue_results if r["blue_detected"] == 0]
        if missed_files:
            print(f"  🔍 Examples of missed detections: {', '.join(missed_files[:3])}")

    # Red
    red_results = results.get("red_objects", [])
    if red_results:
        print(f"\nRED OBJECTS FOLDER ({len(red_results)} images):")
        detected_red = sum(1 for r in red_results if r["red_detected"] > 0)
        missed_red = len(red_results) - detected_red
        false_blue = sum(1 for r in red_results if r["blue_detected"] > 0)
        no_detection = sum(1 for r in red_results if r["total_shapes"] == 0)

        print(f"  ✓ Successfully detected red objects: {detected_red}/{len(red_results)} ({detected_red/len(red_results)*100:.1f}%)")
        print(f"  ✗ Missed red objects: {missed_red}/{len(red_results)} ({missed_red/len(red_results)*100:.1f}%)")
        print(f"  ⚠️ False positive (detected as blue): {false_blue}/{len(red_results)} ({false_blue/len(red_results)*100:.1f}%)")
        print(f"  ❌ No detection at all: {no_detection}/{len(red_results)} ({no_detection/len(red_results)*100:.1f}%)")

        avg_red = sum(r["red_detected"] for r in red_results) / len(red_results)
        print(f"  📊 Average red detections per image: {avg_red:.1f}")

        missed_files = [r["filename"] for r in red_results if r["red_detected"] == 0]
        if missed_files:
            print(f"  🔍 Examples of missed detections: {', '.join(missed_files[:3])}")

    # Overall
    total_images = len(blue_results) + len(red_results)
    correct_detections = (sum(1 for r in blue_results if r["blue_detected"] > 0) +
                          sum(1 for r in red_results if r["red_detected"] > 0))
    if total_images > 0:
        overall_accuracy = correct_detections / total_images * 100
        print(f"\n🎯 OVERALL ACCURACY: {correct_detections}/{total_images} ({overall_accuracy:.1f}%)")

def _parse_args():
    p = argparse.ArgumentParser(description="Process dataset images with improved shape detection")
    p.add_argument("--dataset", default="dataset", help="Path to dataset folder")
    p.add_argument("--output", help="Path to save processed images")
    p.add_argument("--show", action="store_true", help="Show processed images (only if display available)")
    p.add_argument("--sample", type=int, help="Process only first N images from each folder")
    p.add_argument("--no-shape-module", action="store_true", help="Do not use shape_detector module even if present")
    p.add_argument("--log", default="info", help="Log level (debug, info, warning, error)")
    return p.parse_args()

def main():
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    logger = logging.getLogger("dataset_processor")

    if args.show and is_headless():
        logger.warning("Display not detected: --show will be ignored in headless environment")
        args.show = False

    logger.info("Starting dataset processing...")
    results = process_dataset(args.dataset, args.output, args.show, args.sample, use_shape_module=not args.no_shape_module)
    print_summary(results)
    if args.output:
        logger.info("Processed images saved to: %s", args.output)

if __name__ == "__main__":
    main()
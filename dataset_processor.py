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
import json
import random

import cv2
import numpy as np

from common_utils import find_contours_compat, create_color_masks

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
# Improved detection function to handle background elements
# -------------------------
def detect_objects_improved(image: np.ndarray):
    """
    Improved object detection with enhanced filtering (matching block cam.py)
    Uses the same detection logic as the live camera system for consistency.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Apply more blur to reduce noise (matching block cam)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create masks using common utilities (same as block cam)
    red_mask, blue_mask = create_color_masks(hsv_frame)
    
    # Larger morphological operations (matching block cam)
    k = max(5, int(min(w, h) / 150))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    
    # Apply morphological operations with 2 iterations (matching block cam)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    detections = []
    processed_image = image.copy()
    
    # Minimum area based on frame size (matching block cam)
    min_area_px = max(2000, int((w * h) * 0.001))
    center_bias = 0.3
    
    # Process both red and blue objects (exactly like block cam)
    for color, mask in [("red", red_mask), ("blue", blue_mask)]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours using improved criteria (same as block cam)
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue
                
            # Get contour properties
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Filter out edge objects (reduced margin for better edge detection)
            border_margin = min(w, h) * 0.01
            if (x < border_margin or y < border_margin or 
                x + w_rect > w - border_margin or y + h_rect > h - border_margin):
                continue
            
            # Filter by aspect ratio (relaxed for large objects)
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            if area < (w*h)*0.25:  # only enforce for smaller objects
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    continue
            
            # Calculate distance from center (same as block cam)
            contour_center_x = x + w_rect // 2
            contour_center_y = y + h_rect // 2
            dist_from_center = np.sqrt((contour_center_x - center_x)**2 + (contour_center_y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_score = 1.0 - (dist_from_center / max_dist)
            
            # Calculate circularity (same as block cam)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            valid_contours.append((contour, area, center_score, circularity))
        
        # Sort by composite score (same as block cam)
        valid_contours.sort(key=lambda x: x[1] * (1 + center_bias * x[2]) * (1 + 0.5 * x[3]), reverse=True)
        
        # Take only the best contour(s) - max 2 objects per color (same as block cam)
        for i, (contour, area, center_score, circularity) in enumerate(valid_contours[:2]):
            # Use circular bounding (same as block cam)
            (center_x_obj, center_y_obj), radius = cv2.minEnclosingCircle(contour)
            center_x_obj, center_y_obj = int(center_x_obj), int(center_y_obj)
            radius = int(radius)
            
            # Draw on processed image (same as block cam)
            bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
            cv2.circle(processed_image, (center_x_obj, center_y_obj), radius, bbox_color, 3)
            cv2.putText(processed_image, f"{color} object", (center_x_obj - radius, center_y_obj - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
            
            # Convert circle to bounding box for compatibility
            x, y = center_x_obj - radius, center_y_obj - radius
            w_box, h_box = radius * 2, radius * 2
            
            # Store detection in format expected by existing code
            detections.append((color, (x, y, w_box, h_box), area))
    
    return processed_image, detections


def draw_detections_on_image(image, detections):
    """
    Draw detection results on image (matching block cam visualization)
    """
    result = image.copy()
    
    for detection in detections:
        color = detection["color"]
        center_x, center_y = detection["center"]
        radius = detection["radius"]
        
        # Use same colors as block cam
        bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
        
        # Draw circle (same as block cam)
        cv2.circle(result, (center_x, center_y), radius, bbox_color, 3)
        cv2.putText(result, f"{color} object", (center_x - radius, center_y - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
    
    return result

# -------------------------
# Dataset processing
# -------------------------
def clear_output_directory(output_path: str):
    """
    Clear all files from the output directory before processing
    """
    if not output_path:
        return
    
    output_dir = Path(output_path)
    if output_dir.exists():
        try:
            for file in output_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger = logging.getLogger("dataset_processor")
            logger.info("Cleared output directory: %s", output_path)
        except Exception as e:
            logger = logging.getLogger("dataset_processor")
            logger.warning("Failed to clear output directory: %s", str(e))

def process_dataset(dataset_path: str,
                    output_path: str = None,
                    show_images: bool = False,
                    sample_size: int = None,
                    random_sample: bool = False,
                    use_shape_module: bool = True):
    """
    Process images in dataset folder structure:
      dataset/
        Blue objects/
        Red objects/

    Args:
        dataset_path: Path to dataset folder
        output_path: Path to save processed images
        show_images: Whether to display images while processing
        sample_size: Number of images to process from each folder
        random_sample: If True, randomly select sample_size images. If False, take first N.
        use_shape_module: Whether to use shape_detector if available

    Returns: results dict with detailed statistics
    """
    logger = logging.getLogger("dataset_processor")
    dataset_path = Path(dataset_path)
    if output_path:
        outp = Path(output_path)
        outp.mkdir(parents=True, exist_ok=True)
        # Clear old processed images
        clear_output_directory(output_path)

    results = {"blue_objects": [], "red_objects": []}
    stats = {"total_processed": 0, "failed": 0, "errors": []}

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
            if random_sample:
                # Randomly select sample_size images
                if len(image_files) > sample_size:
                    image_files = random.sample(image_files, sample_size)
                logger.info("Randomly selected %d/%d images from %s", len(image_files), len(set(image_files)), folder)
            else:
                # Take first sample_size images
                image_files = image_files[:sample_size]
                logger.info("Selected first %d images from %s", len(image_files), folder)

        if not image_files:
            logger.warning("No images found in %s", folder)
            continue

        logger.info("Processing %d images in %s", len(image_files), folder)

        for idx, img_file in enumerate(image_files, 1):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.warning("Unable to read image: %s", img_file)
                    stats["failed"] += 1
                    stats["errors"].append(f"Failed to read: {img_file.name}")
                    continue

                # Log progress
                if idx % max(1, len(image_files) // 10) == 0:
                    logger.debug("Progress: %d/%d images processed", idx, len(image_files))

                # Prefer advanced shape_detector if available and requested
                if use_shape_module and HAVE_SHAPE_MODULE:
                    try:
                        processed, detections = detect_regular_polygons(img, n=18, debug=False)
                        # If composite detection desired, call detect_square_plus_regular_ngon instead.
                    except Exception as e:
                        # fallback to improved detector if advanced fails
                        logger.debug("shape_detector failed for %s: %s - using fallback", img_file.name, str(e))
                        processed, detections = detect_objects_improved(img)
                else:
                    processed, detections = detect_objects_improved(img)

                # Only consider detections that passed all high-confidence logic 
                # (i.e., survived the improved detector's filtering + ranking)
                high_conf_detections = [
                    d for d in detections
                    if isinstance(d, tuple) and len(d) >= 3
                ]

                blue_count = sum(1 for d in high_conf_detections if d[0] == "blue")
                red_count  = sum(1 for d in high_conf_detections if d[0] == "red")

                record = {
                    "filename": img_file.name,
                    "blue_detected": int(blue_count),
                    "red_detected": int(red_count),
                    "total_shapes": int(len(detections)),
                    "path": str(img_file)
                }
                results[key].append(record)
                stats["total_processed"] += 1

                if show_images and not is_headless():
                    try:
                        cv2.imshow(f"{folder_name} - {img_file.name}", processed)
                        # show for minimal time - allow user to press a key to continue
                        key_press = cv2.waitKey(250) & 0xFF
                        if key_press == ord('q'):
                            logger.info("User quit during image display")
                            show_images = False
                    except Exception as e:
                        logger.debug("cv2.imshow failed: %s - skipping GUI display", str(e))
                        show_images = False

                if output_path:
                    out_file = Path(output_path) / f"processed_{key}_{img_file.name}"
                    try:
                        cv2.imwrite(str(out_file), processed)
                        logger.debug("Saved: %s", out_file)
                    except Exception as e:
                        logger.error("Failed to write processed image %s: %s", out_file, str(e))
                        stats["errors"].append(f"Failed to write: {out_file.name}")

            except Exception as e:
                logger.error("Unexpected error processing %s: %s", img_file, str(e))
                stats["failed"] += 1
                stats["errors"].append(f"Error: {img_file.name} - {str(e)}")
                continue

        # close any open windows for folder loop
        if show_images and not is_headless():
            cv2.destroyAllWindows()

    # Log processing summary
    logger.info("Dataset processing complete: %d processed, %d failed", 
                stats["total_processed"], stats["failed"])
    if stats["errors"]:
        logger.warning("Errors encountered: %d", len(stats["errors"]))
        for error in stats["errors"][:5]:  # Show first 5 errors
            logger.warning("  - %s", error)

    return results

def print_summary(results):
    """
    Print a comprehensive human-readable summary of detection results
    with detailed metrics and problem identification
    """
    print("\n" + "="*60)
    print("DETECTION SUMMARY - COMPREHENSIVE ANALYSIS")
    print("="*60)

    # Blue
    blue_results = results.get("blue_objects", [])
    if blue_results:
        print(f"\n📘 BLUE OBJECTS FOLDER ({len(blue_results)} images):")
        print("-" * 60)
        
        detected_blue = sum(1 for r in blue_results if r["blue_detected"] > 0)
        missed_blue = len(blue_results) - detected_blue
        false_red = sum(1 for r in blue_results if r["red_detected"] > 0)
        no_detection = sum(1 for r in blue_results if r["total_shapes"] == 0)
        
        detection_rate = detected_blue / len(blue_results) * 100
        false_positive_rate = false_red / len(blue_results) * 100
        
        # Color-code the metrics
        rate_symbol = "🟢" if detection_rate >= 90 else "🟡" if detection_rate >= 70 else "🔴"
        false_symbol = "🟢" if false_positive_rate < 10 else "🟡" if false_positive_rate < 20 else "🔴"

        print(f"  {rate_symbol} Successfully detected blue: {detected_blue:3d}/{len(blue_results)} ({detection_rate:5.1f}%)")
        print(f"  ❌ Missed blue detections:     {missed_blue:3d}/{len(blue_results)} ({100-detection_rate:5.1f}%)")
        print(f"  {false_symbol} False positives (as red):    {false_red:3d}/{len(blue_results)} ({false_positive_rate:5.1f}%)")
        print(f"  ⚠️  No detection at all:        {no_detection:3d}/{len(blue_results)}")

        avg_blue = sum(r["blue_detected"] for r in blue_results) / len(blue_results)
        total_detections = sum(r["total_shapes"] for r in blue_results)
        print(f"\n  📊 Average detections per image: {avg_blue:.2f}")
        print(f"  📊 Total objects detected: {total_detections}")

        # Identify problem files
        missed_files = [r["filename"] for r in blue_results if r["blue_detected"] == 0]
        multi_detected = [r["filename"] for r in blue_results if r["total_shapes"] > 1]
        
        if missed_files:
            print(f"\n  🔍 Missed detections ({len(missed_files)} files):")
            for fname in missed_files[:5]:
                print(f"     - {fname}")
            if len(missed_files) > 5:
                print(f"     ... and {len(missed_files) - 5} more")
        
        if multi_detected:
            print(f"\n  ⚠️  Multiple detections per image ({len(multi_detected)} files):")
            for fname in multi_detected[:3]:
                for r in blue_results:
                    if r["filename"] == fname:
                        print(f"     - {fname} ({r['total_shapes']} shapes detected)")
                        break
            if len(multi_detected) > 3:
                print(f"     ... and {len(multi_detected) - 3} more")

    # Red
    red_results = results.get("red_objects", [])
    if red_results:
        print(f"\n📕 RED OBJECTS FOLDER ({len(red_results)} images):")
        print("-" * 60)
        
        detected_red = sum(1 for r in red_results if r["red_detected"] > 0)
        missed_red = len(red_results) - detected_red
        false_blue = sum(1 for r in red_results if r["blue_detected"] > 0)
        no_detection = sum(1 for r in red_results if r["total_shapes"] == 0)
        
        detection_rate = detected_red / len(red_results) * 100
        false_positive_rate = false_blue / len(red_results) * 100
        
        # Color-code the metrics
        rate_symbol = "🟢" if detection_rate >= 90 else "🟡" if detection_rate >= 70 else "🔴"
        false_symbol = "🟢" if false_positive_rate < 10 else "🟡" if false_positive_rate < 20 else "🔴"

        print(f"  {rate_symbol} Successfully detected red:  {detected_red:3d}/{len(red_results)} ({detection_rate:5.1f}%)")
        print(f"  ❌ Missed red detections:      {missed_red:3d}/{len(red_results)} ({100-detection_rate:5.1f}%)")
        print(f"  {false_symbol} False positives (as blue):   {false_blue:3d}/{len(red_results)} ({false_positive_rate:5.1f}%)")
        print(f"  ⚠️  No detection at all:        {no_detection:3d}/{len(red_results)}")

        avg_red = sum(r["red_detected"] for r in red_results) / len(red_results)
        total_detections = sum(r["total_shapes"] for r in red_results)
        print(f"\n  📊 Average detections per image: {avg_red:.2f}")
        print(f"  📊 Total objects detected: {total_detections}")

        # Identify problem files
        missed_files = [r["filename"] for r in red_results if r["red_detected"] == 0]
        multi_detected = [r["filename"] for r in red_results if r["total_shapes"] > 1]
        
        if missed_files:
            print(f"\n  🔍 Missed detections ({len(missed_files)} files):")
            for fname in missed_files[:5]:
                print(f"     - {fname}")
            if len(missed_files) > 5:
                print(f"     ... and {len(missed_files) - 5} more")
        
        if multi_detected:
            print(f"\n  ⚠️  Multiple detections per image ({len(multi_detected)} files):")
            for fname in multi_detected[:3]:
                for r in red_results:
                    if r["filename"] == fname:
                        print(f"     - {fname} ({r['total_shapes']} shapes detected)")
                        break
            if len(multi_detected) > 3:
                print(f"     ... and {len(multi_detected) - 3} more")

    # Overall Statistics
    print(f"\n" + "="*60)
    print("🎯 OVERALL STATISTICS")
    print("="*60)
    
    total_images = len(blue_results) + len(red_results)
    if total_images > 0:
        correct_detections = (sum(1 for r in blue_results if r["blue_detected"] > 0) +
                              sum(1 for r in red_results if r["red_detected"] > 0))
        overall_accuracy = correct_detections / total_images * 100
        
        # Overall accuracy indicator
        if overall_accuracy >= 95:
            indicator = "🟢 EXCELLENT"
        elif overall_accuracy >= 80:
            indicator = "🟡 GOOD"
        elif overall_accuracy >= 60:
            indicator = "🟠 FAIR"
        else:
            indicator = "🔴 POOR"
        
        print(f"  Overall Detection Rate: {correct_detections}/{total_images} ({overall_accuracy:.1f}%) {indicator}")
        
        # Additional metrics
        total_shapes = sum(r["total_shapes"] for r in blue_results + red_results)
        total_correct = (sum(1 for r in blue_results if r["blue_detected"] > 0) + 
                        sum(1 for r in red_results if r["red_detected"] > 0))
        
        print(f"  Total Images: {total_images}")
        print(f"  Total Shapes Detected: {total_shapes}")
        print(f"  Average per Image: {total_shapes/total_images:.2f}")

def generate_csv_report(results, output_file: str = "detection_report.csv"):
    """
    Generate a detailed CSV report of all detection results
    for further analysis in spreadsheet tools
    """
    import csv
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['folder', 'filename', 'blue_detected', 'red_detected', 'total_shapes'])
            writer.writeheader()
            
            for key, records in results.items():
                for record in records:
                    writer.writerow({
                        'folder': key,
                        'filename': record['filename'],
                        'blue_detected': record['blue_detected'],
                        'red_detected': record['red_detected'],
                        'total_shapes': record['total_shapes']
                    })
        
        print(f"\n📄 Detailed CSV report saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Failed to save CSV report: {e}")

def generate_json_report(results, output_file: str = "detection_report.json"):
    """
    Generate a detailed JSON report with statistics
    for programmatic processing and visualization
    """
    try:
        # Calculate summary statistics
        blue_results = results.get("blue_objects", [])
        red_results = results.get("red_objects", [])
        
        total_images = len(blue_results) + len(red_results)
        blue_detected = sum(1 for r in blue_results if r["blue_detected"] > 0)
        red_detected = sum(1 for r in red_results if r["red_detected"] > 0)
        
        report = {
            "timestamp": str(__import__('datetime').datetime.now()),
            "summary": {
                "total_images": total_images,
                "blue_detection_rate": (blue_detected / len(blue_results) * 100) if blue_results else 0,
                "red_detection_rate": (red_detected / len(red_results) * 100) if red_results else 0,
                "overall_accuracy": ((blue_detected + red_detected) / total_images * 100) if total_images > 0 else 0
            },
            "blue_objects": {
                "total": len(blue_results),
                "detected": blue_detected,
                "missed": len(blue_results) - blue_detected,
                "results": blue_results
            },
            "red_objects": {
                "total": len(red_results),
                "detected": red_detected,
                "missed": len(red_results) - red_detected,
                "results": red_results
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Detailed JSON report saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Failed to save JSON report: {e}")

def _parse_args():
    p = argparse.ArgumentParser(
        description="Process dataset images with improved shape detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_processor.py --dataset dataset --output processed
  python dataset_processor.py --dataset dataset --sample 20
  python dataset_processor.py --dataset dataset --sample 20 --random
  python dataset_processor.py --dataset dataset --output processed --csv report.csv --json report.json --log debug
        """
    )
    p.add_argument("--dataset", default="dataset", help="Path to dataset folder (default: dataset)")
    p.add_argument("--output", help="Path to save processed images")
    p.add_argument("--show", action="store_true", help="Show processed images (requires display)")
    p.add_argument("--sample", type=int, help="Process only N images from each folder")
    p.add_argument("--random", action="store_true", help="Randomly select sample images (instead of first N)")
    p.add_argument("--csv", help="Save detection results to CSV file")
    p.add_argument("--json", help="Save detection results to JSON file with statistics")
    p.add_argument("--no-shape-module", action="store_true", help="Do not use shape_detector module even if present")
    p.add_argument("--log", default="info", choices=["debug", "info", "warning", "error"], 
                   help="Log level (default: info)")
    return p.parse_args()

def main():
    args = _parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("dataset_processor")

    if args.show and is_headless():
        logger.warning("Display not detected: --show will be ignored in headless environment")
        args.show = False

    logger.info("=" * 60)
    logger.info("DATASET PROCESSOR - Starting")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  Dataset path: %s", args.dataset)
    logger.info("  Output path: %s", args.output if args.output else "None (display only)")
    logger.info("  Show images: %s", args.show)
    if args.sample:
        logger.info("  Sample size: %d %s", args.sample, "(random)" if args.random else "(sequential)")
    else:
        logger.info("  Sample size: All images")
    logger.info("  Use shape_module: %s", not args.no_shape_module and HAVE_SHAPE_MODULE)
    if args.csv:
        logger.info("  CSV report: %s", args.csv)
    if args.json:
        logger.info("  JSON report: %s", args.json)
    logger.info("=" * 60)

    # Check if dataset path exists
    if not Path(args.dataset).exists():
        logger.error("Dataset path does not exist: %s", args.dataset)
        return

    results = process_dataset(
        args.dataset,
        args.output,
        args.show,
        args.sample,
        random_sample=args.random,
        use_shape_module=not args.no_shape_module
    )
    
    # Print summary report
    print_summary(results)
    
    # Save CSV report if requested
    if args.csv:
        generate_csv_report(results, args.csv)
    
    # Save JSON report if requested
    if args.json:
        generate_json_report(results, args.json)
    
    # Log completion
    if args.output:
        logger.info("✓ Processed images saved to: %s", args.output)
    
    logger.info("=" * 60)
    logger.info("DATASET PROCESSOR - Complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
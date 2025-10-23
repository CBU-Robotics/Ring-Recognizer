#!/usr/bin/env python3
"""
Dataset processor for Ring-Recognizer project
Processes images from the dataset folder and applies detection algorithms
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def detect_shapes_simple(image):
    """
    Simple shape detection based on your block cam logic
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red and blue shapes
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 50, 50])     # Lower red range
    upper_red1 = np.array([10, 255, 255])  
    lower_red2 = np.array([170, 50, 50])   # Upper red range
    upper_red2 = np.array([180, 255, 255]) 
    
    # Blue range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create masks for red and blue
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    # Apply morphological operations to clean up masks
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    result = image.copy()
    detected_shapes = []
    
    # Process both red and blue shapes
    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), 
                                   ("blue", blue_mask, (255, 0, 0))]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter small noise
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Basic size filter - reject very small or very large shapes
            if w < 20 or h < 20 or w > 500 or h > 500:
                continue
                
            cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
            cv2.putText(result, f"{color} shape", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
            detected_shapes.append((color, (x, y, w, h), area))
    
    return result, detected_shapes

def process_dataset(dataset_path, output_path=None, show_images=False, sample_size=None):
    """
    Process all images in the dataset folder
    """
    dataset_path = Path(dataset_path)
    
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
    
    results = {
        "blue_objects": [],
        "red_objects": []
    }
    
    # Process blue objects
    blue_folder = dataset_path / "Blue objects"
    if blue_folder.exists():
        image_files = sorted(blue_folder.glob("*.jpg"))
        if sample_size:
            image_files = image_files[:sample_size]
        
        print(f"Processing {len(image_files)} blue object images...")
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
                
            processed, detections = detect_shapes_simple(image)
            
            # Count detections
            blue_count = sum(1 for d in detections if d[0] == "blue")
            red_count = sum(1 for d in detections if d[0] == "red")
            
            results["blue_objects"].append({
                "filename": img_file.name,
                "blue_detected": blue_count,
                "red_detected": red_count,
                "total_shapes": len(detections)
            })
            
            if show_images:
                cv2.imshow(f"Blue Objects - {img_file.name}", processed)
                cv2.waitKey(500)  # Show for 500ms
            
            if output_path:
                output_file = output_path / f"processed_blue_{img_file.name}"
                cv2.imwrite(str(output_file), processed)
    
    # Process red objects
    red_folder = dataset_path / "Red objects"
    if red_folder.exists():
        image_files = sorted(red_folder.glob("*.jpg"))
        if sample_size:
            image_files = image_files[:sample_size]
            
        print(f"Processing {len(image_files)} red object images...")
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
                
            processed, detections = detect_shapes_simple(image)
            
            # Count detections
            blue_count = sum(1 for d in detections if d[0] == "blue")
            red_count = sum(1 for d in detections if d[0] == "red")
            
            results["red_objects"].append({
                "filename": img_file.name,
                "blue_detected": blue_count,
                "red_detected": red_count,
                "total_shapes": len(detections)
            })
            
            if show_images:
                cv2.imshow(f"Red Objects - {img_file.name}", processed)
                cv2.waitKey(500)  # Show for 500ms
            
            if output_path:
                output_file = output_path / f"processed_red_{img_file.name}"
                cv2.imwrite(str(output_file), processed)
    
    if show_images:
        cv2.destroyAllWindows()
    
    return results

def print_summary(results):
    """
    Print a summary of detection results
    """
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    
    # Blue objects folder analysis
    blue_results = results["blue_objects"]
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
        
        # Show some examples of failures
        missed_files = [r["filename"] for r in blue_results if r["blue_detected"] == 0]
        if missed_files:
            print(f"  🔍 Examples of missed detections: {', '.join(missed_files[:3])}")
    
    # Red objects folder analysis  
    red_results = results["red_objects"]
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
        
        # Show some examples of failures
        missed_files = [r["filename"] for r in red_results if r["red_detected"] == 0]
        if missed_files:
            print(f"  � Examples of missed detections: {', '.join(missed_files[:3])}")
    
    # Overall accuracy
    total_images = len(blue_results) + len(red_results)
    correct_detections = (sum(1 for r in blue_results if r["blue_detected"] > 0) + 
                         sum(1 for r in red_results if r["red_detected"] > 0))
    
    if total_images > 0:
        overall_accuracy = correct_detections / total_images * 100
        print(f"\n🎯 OVERALL ACCURACY: {correct_detections}/{total_images} ({overall_accuracy:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Process dataset images with shape detection")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder")
    parser.add_argument("--output", help="Path to save processed images")
    parser.add_argument("--show", action="store_true", help="Show processed images")
    parser.add_argument("--sample", type=int, help="Process only first N images from each folder")
    
    args = parser.parse_args()
    
    print("Starting dataset processing...")
    results = process_dataset(args.dataset, args.output, args.show, args.sample)
    print_summary(results)
    
    if args.output:
        print(f"\nProcessed images saved to: {args.output}")

if __name__ == "__main__":
    main()

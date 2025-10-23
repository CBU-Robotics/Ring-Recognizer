import cv2
import numpy as np
import math


CAMERA_ANGLE = 45.0
CAMERA_HEIGHT_INCH = 10.5  # Height of the camera from the ground in inches
CAMERA_ROBOT_Y_DELTA = 0.0  # This can be used to adjust the Y position of the camera relative to the robot, if needed


def detect_rings(frame):
    """
    Modified function to detect rings without human detection filtering
    Tracks both circular and elliptical shapes with improved consistency
    """
    # Make a copy of the frame
    result = frame.copy()
    
    # Apply Gaussian blur to reduce noise and improve consistency
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Define color ranges with broader thresholds for red
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 60, 40])     # Lower red range (slightly more permissive)
    upper_red1 = np.array([12, 255, 255])  
    lower_red2 = np.array([158, 60, 40])   # Upper red range (slightly more permissive)
    upper_red2 = np.array([180, 255, 255]) 
    
    # Blue range (slightly more permissive)
    lower_blue = np.array([90, 70, 40])
    upper_blue = np.array([132, 255, 255])
    
    # Create masks for red and blue
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    # Apply morphological operations with different kernel sizes for better consistency
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    # First pass: remove small noise
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Second pass: fill gaps
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_large)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Show debug masks
    debug_view = np.zeros_like(frame)
    debug_view[:, :, 0] = blue_mask  # Blue channel
    debug_view[:, :, 2] = red_mask   # Red channel
    cv2.imshow("Color Masks", debug_view)
    
    # Track detected rings for each color
    detected_rings = []
    
    # Process both red and blue rings
    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), 
                                   ("blue", blue_mask, (255, 0, 0))]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first) for consistent processing
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:  # Reduced minimum area for better detection
                continue
                
            # Calculate shape metrics
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio to filter out non-ring shapes
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # More lenient aspect ratio for rings at various angles
            if aspect_ratio < 0.3 or aspect_ratio > 2.0:
                continue
                
            # Calculate circularity and compactness
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # More flexible circularity check - rings can appear quite different from various angles
            if 0.2 < circularity < 1.8:
                # Additional check: ensure the contour has reasonable solidity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Rings should have reasonable solidity (not too fragmented)
                if solidity > 0.6:
                    cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                    cv2.putText(result, f"{color} ring", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                    detected_rings.append((color, (x, y, w, h)))
    
    return result, detected_rings


def determine_absolute_position(x, y, w, h, frame_width, frame_height):
    """
    Determine the absolute position of the ring based on its bounding box.
    This function can be used to calculate the center or any other position.
    """

    frame_height_in = CAMERA_HEIGHT_INCH*math.tan(math.radians(CAMERA_ANGLE)) # Height of the frame in inches based on camera angle and height
    frame_width_in = (frame_height_in/frame_height)*frame_width # Width of the frame in inches based on height

    pixel_to_inch_ratio = frame_width_in / frame_width # Calculate the ratio of inches to pixels

    print(f"Frame dimensions in inches: {frame_width_in:.2f} x {frame_height_in:.2f}")

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    center_x_in = ((center_x+(frame_width/2)-frame_width) / frame_width) * frame_width_in # Convert pixel x coordinate to inches
    center_y_in = ((-center_y+frame_height) / frame_height) * frame_height_in + CAMERA_ROBOT_Y_DELTA # Convert pixel y coordinate to inches

    center_x_in = round(center_x_in, 2)  # Round to 2 decimal places for readability
    center_y_in = round(center_y_in, 2)  # Round to 2 decimal places for readability
    
    return (center_x_in, center_y_in)


def main():
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    print("Press 'q' to exit")
    
    # Parameters for tracking stability
    prev_rings = []
    ring_history = []  # Track ring positions over time for stability
    frame_count = 0
    
    while True:
        try:    
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame from video stream")
                continue
        except Exception as e:
            # Handle the case where frame reading fails
            print(f"Error reading frame: {e}")
            continue

        frame_height, frame_width = frame.shape[:2]
        result, current_rings = detect_rings(frame)
        
        # Improved tracking for stability
        frame_count += 1
        
        # Store ring history for smoothing
        ring_history.append(current_rings)
        if len(ring_history) > 5:  # Keep last 5 frames
            ring_history.pop(0)
        
        # Use current rings if detected, otherwise use previous stable detection
        stable_rings = current_rings if len(current_rings) > 0 else prev_rings
        
        # Only update prev_rings if we have a confident detection
        if len(current_rings) > 0:
            # Check if detection is consistent with recent history
            if len(ring_history) >= 3:
                recent_detections = sum(len(rings) for rings in ring_history[-3:])
                if recent_detections >= 3:  # At least 1 ring detected per frame in last 3 frames
                    prev_rings = current_rings
            else:
                prev_rings = current_rings
        
        # Draw stable rings
        for color, (x, y, w, h) in stable_rings:
            bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
            position = determine_absolute_position(x, y, w, h, frame_width, frame_height)
            cv2.putText(result, f"{color} {position}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
        
        # Show result
        cv2.imshow("Ring Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

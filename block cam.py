import cv2
import numpy as np

# Ball Detection System
# Detects red and blue colored balls

from common_utils import determine_absolute_position, create_color_masks, CAMERA_ANGLE, CAMERA_HEIGHT_INCH, CAMERA_ROBOT_Y_DELTA


def detect_balls(frame):
    """
    Detect red and blue balls using circular shape detection
    """
    # Make a copy of the frame
    result = frame.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create masks for red and blue using common utilities
    red_mask, blue_mask = create_color_masks(hsv_frame)
    
    # Apply morphological operations to clean up masks
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    # Show debug masks
    debug_view = np.zeros_like(frame)
    debug_view[:, :, 0] = blue_mask  # Blue channel
    debug_view[:, :, 2] = red_mask   # Red channel
    cv2.imshow("Color Masks", debug_view)
    
    # Track detected balls for each color
    detected_balls = []
    
    # Process both red and blue balls
    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), 
                                   ("blue", blue_mask, (255, 0, 0))]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Process each contour to find ball-like shapes
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small noise (smaller threshold for balls)
                continue
            
            # Use circular bounding instead of rectangular
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            center_x, center_y = int(center_x), int(center_y)
            radius = int(radius)
            
            # Size filter for balls - reasonable ball sizes
            if radius < 8 or radius > 150:
                continue
            
            # Calculate circularity to filter for ball-like shapes
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Balls should have high circularity (close to 1.0)
                if circularity < 0.3:  # Relaxed threshold for real-world conditions
                    continue
            
            # Draw circle bounding
            cv2.circle(result, (center_x, center_y), radius, bbox_color, 2)
            cv2.putText(result, f"{color} ball", (center_x - radius, center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
            
            # For compatibility with position calculation, convert circle to bounding box
            x, y = center_x - radius, center_y - radius
            w, h = radius * 2, radius * 2
            detected_balls.append((color, (center_x, center_y), radius, (x, y, w, h)))
    
    return result, detected_balls


def main():
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    print("Press 'q' to exit")
    
    # Parameters for tracking stability
    prev_shapes = []
    
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
        result, current_balls = detect_balls(frame)
        
        # Basic tracking to stabilize detection (prevent flickering)
        if len(current_balls) > 0:
            prev_shapes = current_balls
        
        # Draw detected balls
        for ball_data in prev_shapes:
            if len(ball_data) == 4:  # New circular format: color, (center_x, center_y), radius, (x, y, w, h)
                color, (center_x, center_y), radius, (x, y, w, h) = ball_data
            elif len(ball_data) == 3:  # Old format with radius
                color, (x, y, w, h), radius = ball_data
                center_x, center_y = x + w//2, y + h//2
            else:  # Backward compatibility
                color, (x, y, w, h) = ball_data
                radius = max(w, h) // 2
                center_x, center_y = x + w//2, y + h//2
                
            bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
            cv2.circle(result, (center_x, center_y), radius, bbox_color, 2)
            position = determine_absolute_position(x, y, w, h, frame_width, frame_height)
            cv2.putText(result, f"{color} ball {position}", (center_x - radius, center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
        
        # Show result
        cv2.imshow("Ball Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
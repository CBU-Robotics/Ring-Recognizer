import cv2
import numpy as np

# Simple Shape Detection System
# Detects red and blue colored shapes

from common_utils import determine_absolute_position, CAMERA_ANGLE, CAMERA_HEIGHT_INCH, CAMERA_ROBOT_Y_DELTA


def detect_blocks(frame):
    """
    Simplified function to detect red and blue shapes
    """
    # Make a copy of the frame
    result = frame.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
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
    
    # Show debug masks
    debug_view = np.zeros_like(frame)
    debug_view[:, :, 0] = blue_mask  # Blue channel
    debug_view[:, :, 2] = red_mask   # Red channel
    cv2.imshow("Color Masks", debug_view)
    
    # Track detected shapes for each color
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
            detected_shapes.append((color, (x, y, w, h)))
    
    return result, detected_shapes


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
        result, current_shapes = detect_blocks(frame)
        
        # Basic tracking to stabilize boxes (prevent flickering)
        if len(current_shapes) > 0:
            prev_shapes = current_shapes
        
        # Draw detected shapes
        for color, (x, y, w, h) in prev_shapes:
            bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
            position = determine_absolute_position(x, y, w, h, frame_width, frame_height)
            cv2.putText(result, f"{color} {position}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
        
        # Show result
        cv2.imshow("Shape Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
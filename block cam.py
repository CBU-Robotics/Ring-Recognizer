import cv2
import numpy as np

# Ball Detection System
# Detects red and blue colored balls

from common_utils import determine_absolute_position, create_color_masks, CAMERA_ANGLE, CAMERA_HEIGHT_INCH, CAMERA_ROBOT_Y_DELTA


def detect_balls(frame):
    """
    Detect red and blue objects using improved filtering (adapted from dataset processor)
    """
    # Make a copy of the frame
    result = frame.copy()
    
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Apply more blur to reduce noise (matching dataset processor)
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create masks for red and blue using common utilities
    red_mask, blue_mask = create_color_masks(hsv_frame)
    
    # Larger morphological operations to clean up masks (matching dataset processor)
    k = max(5, int(min(w, h) / 150))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Show debug masks
    debug_view = np.zeros_like(frame)
    debug_view[:, :, 0] = blue_mask  # Blue channel
    debug_view[:, :, 2] = red_mask   # Red channel
    cv2.imshow("Color Masks", debug_view)
    
    # Track detected objects
    detected_balls = []
    
    # Minimum area based on frame size (matching dataset processor)
    min_area_px = max(2000, int((w * h) * 0.001))
    center_bias = 0.3
    
    # Process both red and blue objects
    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), 
                                   ("blue", blue_mask, (255, 0, 0))]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours using improved criteria
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue
                
            # Get contour properties
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Filter out edge objects (too close to frame borders)
            border_margin = min(w, h) * 0.05  # 5% margin from edges
            if (x < border_margin or y < border_margin or 
                x + w_rect > w - border_margin or y + h_rect > h - border_margin):
                continue
            
            # Filter by aspect ratio (avoid very elongated shapes like edges)
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            if aspect_ratio > 3 or aspect_ratio < 0.33:  # Too elongated
                continue
            
            # Calculate distance from center (prefer center objects)
            contour_center_x = x + w_rect // 2
            contour_center_y = y + h_rect // 2
            dist_from_center = np.sqrt((contour_center_x - center_x)**2 + (contour_center_y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_score = 1.0 - (dist_from_center / max_dist)
            
            # Calculate circularity/compactness
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            valid_contours.append((contour, area, center_score, circularity))
        
        # Sort by composite score: area + center bias + circularity
        valid_contours.sort(key=lambda x: x[1] * (1 + center_bias * x[2]) * (1 + 0.5 * x[3]), reverse=True)
        
        # Take only the best contour(s) - usually just the main object
        for i, (contour, area, center_score, circularity) in enumerate(valid_contours[:2]):  # Max 2 objects per color
            # Use circular bounding instead of rectangular
            (center_x_obj, center_y_obj), radius = cv2.minEnclosingCircle(contour)
            center_x_obj, center_y_obj = int(center_x_obj), int(center_y_obj)
            radius = int(radius)
            
            # Draw circle bounding
            cv2.circle(result, (center_x_obj, center_y_obj), radius, bbox_color, 3)
            cv2.putText(result, f"{color} object", (center_x_obj - radius, center_y_obj - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
            
            # For compatibility with position calculation, convert circle to bounding box
            x, y = center_x_obj - radius, center_y_obj - radius
            w_box, h_box = radius * 2, radius * 2
            detected_balls.append((color, (center_x_obj, center_y_obj), radius, (x, y, w_box, h_box)))
    
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
        
        # Draw detected objects
        for ball_data in prev_shapes:
            if len(ball_data) == 4:  # New improved format: color, (center_x, center_y), radius, (x, y, w, h)
                color, (center_x, center_y), radius, (x, y, w, h) = ball_data
            elif len(ball_data) == 3:  # Old format with radius
                color, (x, y, w, h), radius = ball_data
                center_x, center_y = x + w//2, y + h//2
            else:  # Backward compatibility
                color, (x, y, w, h) = ball_data
                radius = max(w, h) // 2
                center_x, center_y = x + w//2, y + h//2
                
            bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
            cv2.circle(result, (center_x, center_y), radius, bbox_color, 3)
            position = determine_absolute_position(x, y, w, h, frame_width, frame_height)
            cv2.putText(result, f"{color} object {position}", (center_x - radius, center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
        
        # Show result
        cv2.imshow("Object Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
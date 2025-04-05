import cv2
import numpy as np
import math


CAMERA_ANGLE = 45.0
CAMERA_HEIGHT_INCH = 10.5  # Height of the camera from the ground in inches
CAMERA_ROBOT_Y_DELTA = 0.0  # This can be used to adjust the Y position of the camera relative to the robot, if needed


def detect_rings(frame):
    """
    Modified function to detect rings without human detection filtering
    Tracks both circular and elliptical shapes
    """
    # Make a copy of the frame
    result = frame.copy()
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges with broader thresholds for red
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 70, 50])     # Lower red range
    upper_red1 = np.array([10, 255, 255])  
    lower_red2 = np.array([160, 70, 50])   # Upper red range
    upper_red2 = np.array([180, 255, 255]) 
    
    # Blue range
    lower_blue = np.array([92, 80, 50])
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
    
    # Track detected rings for each color
    detected_rings = []
    ring_sizes = []  # To store sizes of detected rings for tracking stability
    ring_positions = []  # To store positions of detected rings for tracking stability
    
    # Process both red and blue rings
    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), 
                                   ("blue", blue_mask, (255, 0, 0))]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Filter small noise
                continue
                
            # Calculate shape metrics
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio to filter out non-ring shapes
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Skip if aspect ratio is too extreme (probably not a ring)
            # Allow wider range to catch rings from different angles
            if aspect_ratio < 0.4 or aspect_ratio > 1.4:
                continue
                
            # Calculate circularity (for front view) or elongation (for side view)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # For front view: circularity closer to 1
            # For side view: more elongated with lower circularity
            # Accept a wider range to catch different viewing angles
            if 0.4 < circularity < 1.5:
                cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                #cv2.putText(result, f"{color} ring", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
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
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    print("Press 'q' to exit")
    
    # Parameters for tracking stability
    prev_rings = []
    
    while True:
        try:    
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame from video stream")
                continue
        except Exception as e:
            # Handle the case where frame reading fails
            print(f"Error reading frame: {e}")

        frame_height, frame_width = frame.shape[:2]
        result, current_rings = detect_rings(frame)
        
        # Basic tracking to stabilize boxes (prevent flickering)
        if len(current_rings) > 0:
            prev_rings = current_rings
            for color, (x, y, w, h) in prev_rings:
                bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
                cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                cv2.putText(result, f"{color} {determine_absolute_position(x,y,w,h,frame_width,frame_height)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
        
        # Show result
        cv2.imshow("Ring Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
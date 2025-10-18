import cv2
import numpy as np
import math

# Block Detection System
# Detects 18-sided hollow plastic polygonal blocks (red/blue)
# Block specifications: 40g weight, 3.25" between flat faces, 3.85" between corners

CAMERA_ANGLE = 45.0
CAMERA_HEIGHT_INCH = 10.5  # Height of the camera from the ground in inches
CAMERA_ROBOT_Y_DELTA = 0.0  # This can be used to adjust the Y position of the camera relative to the robot, if needed


def detect_blocks(frame):
    """
    Modified function to detect 18-sided polygonal blocks (red/blue)
    Tracks polygonal shapes using vertex counting and approximation
    """
    # Make a copy of the frame
    result = frame.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple face detection to create exclusion zones
    # Load face cascade (built into OpenCV)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    except:
        faces = []  # If face detection fails, continue without it
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges specifically for plastic blocks (tighter to avoid clothing/faces)
    # Red has two ranges in HSV - made more restrictive
    lower_red1 = np.array([0, 80, 80])     # Lower red range (higher saturation/value)
    upper_red1 = np.array([8, 255, 255])  
    lower_red2 = np.array([172, 80, 80])   # Upper red range (higher saturation/value)
    upper_red2 = np.array([180, 255, 255]) 
    
    # Blue range (more restrictive for plastic blocks)
    lower_blue = np.array([105, 80, 80])   # Higher saturation and value to avoid clothing
    upper_blue = np.array([125, 255, 255])
    
    # Create masks for red and blue
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    # Apply morphological operations to clean up masks (adjusted for blocks)
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for better polygon preservation
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    # Show debug masks
    debug_view = np.zeros_like(frame)
    debug_view[:, :, 0] = blue_mask  # Blue channel
    debug_view[:, :, 2] = red_mask   # Red channel
    cv2.imshow("Color Masks", debug_view)
    
    # Track detected blocks for each color
    detected_blocks = []
    block_sizes = []  # To store sizes of detected blocks for tracking stability
    block_positions = []  # To store positions of detected blocks for tracking stability
    
    # Process both red and blue blocks
    for color, mask, bbox_color in [("red", red_mask, (0, 0, 255)), 
                                   ("blue", blue_mask, (255, 0, 0))]:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter small noise
                continue
                
            # Calculate shape metrics
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this contour overlaps with any detected face
            overlaps_face = False
            for (fx, fy, fw, fh) in faces:
                # Check if contour bounding box overlaps with face bounding box
                if (x < fx + fw and x + w > fx and y < fy + fh and y + h > fy):
                    overlaps_face = True
                    break
            
            if overlaps_face:
                continue  # Skip anything that overlaps with a detected face
            
            # Calculate aspect ratio to filter out non-block shapes
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Skip faces and very large objects (likely humans)
            # Blocks should be much smaller than faces
            if (area > 15000 or  # Too large (likely face/body)
                aspect_ratio < 0.6 or aspect_ratio > 1.8 or  # Blocks should be roughly square
                w > 200 or h > 200):  # Size limit in pixels
                continue
            
            # Use polygon approximation to count vertices
            # Approximate the contour to a polygon
            epsilon = 0.02 * perimeter  # Approximation accuracy parameter
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            vertex_count = len(approx_polygon)
            
            # Calculate additional geometric properties to filter out faces/non-blocks
            # 1. Solidity: ratio of contour area to convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 2. Extent: ratio of contour area to bounding rectangle area
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # 3. Compactness: circularity measure
            compactness = (perimeter * perimeter) / area if area > 0 else 0
            
            # Filter criteria for blocks (reject faces and irregular shapes):
            # - Blocks should be fairly solid (high solidity)
            # - Blocks should fill most of their bounding box (reasonable extent)
            # - Blocks should have reasonable compactness
            # - Area should be appropriate for blocks at camera distance
            if (solidity > 0.85 and  # High solidity - blocks are solid shapes
                extent > 0.6 and extent < 0.95 and  # Good extent - fills bounding box well
                compactness < 20 and  # Reasonable compactness
                area > 1000 and area < 15000 and  # Appropriate size range for blocks
                12 <= vertex_count <= 24):  # Polygon vertex count for 18-sided blocks
                
                cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                cv2.putText(result, f"{color} block (v:{vertex_count})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
                detected_blocks.append((color, (x, y, w, h)))
    
    return result, detected_blocks


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
    prev_blocks = []
    
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
        result, current_blocks = detect_blocks(frame)
        
        # Basic tracking to stabilize boxes (prevent flickering)
        if len(current_blocks) > 0:
            prev_blocks = current_blocks
            for color, (x, y, w, h) in prev_blocks:
                bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
                cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                cv2.putText(result, f"{color} block {determine_absolute_position(x,y,w,h,frame_width,frame_height)}", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
        
        # Show result
        cv2.imshow("Block Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
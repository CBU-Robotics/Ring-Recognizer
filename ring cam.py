import cv2
import numpy as np

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
    lower_blue = np.array([90, 80, 50])
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
            if 0.2 < circularity < 1.5:
                cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                cv2.putText(result, f"{color} ring", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                detected_rings.append((color, (x, y, w, h)))
    
    return result, detected_rings

def main():
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    print("Press 'q' to exit")
    
    # Parameters for tracking stability
    prev_rings = []
    stability_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result, current_rings = detect_rings(frame)
        
        # Basic tracking to stabilize boxes (prevent flickering)
        if len(current_rings) > 0:
            # Update the tracking
            prev_rings = current_rings
            stability_counter = 5  # Keep showing for 5 frames even if detection is lost
        elif stability_counter > 0:
            # No rings detected but we'll show the previous ones for stability
            stability_counter -= 1
            for color, (x, y, w, h) in prev_rings:
                bbox_color = (0, 0, 255) if color == "red" else (255, 0, 0)
                cv2.rectangle(result, (x, y), (x + w, y + h), bbox_color, 2)
                cv2.putText(result, f"{color} ring (tracked)", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
        
        # Show result
        cv2.imshow("Ring Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
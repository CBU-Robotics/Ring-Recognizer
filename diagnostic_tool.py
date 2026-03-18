#!/usr/bin/env python3
"""
Diagnostic Tool for Ring-Recognizer
This tool helps visualize and debug detection issues in both block_cam.py and ring_cam.py
"""

import cv2
import numpy as np
from common_utils import create_color_masks, determine_absolute_position

class DiagnosticViewer:
    """Comprehensive diagnostic viewer for object detection"""
    
    def __init__(self):
        self.frame = None
        self.hsv_frame = None
        self.red_mask = None
        self.blue_mask = None
        
    def analyze_frame(self, frame):
        """Analyze a single frame and display diagnostic info"""
        self.frame = frame
        h, w = frame.shape[:2]
        
        # Create HSV version
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        self.hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create masks
        self.red_mask, self.blue_mask = create_color_masks(self.hsv_frame)
        
        # Create a comprehensive display
        self._display_diagnostics()
        
    def _display_diagnostics(self):
        """Display multiple diagnostic views"""
        h, w = self.frame.shape[:2]
        
        # 1. Original Frame
        display = self.frame.copy()
        
        # 2. Color Masks Panel
        mask_display = np.zeros((h, w, 3), dtype=np.uint8)
        mask_display[:, :, 0] = self.blue_mask   # Blue channel for blue mask
        mask_display[:, :, 2] = self.red_mask    # Red channel for red mask
        
        # 3. Contour Analysis
        red_contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(self.blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_display = self.frame.copy()
        
        # Draw all red contours with statistics
        for i, contour in enumerate(red_contours):
            area = cv2.contourArea(contour)
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            
            cv2.drawContours(contour_display, [contour], 0, (0, 0, 255), 2)
            cv2.putText(contour_display, 
                       f"R{i}: A={area:.0f} AR={aspect_ratio:.2f} C={circularity:.2f}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw all blue contours with statistics
        for i, contour in enumerate(blue_contours):
            area = cv2.contourArea(contour)
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            
            cv2.drawContours(contour_display, [contour], 0, (255, 0, 0), 2)
            cv2.putText(contour_display, 
                       f"B{i}: A={area:.0f} AR={aspect_ratio:.2f} C={circularity:.2f}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 4. Create composite view
        h_half = h // 2
        w_half = w // 2
        
        composite = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        composite[0:h, 0:w] = cv2.resize(self.frame, (w, h))  # Original
        composite[0:h, w:w*2] = cv2.resize(mask_display, (w, h))  # Masks
        composite[h:h*2, 0:w] = cv2.resize(contour_display, (w, h))  # Contours
        
        # Statistics panel
        stats_panel = np.zeros((h, w, 3), dtype=np.uint8)
        stats = [
            f"RED: {len(red_contours)} contours",
            f"BLUE: {len(blue_contours)} contours",
            f"Frame: {w}x{h}",
        ]
        
        y_pos = 30
        for stat in stats:
            cv2.putText(stats_panel, stat, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 40
        
        # Add detailed red contour stats
        if red_contours:
            y_pos += 20
            cv2.putText(stats_panel, "RED CONTOURS:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
            for i, contour in enumerate(red_contours[:5]):  # Show top 5
                area = cv2.contourArea(contour)
                cv2.putText(stats_panel, f"  [{i}] Area: {area:.0f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_pos += 25
        
        # Add detailed blue contour stats
        if blue_contours:
            y_pos += 20
            cv2.putText(stats_panel, "BLUE CONTOURS:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_pos += 30
            for i, contour in enumerate(blue_contours[:5]):  # Show top 5
                area = cv2.contourArea(contour)
                cv2.putText(stats_panel, f"  [{i}] Area: {area:.0f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_pos += 25
        
        composite[h:h*2, w:w*2] = cv2.resize(stats_panel, (w, h))
        
        cv2.imshow("DIAGNOSTIC VIEW - Press 'q' to quit, 's' to save", composite)
        
        return composite, red_contours, blue_contours
    
    def run_live_diagnostic(self):
        """Run live diagnostic on camera feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("Diagnostic Tool - Live Camera Feed")
        print("Press 'q' to quit")
        print("Press 's' to save a frame snapshot")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            self.analyze_frame(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"diagnostic_frame_{frame_count}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main diagnostic entry point"""
    print("=" * 60)
    print("RING-RECOGNIZER DIAGNOSTIC TOOL")
    print("=" * 60)
    print("\nStarting live diagnostic viewer...")
    print("This shows:")
    print("  - Top-left: Original frame")
    print("  - Top-right: Color masks (Red=Red channel, Blue=Blue channel)")
    print("  - Bottom-left: All detected contours with stats")
    print("  - Bottom-right: Summary statistics")
    print("\nWatch for issues like:")
    print("  ✓ Multiple contours detected instead of one per object")
    print("  ✓ Contours touching/merging together")
    print("  ✓ Missing detections (low circularity/aspect ratio)")
    print("  ✓ False positives from background")
    print("\n" + "=" * 60)
    
    viewer = DiagnosticViewer()
    viewer.run_live_diagnostic()


if __name__ == "__main__":
    main()

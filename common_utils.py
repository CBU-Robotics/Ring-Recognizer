#!/usr/bin/env python3
"""
common_utils.py - Shared utilities for Ring-Recognizer project

This module contains common functions and constants used across multiple
files in the Ring-Recognizer project to avoid code duplication.
"""

import math
import cv2
import numpy as np


# -------------------------
# Camera Configuration Constants
# -------------------------
CAMERA_ANGLE = 45.0
CAMERA_HEIGHT_INCH = 10.5  # Height of the camera from the ground in inches
CAMERA_ROBOT_Y_DELTA = 0.0  # This can be used to adjust the Y position of the camera relative to the robot, if needed


# -------------------------
# OpenCV Compatibility Helpers
# -------------------------
def find_contours_compat(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    """
    Wrap cv2.findContours to be compatible across OpenCV versions.
    Returns (contours, hierarchy)
    
    Args:
        mask: Binary image mask
        mode: Contour retrieval mode (default: cv2.RETR_EXTERNAL)
        method: Contour approximation method (default: cv2.CHAIN_APPROX_SIMPLE)
    
    Returns:
        Tuple of (contours, hierarchy)
    """
    res = cv2.findContours(mask, mode, method)
    # OpenCV sometimes returns (contours, hierarchy) or (image, contours, hierarchy)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    return contours, hierarchy


# -------------------------
# Position Calculation
# -------------------------
def determine_absolute_position(x, y, w, h, frame_width, frame_height):
    """
    Determine the absolute position of an object based on its bounding box.
    This function calculates the center position in inches based on camera parameters.
    
    Args:
        x: X coordinate of bounding box top-left corner (pixels)
        y: Y coordinate of bounding box top-left corner (pixels)
        w: Width of bounding box (pixels)
        h: Height of bounding box (pixels)
        frame_width: Width of the frame (pixels)
        frame_height: Height of the frame (pixels)
    
    Returns:
        Tuple of (center_x_in, center_y_in) in inches
    """
    frame_height_in = CAMERA_HEIGHT_INCH * math.tan(math.radians(CAMERA_ANGLE))  # Height of the frame in inches based on camera angle and height
    frame_width_in = (frame_height_in / frame_height) * frame_width  # Width of the frame in inches based on height

    print(f"Frame dimensions in inches: {frame_width_in:.2f} x {frame_height_in:.2f}")

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    center_x_in = ((center_x + (frame_width / 2) - frame_width) / frame_width) * frame_width_in  # Convert pixel x coordinate to inches
    center_y_in = ((-center_y + frame_height) / frame_height) * frame_height_in + CAMERA_ROBOT_Y_DELTA  # Convert pixel y coordinate to inches

    center_x_in = round(center_x_in, 2)  # Round to 2 decimal places for readability
    center_y_in = round(center_y_in, 2)  # Round to 2 decimal places for readability
    
    return (center_x_in, center_y_in)


# -------------------------
# Color Detection Helpers
# -------------------------
def get_color_ranges():
    """
    Get standard HSV color ranges for red and blue detection.
    
    Returns:
        Dictionary containing color range definitions with keys:
        - 'red1': Tuple of (lower_bound, upper_bound) for lower red range
        - 'red2': Tuple of (lower_bound, upper_bound) for upper red range  
        - 'blue': Tuple of (lower_bound, upper_bound) for blue range
    """
    return {
        'red1': (np.array([0, 60, 40]), np.array([12, 255, 255])),
        'red2': (np.array([158, 60, 40]), np.array([180, 255, 255])),
        'blue': (np.array([90, 70, 40]), np.array([132, 255, 255]))
    }


def create_color_masks(hsv_frame):
    """
    Create color masks for red and blue objects from an HSV frame.
    
    Args:
        hsv_frame: Frame in HSV color space
    
    Returns:
        Tuple of (red_mask, blue_mask)
    """
    color_ranges = get_color_ranges()
    
    # Red has two ranges in HSV (wraps around at 0/180)
    lower_red1, upper_red1 = color_ranges['red1']
    lower_red2, upper_red2 = color_ranges['red2']
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Blue range
    lower_blue, upper_blue = color_ranges['blue']
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    return red_mask, blue_mask




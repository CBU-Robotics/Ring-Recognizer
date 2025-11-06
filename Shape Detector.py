#!/usr/bin/env python3
"""
shape_detector.py (improved, renamed from 'Shape Detector.py')

- Robust findContours compatibility
- Optional relaxing of strict vertex equality (accept n +/- slack)
- Exported detection functions for reuse by dataset_processor
- Minor numeric guards and configurable tolerances
"""

import argparse
import math
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from common_utils import find_contours_compat

# -------------------------
# Compatibility helpers
# -------------------------

def approximate_contour(cnt: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = max(0.001, epsilon_factor * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    return approx

def contour_center(cnt: np.ndarray) -> Tuple[float, float]:
    M = cv2.moments(cnt)
    if M.get("m00", 0) == 0:
        pts = cnt.reshape(-1, 2)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])

def side_lengths_from_vertices(verts: np.ndarray) -> np.ndarray:
    pts = verts.reshape(-1, 2)
    diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return dists

def radii_from_centroid(verts: np.ndarray, centroid: Tuple[float, float]) -> np.ndarray:
    pts = verts.reshape(-1, 2)
    diffs = pts - np.array(centroid, dtype=float)
    return np.linalg.norm(diffs, axis=1)

def coef_of_variation(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("inf")
    mean = arr.mean()
    if mean == 0:
        return float("inf")
    return float(arr.std()) / float(mean)

# -------------------------
# Regular polygon test
# -------------------------
def is_regular_polygon(cnt: np.ndarray,
                       n_sides: int,
                       eps_factor: float = 0.02,
                       max_side_cv: float = 0.15,
                       max_radius_cv: float = 0.12,
                       vertex_slack: int = 1) -> Tuple[bool, Dict[str, float]]:
    """
    Test whether contour corresponds to an approximately regular polygon with n_sides.
    vertex_slack allows acceptance of detected vertex count in [n_sides-vertex_slack, n_sides+vertex_slack]
    """
    approx = approximate_contour(cnt, epsilon_factor=eps_factor)
    detected = len(approx)

    metrics = {"detected_vertices": float(detected), "side_cv": float("nan"), "radius_cv": float("nan")}

    if not (n_sides - vertex_slack <= detected <= n_sides + vertex_slack):
        return False, metrics

    verts = approx.reshape(-1, 2)
    side_lens = side_lengths_from_vertices(verts)
    side_cv = coef_of_variation(side_lens)

    c = tuple(map(float, contour_center(cnt)))
    radii = radii_from_centroid(verts, c)
    radius_cv = coef_of_variation(radii)

    metrics["side_cv"] = float(side_cv)
    metrics["radius_cv"] = float(radius_cv)

    is_ok = (side_cv <= max_side_cv) and (radius_cv <= max_radius_cv)
    return bool(is_ok), metrics

# -------------------------
# Detection entry points
# -------------------------
def detect_regular_polygons(image: np.ndarray,
                            n: int = 18,
                            min_area: int = 500,
                            eps_factor: float = 0.02,
                            max_side_cv: float = 0.15,
                            max_radius_cv: float = 0.12,
                            vertex_slack: int = 1,
                            debug: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    k = max(3, int(min(image.shape[:2]) / 200))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = find_contours_compat(edged)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        ok, metrics = is_regular_polygon(cnt, n_sides=n, eps_factor=eps_factor,
                                        max_side_cv=max_side_cv, max_radius_cv=max_radius_cv,
                                        vertex_slack=vertex_slack)
        if not ok:
            if debug:
                print(f"Reject poly: verts={metrics['detected_vertices']}, side_cv={metrics['side_cv']:.3f}, radius_cv={metrics['radius_cv']:.3f}, area={area:.1f}")
            continue

        c = contour_center(cnt)
        detections.append({"n": n, "contour": cnt, "center": (int(round(c[0])), int(round(c[1]))), "area": area, "metrics": metrics})
        approx = approximate_contour(cnt, epsilon_factor=eps_factor)
        cv2.drawContours(out, [approx], -1, (0, 200, 0), 2)
        cx, cy = int(round(c[0])), int(round(c[1]))
        cv2.putText(out, f"{n}-gon", (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if debug:
            print(f"Found regular {n}-gon at {cx,cy} area={area:.1f} metrics={metrics}")

    return out, detections

# Basic contour classifier (kept for compatibility)
def detect_shapes_contours(image: np.ndarray,
                           min_area: int = 100,
                           debug: bool = False):
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = find_contours_compat(edged)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)
        shape = "unknown"
        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h) if h != 0 else 0
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif vertices == 5:
            shape = "pentagon"
        elif vertices == 6:
            shape = "hexagon"
        else:
            circularity = 4 * math.pi * area / (peri * peri) if peri != 0 else 0
            if circularity > 0.7:
                shape = "circle"
            else:
                shape = f"{vertices}-sided"

        M = cv2.moments(cnt)
        cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M.get("m00", 0) != 0 else (0, 0)
        detections.append({"shape": shape, "contour": cnt, "center": (cX, cY), "area": area, "vertices": vertices})
        cv2.drawContours(orig, [approx], -1, (0, 255, 0), 2)
        cv2.putText(orig, shape, (cX - 40, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if debug:
            print(f"Detected {shape} - area: {area:.1f}, vertices: {vertices}, center: {cX, cY}")

    return orig, detections

def detect_circles_hough(image: np.ndarray,
                         dp: float = 1.2,
                         min_dist: int = 20,
                         param1: int = 100,
                         param2: int = 30,
                         min_radius: int = 0,
                         max_radius: int = 0,
                         debug: bool = False):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blurred,
                               cv2.HOUGH_GRADIENT,
                               dp=dp,
                               minDist=min_dist,
                               param1=param1,
                               param2=param2,
                               minRadius=min_radius,
                               maxRadius=max_radius)
    detections = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append({"center": (int(x), int(y)), "radius": int(r)})
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            if debug:
                print(f"Hough circle: center=({x},{y}), r={r}")
    else:
        if debug:
            print("No Hough circles detected")
    return output, detections

# If run as script, keep the CLI (shortened)
def _parse_args():
    parser = argparse.ArgumentParser(description="Shape detector utility (improved)")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--mode", choices=["contours", "hough", "regular"], default="regular")
    parser.add_argument("--n", type=int, default=18, help="Number of sides for regular polygon search")
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--side-cv", type=float, default=0.15)
    parser.add_argument("--radius-cv", type=float, default=0.12)
    parser.add_argument("--vertex-slack", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main():
    args = _parse_args()
    if not args.image:
        print("Please provide --image")
        return

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Unable to read image {args.image}")

    if args.mode == "regular":
        annotated, detections = detect_regular_polygons(image, n=args.n, min_area=args.min_area,
                                                        eps_factor=args.eps, max_side_cv=args.side_cv,
                                                        max_radius_cv=args.radius_cv, vertex_slack=args.vertex_slack,
                                                        debug=args.debug)
    elif args.mode == "contours":
        annotated, detections = detect_shapes_contours(image, min_area=args.min_area, debug=args.debug)
    elif args.mode == "hough":
        annotated, detections = detect_circles_hough(image, debug=args.debug)
    else:
        raise ValueError("Unknown mode")

    cv2.imshow("Annotated", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
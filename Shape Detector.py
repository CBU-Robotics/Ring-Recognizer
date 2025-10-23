#!/usr/bin/env python3
"""
shape_detector.py

Shape detection utilities with extra support for finding "more even" regular polygons
and composite shapes (a central square surrounded by a regular n-gon).

This file extends a simple contour-based detector with:
 - is_regular_polygon: checks side-length and radial-evenness tolerances
 - detect_regular_polygons: finds polygons with N sides that are "even"
 - detect_square_plus_regular_ngon: looks for a central square and a surrounding regular n-gon

Dependencies:
  - opencv-python
  - numpy

Examples:
  # Detect regular 12-gon (dodecagon) in an image:
  python shape_detector.py --image examples/test.jpg --mode regular --n 12 --debug

  # Detect composite: square surrounded by regular 12-gon
  python shape_detector.py --image examples/test.jpg --mode composite --n 12 --debug

Notes:
 - "More even" is implemented via two measures:
    * side length consistency (coefficient-of-variation of side lengths)
    * radial consistency from polygon centroid (coefficient-of-variation of radii)
 - Tolerances are configurable from the CLI.
"""

import argparse
import math
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np


def approximate_contour(cnt: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
    return approx


def contour_center(cnt: np.ndarray) -> Tuple[float, float]:
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        # fallback to average of points
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


def is_regular_polygon(cnt: np.ndarray,
                       n_sides: int,
                       eps_factor: float = 0.02,
                       max_side_cv: float = 0.15,
                       max_radius_cv: float = 0.12) -> Tuple[bool, Dict[str, float]]:
    """
    Test whether a contour corresponds to an approximately regular polygon with n_sides.
    Returns (is_regular, metrics) where metrics include side_cv, radius_cv, detected_vertices.
    """
    approx = approximate_contour(cnt, epsilon_factor=eps_factor)
    detected = len(approx)

    metrics = {"detected_vertices": float(detected), "side_cv": float("nan"), "radius_cv": float("nan")}

    if detected != n_sides:
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


def detect_regular_polygons(image: np.ndarray,
                            n: int = 12,
                            min_area: int = 500,
                            eps_factor: float = 0.02,
                            max_side_cv: float = 0.15,
                            max_radius_cv: float = 0.12,
                            debug: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect contours that are approximately regular n-sided polygons.
    Returns annotated image and list of detections:
      { "n": n, "contour": cnt, "center": (x,y), "area": area, "metrics": {...} }
    """
    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Dict[str, Any]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        ok, metrics = is_regular_polygon(
            cnt,
            n_sides=n,
            eps_factor=eps_factor,
            max_side_cv=max_side_cv,
            max_radius_cv=max_radius_cv
        )
        if not ok:
            if debug:
                # Print debug metrics for rejected candidates
                print(f"Reject poly: verts={metrics['detected_vertices']}, "
                      f"side_cv={metrics['side_cv']:.3f}, radius_cv={metrics['radius_cv']:.3f}, area={area:.1f}")
            continue

        c = contour_center(cnt)
        detections.append({
            "n": n,
            "contour": cnt,
            "center": (int(round(c[0])), int(round(c[1]))),
            "area": area,
            "metrics": metrics
        })

        # annotate
        approx = approximate_contour(cnt, epsilon_factor=eps_factor)
        cv2.drawContours(out, [approx], -1, (0, 200, 0), 2)
        cx, cy = int(round(c[0])), int(round(c[1]))
        cv2.putText(out, f"{n}-gon", (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if debug:
            print(f"Found regular {n}-gon at {cx,cy} area={area:.1f} metrics={metrics}")

    return out, detections


def is_square_like(cnt: np.ndarray,
                   eps_factor: float = 0.02,
                   max_side_cv: float = 0.18,
                   angle_tol_deg: float = 12.0) -> Tuple[bool, Dict[str, float]]:
    """
    Determine whether contour is an approximately square quadrilateral.
    Returns (is_square, metrics) where metrics include side_cv, mean_angle_deviation.
    """
    approx = approximate_contour(cnt, epsilon_factor=eps_factor)
    if len(approx) != 4:
        return False, {"detected_vertices": float(len(approx)), "side_cv": float("nan"), "angle_dev": float("nan")}

    verts = approx.reshape(-1, 2)
    side_lens = side_lengths_from_vertices(verts)
    side_cv = coef_of_variation(side_lens)

    # compute internal angles
    def angle_between(a, b, c):
        # angle at b formed by a-b-c
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        cosang = np.clip(cosang, -1.0, 1.0)
        return math.degrees(math.acos(cosang))

    angles = []
    pts = verts
    for i in range(4):
        a = pts[(i - 1) % 4]
        b = pts[i]
        c = pts[(i + 1) % 4]
        angles.append(angle_between(a, b, c))
    angles = np.array(angles)
    angle_dev = np.mean(np.abs(angles - 90.0))

    metrics = {"detected_vertices": float(len(approx)), "side_cv": float(side_cv), "angle_dev": float(angle_dev)}

    ok = (side_cv <= max_side_cv) and (angle_dev <= angle_tol_deg)
    return bool(ok), metrics


def detect_square_plus_regular_ngon(image: np.ndarray,
                                    n: int = 12,
                                    min_area_outer: int = 1000,
                                    min_area_inner: int = 200,
                                    max_centroid_offset: float = 0.12,
                                    eps_factor: float = 0.02,
                                    max_side_cv_outer: float = 0.15,
                                    max_radius_cv_outer: float = 0.12,
                                    debug: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Look for a composite shape: an inner square and a surrounding regular n-gon.
    - inner square should be inside the outer polygon.
    - centroids should be near each other (within max_centroid_offset * outer_radius)
    Returns annotated image and list of composite detections.
    """
    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    outers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_inner:
            continue

        # try square
        sq_ok, sq_metrics = is_square_like(cnt, eps_factor=eps_factor)
        if sq_ok and area >= min_area_inner:
            c = contour_center(cnt)
            squares.append({"contour": cnt, "area": area, "center": c, "metrics": sq_metrics})
            if debug:
                print(f"Candidate square: area={area:.1f}, center={c}, metrics={sq_metrics}")
            continue

        # try outer polygon
        ok_outer, outer_metrics = is_regular_polygon(
            cnt,
            n_sides=n,
            eps_factor=eps_factor,
            max_side_cv=max_side_cv_outer,
            max_radius_cv=max_radius_cv_outer
        )
        if ok_outer and area >= min_area_outer:
            c = contour_center(cnt)
            outers.append({"contour": cnt, "area": area, "center": c, "metrics": outer_metrics})
            if debug:
                print(f"Candidate outer {n}-gon: area={area:.1f}, center={c}, metrics={outer_metrics}")

    composites: List[Dict[str, Any]] = []

    # match squares to outer polygons by containment and centroid proximity
    for outer in outers:
        outer_cnt = outer["contour"]
        outer_center = outer["center"]
        # approximate outer radius (mean radius from centroid)
        approx_outer = approximate_contour(outer_cnt, epsilon_factor=eps_factor)
        radii = radii_from_centroid(approx_outer, outer_center)
        mean_outer_radius = float(np.mean(radii)) if radii.size > 0 else 1.0
        for sq in squares:
            sq_center = sq["center"]
            # ensure square center inside outer polygon
            inside = cv2.pointPolygonTest(outer_cnt, (sq_center[0], sq_center[1]), measureDist=False)
            centroid_dist = math.hypot(sq_center[0] - outer_center[0], sq_center[1] - outer_center[1])
            centroid_offset_norm = centroid_dist / (mean_outer_radius + 1e-9)

            if inside > 0 and centroid_offset_norm <= max_centroid_offset:
                # accept composite
                composites.append({
                    "outer": outer,
                    "square": sq,
                    "centroid_offset": centroid_offset_norm
                })
                # annotate
                cv2.drawContours(out, [approx_outer], -1, (0, 200, 0), 2)
                cv2.putText(out, f"{n}-gon", (int(outer_center[0]) - 30, int(outer_center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                approx_sq = approximate_contour(sq["contour"], epsilon_factor=eps_factor)
                cv2.drawContours(out, [approx_sq], -1, (255, 0, 0), 2)
                cv2.putText(out, "square", (int(sq_center[0]) - 30, int(sq_center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if debug:
                    print(f"Composite found: outer_center={outer_center}, square_center={sq_center}, "
                          f"centroid_offset={centroid_offset_norm:.3f}")

    return out, composites


def _parse_args():
    parser = argparse.ArgumentParser(description="Shape detector utility (extended for regular polygons)")
    parser.add_argument("--image", help="Path to input image (optional if using --camera)")
    parser.add_argument("--camera", action="store_true", help="Use camera for live detection")
    parser.add_argument("--mode", choices=["contours", "hough", "regular", "composite"], default="regular",
                        help="Detection mode: contours (basic), hough (circles), regular (nth-gon), composite (square + n-gon)")
    parser.add_argument("--n", type=int, default=12, help="Number of sides for regular polygon search")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum area for polygon detection (regular)")
    parser.add_argument("--min-area-outer", type=int, default=1000, help="Min area for outer polygon (composite)")
    parser.add_argument("--min-area-inner", type=int, default=200, help="Min area for inner square (composite)")
    parser.add_argument("--eps", type=float, default=0.02, help="approxPolyDP epsilon factor")
    parser.add_argument("--side-cv", type=float, default=0.15, help="max coefficient-of-variation for side lengths")
    parser.add_argument("--radius-cv", type=float, default=0.12, help="max coefficient-of-variation for radii")
    parser.add_argument("--centroid-offset", type=float, default=0.12,
                        help="max normalized centroid offset for composite detection")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--output", help="Optional path to save annotated image")
    # Hough-specific options (kept for backwards compatibility)
    parser.add_argument("--dp", type=float, default=1.2)
    parser.add_argument("--min-dist", type=int, default=20)
    parser.add_argument("--param1", type=int, default=100)
    parser.add_argument("--param2", type=int, default=30)
    parser.add_argument("--min-radius", type=int, default=0)
    parser.add_argument("--max-radius", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    
    if not args.image and not args.camera:
        print("Error: Must specify either --image or --camera")
        return
    
    if args.camera:
        # Camera mode - live detection
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera mode - Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame based on mode
            if args.mode == "contours":
                annotated, detections = detect_shapes_contours(frame, min_area=args.min_area, debug=args.debug)
            elif args.mode == "hough":
                annotated, detections = detect_circles_hough(frame,
                                                            dp=args.dp,
                                                            min_dist=args.min_dist,
                                                            param1=args.param1,
                                                            param2=args.param2,
                                                            min_radius=args.min_radius,
                                                            max_radius=args.max_radius,
                                                            debug=args.debug)
            elif args.mode == "regular":
                annotated, detections = detect_regular_polygons(
                    frame,
                    n=args.n,
                    min_area=args.min_area,
                    eps_factor=args.eps,
                    max_side_cv=args.side_cv,
                    max_radius_cv=args.radius_cv,
                    debug=args.debug
                )
            elif args.mode == "composite":
                annotated, detections = detect_square_plus_regular_ngon(
                    frame,
                    n=args.n,
                    min_area_outer=args.min_area_outer,
                    min_area_inner=args.min_area_inner,
                    max_centroid_offset=args.centroid_offset,
                    eps_factor=args.eps,
                    max_side_cv_outer=args.side_cv,
                    max_radius_cv_outer=args.radius_cv,
                    debug=args.debug
                )
            else:
                raise ValueError("Unknown mode")
            
            cv2.imshow("Shape Detection - Press 'q' to quit, 's' to save", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.output:
                cv2.imwrite(args.output, annotated)
                print(f"Frame saved to {args.output}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # Image mode - static image processing
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Unable to read image {args.image}")

        if args.mode == "contours":
            annotated, detections = detect_shapes_contours(image, min_area=args.min_area, debug=args.debug)
        elif args.mode == "hough":
            annotated, detections = detect_circles_hough(image,
                                                        dp=args.dp,
                                                        min_dist=args.min_dist,
                                                        param1=args.param1,
                                                        param2=args.param2,
                                                        min_radius=args.min_radius,
                                                        max_radius=args.max_radius,
                                                        debug=args.debug)
        elif args.mode == "regular":
            annotated, detections = detect_regular_polygons(
                image,
                n=args.n,
                min_area=args.min_area,
                eps_factor=args.eps,
                max_side_cv=args.side_cv,
                max_radius_cv=args.radius_cv,
                debug=args.debug
            )
        elif args.mode == "composite":
            annotated, detections = detect_square_plus_regular_ngon(
                image,
                n=args.n,
                min_area_outer=args.min_area_outer,
                min_area_inner=args.min_area_inner,
                max_centroid_offset=args.centroid_offset,
                eps_factor=args.eps,
                max_side_cv_outer=args.side_cv,
                max_radius_cv_outer=args.radius_cv,
                debug=args.debug
            )
        else:
            raise ValueError("Unknown mode")

        if args.output:
            cv2.imwrite(args.output, annotated)
            if args.debug:
                print(f"Annotated image written to {args.output}")
        else:
            cv2.imshow("Annotated", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Print concise detection summary
        if args.mode in ("regular", "contours"):
            for d in detections:
                print({k: v for k, v in d.items() if k != "contour"})
        else:
            for d in detections:
                # composite or hough: print higher-level info
                if args.mode == "composite":
                    outer = d["outer"]
                    square = d["square"]
                    print({
                        "outer_center": outer["center"],
                        "outer_area": outer["area"],
                        "outer_metrics": outer["metrics"],
                        "square_center": square["center"],
                        "square_area": square["area"],
                        "square_metrics": square["metrics"],
                        "centroid_offset": d["centroid_offset"]
                    })
                else:
                    print(d)


# The basic contour and Hough functions from the earlier implementation are reused here.
# To avoid duplication of code in this single-file example, we add minimal implementations
# (these are compatible with the previous assistant message).
def detect_shapes_contours(image: np.ndarray,
                           min_area: int = 100,
                           debug: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
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
                         debug: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
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


if __name__ == "__main__":
    main()
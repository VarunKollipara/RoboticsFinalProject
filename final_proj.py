import cv2
import numpy as np
import time
import math
import threading
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI
from dataclasses import dataclass
import json
import os
import logging

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

# ==========================================
# CONFIGURATION
# ==========================================
ROBOT_IP = '192.168.1.183'
GRIPPER_LENGTH = 0.067 * 1000

MARKER_STAGING_POSE = [] # TODO: Define
ERASER_STAGING_POSE = [] # TODO: Define

# AprilTag configurations for the whiteboard
TAG_SIZE = 0.08
TAG_CENTER_COORDINATES = [
    [0.38, 0.4],
    [0.38, -0.4],
    [0.0, 0.4],
    [0.0, -0.4]
]

# Set to True to use the values below. Set to False to use JSON/Teaching mode.
USE_HARDCODED_POSITIONS = True 
# Y val is how close to board the robot should be
HARDCODED_POSITIONS = {
    # --- ERASER WORKSPACE ---
    'eraser_top_left': [320.3, -128.1, 569.1, 83.2, 1.4, 5],
    'eraser_top_right': [118.4, -129.6, 580.3, 85.4, 2.9, 2.3],
    'eraser_bottom_right': [147.4, -130.2, 393.3, 83.6, 5.4, 3],
    'eraser_bottom_left': [307.7, -127, 394.7, 87.8, 0.9, 8.9],
    
    # --- MARKER WORKSPACE ---
    'marker_top_left': [331.1, -71.5, 561.9, 86.9, 1.5, 90.3], 
    'marker_top_right': [131.1, -84.5, 567.9, 81.6, 6.3, 87.2],
    'marker_bottom_right': [133.2, -85.1, 376, 92.9, 5.3, 90.8],
    'marker_bottom_left': [314.5, -73.7, 380.6, 93.7, 4.2, 90.8],
    
    # --- TOOL STAGING ---
    'eraser_staging': [326.7, 101.7, 34.5, 172.9, -6, 14.1],
    'marker_staging': [335.1, 246.9, 22.6, 179, -8.7, 1.8]

}

DRAW_HOVER_OFFSET_M = 0.03
DRAW_CONTACT_OFFSET_M = 0.0
DRAW_RETREAT_OFFSET_M = 0.03
DRAW_POINT_SPACING_M = 0.005
DRAW_TRAVEL_SPEED = 120.0
DRAW_STROKE_SPEED = 40.0
MIN_ZED_VALID_DEPTH_M = 0.05
MAX_ZED_VALID_DEPTH_M = 3.0

import logging

# ==========================================
# DEBUG LOGGING SETUP
# ==========================================
logging.basicConfig(
    filename='robot_debug.log', 
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    filemode='w' # 'w' overwrites the file every time you run the script. Use 'a' to append.
)

# Optional: Also print to terminal if you want, but the file is your safe backup
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


# ==========================================
# ERASER / WIPING CONFIGURATION
# ==========================================

# Temporary eraser wiping pose.
# This is estimated from current hardcoded board poses.
ERASER_WIPE_RPY = [-7.3, 81.8, -95.9]

# +Y means moving away from the board.
# Larger value = safer but may not touch the board.
# Smaller value = closer contact but higher collision risk.
ERASER_CONTACT_Y_OFFSET_MM = -30.0

# Safe hover distance before touching the board.
ERASER_SAFE_Y_OFFSET_MM = 15.0

# Avoid wiping all the way to the AprilTag boundary.
ERASER_INSET_X_MM = 100.0
ERASER_INSET_Z_MM = 100.0

# Wiping parameters.
ERASER_NUM_SWEEPS = 6
ERASER_APPROACH_SPEED = 45.0
ERASER_WIPE_SPEED = 35.0

# Fine-tuning translation offsets for the eraser (in millimeters)
ERASER_OFFSET_X_MM = 15  # Shift left/right (Negative X goes right)
ERASER_OFFSET_Z_MM = 0   # Shift up/down (Positive Z goes up)


# Fine-tuning translation offsets for the marker (in millimeters)
MARKER_OFFSET_X_MM = 0.0   # Shift left/right (Negative X)
MARKER_OFFSET_Z_MM = 0.0   # Shift up/down (Positive Z)

# Depth offsets for the marker
MARKER_CONTACT_Y_OFFSET_MM = -3.5  # Distance from board to draw
MARKER_SAFE_Y_OFFSET_MM = 15.0    # Distance to pull back when hovering

MARKER_DRAW_RPY = [88.7, -1.6, 87.4]


# ==========================================
# PERCEPTION
# ==========================================
class DrawingPerception:
    def __init__(self):
        # Minimum pixel area to ignore random noise/smudges on the board
        self.min_contour_area = 80 
        # Higher factor = more simplified shapes 
        self.epsilon_factor = 0.005 

    # drawing extraction uses centerline skeleton tracing
    def extract_drawing_paths(self, bgr_image, board_polygon=None):
        processed_image = bgr_image.copy()

        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            55,
            2
        )

        if board_polygon is not None:
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [board_polygon], 255)

            kernel = np.ones((3, 3), np.uint8)
            #offset for how much to erode the mask by board boundary
            mask = cv2.erode(mask, kernel, iterations=55)

            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cleaned = remove_small_components(thresh, 40)
        skeleton = skeletonize_binary(cleaned)

        # cv2.imshow("Debug: Skeleton", skeleton)

        drawing_paths = trace_skeleton_paths(skeleton, self.epsilon_factor)

        print(f"  [Debug] Paths before stitching: {len(drawing_paths)}")
        drawing_paths = merge_nearby_paths(drawing_paths, max_gap_px=5)
        print(f"  [Debug] Paths after stitching: {len(drawing_paths)}")
        
        drawing_paths = order_paths_for_drawing(drawing_paths)

        return drawing_paths, skeleton

    # makes closed loops render as closed circles
    def draw_debug_paths(self, original_image, paths):
        debug_img = original_image.copy()

        for path in paths:
            if len(path) < 2:
                continue

            pts = np.array(path, np.int32).reshape((-1, 1, 2))
            is_closed = len(path) >= 3 and path[0] == path[-1]

            cv2.polylines(
                debug_img,
                [pts],
                isClosed=is_closed,
                color=(0, 255, 0),
                thickness=2
            )

        return debug_img

@dataclass
class DrawWaypoint:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    pen_down: bool
    speed: float

def merge_nearby_paths(paths, max_gap_px=30):
    """
    Acts as a magnet to stitch shattered paths back together.
    If the end of one stroke is within max_gap_px of another, they merge.
    """
    if not paths:
        return []

    # Convert to lists so we can easily append/reverse
    active_paths = [list(p) for p in paths]
    merged = []

    while active_paths:
        base = active_paths.pop(0)
        changed = True

        while changed:
            changed = False
            best_idx = -1
            best_dist = max_gap_px
            mode = ''

            base_start = np.array(base[0])
            base_end = np.array(base[-1])

            for i, target in enumerate(active_paths):
                target_start = np.array(target[0])
                target_end = np.array(target[-1])

                # Check all 4 possible combinations of endpoints
                d_ss = np.linalg.norm(base_start - target_start)
                if d_ss < best_dist: best_dist = d_ss; best_idx = i; mode = 'ss'
                
                d_se = np.linalg.norm(base_start - target_end)
                if d_se < best_dist: best_dist = d_se; best_idx = i; mode = 'se'
                
                d_es = np.linalg.norm(base_end - target_start)
                if d_es < best_dist: best_dist = d_es; best_idx = i; mode = 'es'
                
                d_ee = np.linalg.norm(base_end - target_end)
                if d_ee < best_dist: best_dist = d_ee; best_idx = i; mode = 'ee'

            # If a nearby path was found, stitch them together
            if best_idx != -1:
                target = active_paths.pop(best_idx)
                if mode == 'es':   base = base + target              # End to Start
                elif mode == 'ee': base = base + target[::-1]        # End to End (reverse target)
                elif mode == 'se': base = target + base              # Start to End
                elif mode == 'ss': base = target[::-1] + base        # Start to Start (reverse target)
                changed = True

        merged.append(base)
        
    return merged

# ==========================================
# MATH & CALIBRATION
# ==========================================
def get_pnp_pairs(tags):
    world_points = np.empty([0, 3])
    image_points = np.empty([0, 2])

    for tag in tags:
        if tag.tag_id > 3:
            continue
        
        tag_center = TAG_CENTER_COORDINATES[tag.tag_id]

        wp = np.zeros(3); wp[0] = tag_center[0] - (TAG_SIZE / 2); wp[1] = tag_center[1] + (TAG_SIZE / 2)
        ip = tag.corners[0]
        world_points = np.vstack([world_points, wp]); image_points = np.vstack([image_points, ip])

        wp = np.zeros(3); wp[0] = tag_center[0] - (TAG_SIZE / 2); wp[1] = tag_center[1] - (TAG_SIZE / 2)
        ip = tag.corners[1]
        world_points = np.vstack([world_points, wp]); image_points = np.vstack([image_points, ip])

        wp = np.zeros(3); wp[0] = tag_center[0] + (TAG_SIZE / 2); wp[1] = tag_center[1] - (TAG_SIZE / 2)
        ip = tag.corners[2]
        world_points = np.vstack([world_points, wp]); image_points = np.vstack([image_points, ip])

        wp = np.zeros(3); wp[0] = tag_center[0] + (TAG_SIZE / 2); wp[1] = tag_center[1] + (TAG_SIZE / 2)
        ip = tag.corners[3]
        world_points = np.vstack([world_points, wp]); image_points = np.vstack([image_points, ip])

    return world_points, image_points

def get_transform_camera_robot(observation, camera_intrinsic):
    detector = Detector(families='tag36h11')
    if len(observation.shape) > 2:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    tags = detector.detect(observation, estimate_tag_pose=False)
    
    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        return None

    success, rotation_vec, translation = cv2.solvePnP(world_points, image_points, camera_intrinsic, None)
    if success is not True:
        return None
        
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation.flatten()
    return transform_mat
    
def get_board_polygon(observation):
    """
    Detects AprilTags 0, 1, 2, and 3 and returns the board polygon.

    Expected tag layout:
        0 = bottom-left
        1 = top-left
        2 = top-right
        3 = bottom-right

    Output order:
        top-left, top-right, bottom-right, bottom-left
    """
    detector = Detector(families='tag36h11')

    # ZED images are often BGRA, so handle both 3-channel and 4-channel images.
    if len(observation.shape) == 3:
        if observation.shape[2] == 4:
            gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    else:
        gray = observation.copy()

    # Make sure AprilTag detector gets clean uint8 grayscale.
    gray = np.ascontiguousarray(gray.astype(np.uint8))

    tags = detector.detect(gray, estimate_tag_pose=False)

    print(f"[AprilTag Debug] Detected {len(tags)} tag(s).")
    for tag in tags:
        print(f"  tag_id={tag.tag_id}, center={tag.center}")

    centers = {}
    for tag in tags:
        if tag.tag_id in [0, 1, 2, 3]:
            centers[tag.tag_id] = tag.center

    missing = [tag_id for tag_id in [0, 1, 2, 3] if tag_id not in centers]
    if missing:
        print(f"[AprilTag Debug] Missing board tag(s): {missing}")
        return None

    pts = np.array(
        [
            centers[1],  # top-left
            centers[2],  # top-right
            centers[3],  # bottom-right
            centers[0],  # bottom-left
        ],
        dtype=np.int32
    )

    return pts.reshape((-1, 1, 2))


# ==========================================
# OBJECT DETECTION
# ==========================================
"""object_detection — Detect marker and eraser on the table via HSV + depth."""

def pixel_to_3d(u, v, depth_m, K):
    """Back-project pixel (u,v) + depth to 3D point in camera frame."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return np.array([x, y, depth_m])

def median_depth(depth, u, v, radius=3):
    """Median depth in a small patch around (u,v), ignoring zeros/nans."""
    h, w = depth.shape[:2]
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    patch = depth[v0:v1, u0:u1].flatten()
    valid = patch[(patch > 0.1) & (patch < 3.0) & np.isfinite(patch)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0

MARKER_HSV_LO = np.array([0, 0, 160])
MARKER_HSV_HI = np.array([180, 70, 255])
MARKER_MIN_AREA = 500
MARKER_ASPECT_RANGE = (3.0, 15.0)

ERASER_HSV_LO = np.array([0, 0, 0])
ERASER_HSV_HI = np.array([180, 80, 80])
ERASER_MIN_AREA = 800
ERASER_ASPECT_RANGE = (1.0, 4.0)

DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 2.0

@dataclass
class GraspTarget:
    """Pose of a graspable object (marker or eraser)."""
    position_cam: np.ndarray
    position_robot: np.ndarray
    grasp_axis: np.ndarray
    confidence: float
    label: str
    centroid_px: tuple[int, int]

def detect_object_hsv(rgb, depth, K, hsv_lo, hsv_hi, min_area,
                      aspect_range, label, T_cam_to_robot=None):
    """
    Detect a colored object using HSV thresholding + depth.

    Returns GraspTarget or None. Picks the largest valid contour.
    """
    if rgb is None or depth is None:
        return None

    if len(rgb.shape) > 2 and rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area and aspect ratio
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        ar = max(w, h) / min(w, h)
        if aspect_range[0] <= ar <= aspect_range[1]:
            candidates.append((c, area, rect))

    if not candidates:
        # Fall back to largest contour without aspect filter
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None
        candidates = [(c, cv2.contourArea(c), cv2.minAreaRect(c))]

    # Pick largest
    contour, area, rect = max(candidates, key=lambda x: x[1])

    # Centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cu = int(M["m10"] / M["m00"])
    cv_ = int(M["m01"] / M["m00"])

    # Depth at centroid
    d = median_depth(depth, cu, cv_)
    if d < DEPTH_MIN_M or d > DEPTH_MAX_M:
        return None

    # 3D position in camera frame
    p_cam = pixel_to_3d(cu, cv_, d, K)

    # Long-axis direction from minAreaRect
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
    rad = np.deg2rad(angle)
    axis_cam = np.array([np.cos(rad), np.sin(rad), 0.0])

    # Transform to robot frame if calibration available
    if T_cam_to_robot is not None:
        p_hom = np.append(p_cam, 1.0)
        p_robot = (T_cam_to_robot @ p_hom)[:3]
        R = T_cam_to_robot[:3, :3]
        axis_robot = R @ axis_cam
        axis_robot /= np.linalg.norm(axis_robot) + 1e-9
    else:
        p_robot = p_cam.copy()
        axis_robot = axis_cam

    conf = float(np.clip(area / (min_area * 5), 0.0, 1.0))

    return GraspTarget(
        position_cam=p_cam,
        position_robot=p_robot,
        grasp_axis=axis_robot,
        confidence=conf,
        label=label,
        centroid_px=(cu, cv_),
    )

def detect_marker(rgb, depth, K, T_cam_to_robot=None):
    return detect_object_hsv(
        rgb, depth, K,
        MARKER_HSV_LO, MARKER_HSV_HI,
        MARKER_MIN_AREA, MARKER_ASPECT_RANGE,
        "marker", T_cam_to_robot
    )

def detect_eraser(rgb, depth, K, T_cam_to_robot=None):
    return detect_object_hsv(
        rgb, depth, K,
        ERASER_HSV_LO, ERASER_HSV_HI,
        ERASER_MIN_AREA, ERASER_ASPECT_RANGE,
        "eraser", T_cam_to_robot
    )

# Debug
def draw_detection(rgb, target):
    if len(rgb.shape) > 2 and rgb.shape[2] == 4:
        vis = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
    else:
        vis = rgb.copy()

    if target is None:
        return vis

    cu, cv_ = target.centroid_px
    color = (0, 255, 0) if target.label == "marker" else (0, 165, 255)
    cv2.circle(vis, (cu, cv_), 8, color, -1)
    text = f"{target.label} c={target.confidence:.2f}"
    cv2.putText(vis, text, (cu + 12, cv_ - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis

def detect_scene_objects(cv_image, point_cloud, camera_intrinsic, t_cam_robot):
    if len(cv_image.shape) > 2 and cv_image.shape[2] == 4:
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
    else:
        rgb = cv_image.copy()

    depth = point_cloud[:, :, 2]

    marker = detect_marker(rgb, depth, camera_intrinsic, t_cam_robot)
    eraser = detect_eraser(rgb, depth, camera_intrinsic, t_cam_robot)

    marker_debug = draw_detection(rgb, marker)
    eraser_debug = draw_detection(rgb, eraser)

    return marker, eraser, marker_debug, eraser_debug


# ==========================================
# TODO FUNCTIONS
# ==========================================

def grasp_eraser(arm, staging_pose):
    print("  [Action] Picking up the eraser...")
    tx, ty, tz, r, p, y = staging_pose
    
    # Open Lite 6 Gripper
    code = arm.open_lite6_gripper()
    if code != 0:
        print("[GRIPPER FAULT] Failed to open Lite 6 gripper.")
        return
    time.sleep(1.0) 
    
    # Hover 60mm above staging
    arm.set_position(tx, ty, tz + 60, r, p, y, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    
    
    # Drop down to grasp
    arm.set_position(tx, ty, tz-15, r, p, y, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    
    # Close Lite 6 Gripper
    arm.close_lite6_gripper()
    time.sleep(1.0) 
    
    # Lift up
    arm.set_position(tx, ty, tz + 60, r, p, y, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)


def move_and_verify(arm, x, y, z, roll, pitch, yaw, speed, wait=True):
    """Sends a move command and immediately halts the script if the robot rejects it."""
    code = arm.set_position(
        x=x, y=y, z=z, 
        roll=roll, pitch=pitch, yaw=yaw, 
        is_radian=False, wait=wait, speed=speed
    )
    
    if code != 0:
        error_msg = f"KINEMATIC ERROR {code}! Unreachable position: X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f} | RPY: {roll}, {pitch}, {yaw}"
        logging.error(error_msg)
        print(f"\n[CRASH] {error_msg}")
        
        # Stop the robot immediately and crash the Python script
        arm.set_state(4) 
        raise RuntimeError("Robot halted due to unreachable position.")


def clear_board(arm, board_bounds):
    print("  [Action] Sweeping the board...")
    tl, tr, br, bl = board_bounds

    # This is the gripper wiping pose.
    # It controls how the gripper faces the wall.
    roll, pitch, yaw = tl[3:]

    def dist3(a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def interp_point(a, b, fraction):
        return [
            a[0] + fraction * (b[0] - a[0]),
            a[1] + fraction * (b[1] - a[1]),
            a[2] + fraction * (b[2] - a[2])
        ]

    def board_point(u, v):
        """
        u = 0 means left side, u = 1 means right side.
        v = 0 means top side, v = 1 means bottom side.
        """
        top = interp_point(tl, tr, u)
        bottom = interp_point(bl, br, u)
        point = interp_point(top, bottom, v)

        # +Y is away from the board.
        # This offset prevents the gripper from hitting the wall directly.
        point[1] = point[1] + ERASER_CONTACT_Y_OFFSET_MM

        return point

    # Estimate board width and height in mm.
    board_width = 0.5 * (dist3(tl, tr) + dist3(bl, br))
    board_height = 0.5 * (dist3(tl, bl) + dist3(tr, br))

    # Convert inset distance from mm to normalized board coordinates.
    u_min = ERASER_INSET_X_MM / board_width
    u_max = 1.0 - u_min

    v_min = ERASER_INSET_Z_MM / board_height
    v_max = 1.0 - v_min

    # Clamp to avoid weird values if board size is estimated badly.
    u_min = max(0.0, min(0.45, u_min))
    u_max = max(0.55, min(1.0, u_max))

    v_min = max(0.0, min(0.45, v_min))
    v_max = max(0.55, min(1.0, v_max))

    num_sweeps = ERASER_NUM_SWEEPS
    last_pt = None

    for i in range(num_sweeps + 1):
        fraction = i / num_sweeps

        # Move from left side to right side by columns.
        u = u_min + fraction * (u_max - u_min)

        top_pt = board_point(u, v_min)
        bottom_pt = board_point(u, v_max)

        # Default starts from upper-left and moves downward first.
        # Then the next column moves upward.
        if i % 2 == 0:
            start_pt = top_pt
            end_pt = bottom_pt
        else:
            start_pt = bottom_pt
            end_pt = top_pt

        # First column:
        # Move to a safe point first, then slowly approach the board.
        if i == 0:
            safe_start = [
                start_pt[0],
                start_pt[1] + ERASER_SAFE_Y_OFFSET_MM,
                start_pt[2]
            ]

            logging.info(f"Sweep {i}: Pulling up from staging area...")
            _, current_pose = arm.get_position(is_radian=False)
            arm.set_position(
                current_pose[0], current_pose[1], current_pose[2] + 100.0, 
                current_pose[3], current_pose[4], current_pose[5],
                is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
            )
            
            logging.info(f"Sweep {i}: Moving to safe_start XYZ (keeping orientation)...")
            arm.set_position(
                *safe_start,
                current_pose[3], current_pose[4], current_pose[5],
                is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
            )
            
            logging.info(f"Sweep {i}: Twisting wrist to wipe orientation {roll, pitch, yaw}...")
            arm.set_position(
                *safe_start,
                roll, pitch, yaw,
                is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
            )

            logging.info(f"Sweep {i}: Plunging to board surface...")
            arm.set_position(
                *start_pt,
                roll, pitch, yaw,
                is_radian=False,
                wait=True,
                speed=ERASER_APPROACH_SPEED
            )

        else:
            logging.info(f"Sweep {i}: Moving to column start...")
            arm.set_position(
                *start_pt,
                roll, pitch, yaw,
                is_radian=False,
                wait=True,
                speed=ERASER_WIPE_SPEED
            )

        logging.info(f"Sweep {i}: Wiping down column...")
        arm.set_position(
            *end_pt,
            roll, pitch, yaw,
            is_radian=False,
            wait=True,
            speed=ERASER_WIPE_SPEED
        )
        
        logging.info(f"Sweep {i}: Column complete.")
        last_pt = end_pt

    # Retreat away from the board safely.
    if last_pt is not None:
        safe_end = [
            last_pt[0],
            last_pt[1] + ERASER_SAFE_Y_OFFSET_MM,
            last_pt[2]
        ]

        arm.set_position(
            *safe_end,
            roll, pitch, yaw,
            is_radian=False,
            wait=True,
            speed=DRAW_TRAVEL_SPEED
        )


def grasp_marker(arm, eraser_pose, marker_pose):
    print("  [Action] Swapping eraser for marker...")
    ex, ey, ez, er, ep, eyaw = eraser_pose
    mx, my, mz, mr, mp, myaw = marker_pose
    
    # --- Drop off Eraser ---
    arm.set_position(ex, ey, ez + 60, er, ep, eyaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    arm.set_position(ex, ey, ez, er, ep, eyaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    arm.open_lite6_gripper()
    time.sleep(1.0)
    arm.set_position(ex, ey, ez + 60, er, ep, eyaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    
    # --- Pick up Marker ---
    arm.set_position(mx, my, mz + 60, mr, mp, myaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    arm.set_position(mx, my, mz, mr, mp, myaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    arm.close_lite6_gripper()
    time.sleep(1.0)
    arm.set_position(mx, my, mz + 60, mr, mp, myaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)

def execute_drawing(arm, pixel_paths, board_poly, board_bounds):
    print("  [Action] Continuous drag drawing paths on the whiteboard...")
    if not pixel_paths or board_poly is None:
        return

    # 1. Setup Perspective Transform (Homography) from Pixel Space to a 0.0-1.0 Square
    src_pts = board_poly.reshape((4, 2)).astype(np.float32)
    dst_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32) # BL, TL, TR, BR layout match
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    tl, tr, br, bl = board_bounds
    roll, pitch, yaw = tl[3:] # Use the dedicated marker orientation

    def map_to_3d(u, v):
        """Transforms a 2D camera pixel directly into the physical 3D plane bounds"""
        vec = np.array([u, v, 1.0])
        mapped = M @ vec
        nx, ny = mapped[0] / mapped[2], mapped[1] / mapped[2]

        SCALE_FACTOR = 1.0

        nx = ((nx - 0.5) * SCALE_FACTOR) + 0.5
        ny = ((ny - 0.5) * SCALE_FACTOR) + 0.5

        nx, ny = max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))
        
        def interp(val_idx):
            top_edge = tl[val_idx] + nx * (tr[val_idx] - tl[val_idx])
            bottom_edge = bl[val_idx] + nx * (br[val_idx] - bl[val_idx])
            return top_edge + ny * (bottom_edge - top_edge)
        
        # Calculate base coordinates
        base_x = interp(0)
        base_y = interp(1)
        base_z = interp(2)
        
        # Apply the MARKER fine-tuning translations
        final_x = base_x + MARKER_OFFSET_X_MM
        final_y = base_y # Y is handled by the Plunge/Hover offsets
        final_z = base_z + MARKER_OFFSET_Z_MM
        
        return final_x, final_y, final_z

    # 2. Execute Strokes
    # We use a smaller distance filter here (5mm) compared to the eraser (10mm) 
    # to preserve the sharp details and curves of the drawing while still stopping the stutter.
    MIN_DRAW_DIST_MM = 5.0 
    is_first_stroke = True

    for path in pixel_paths:
        if len(path) < 2: 
            continue
            
        # Get start point
        sx, sy, sz = map_to_3d(path[0][0], path[0][1])
        
        if is_first_stroke:
            print("    -> Transiting from staging to board...")
            # Grab current pose so we know the staging wrist angle
            _, current_pose = arm.get_position(is_radian=False)
            c_roll, c_pitch, c_yaw = current_pose[3], current_pose[4], current_pose[5]
            
            # 1. Fly to a point FAR away from the board to force the elbow to bend
            DEEP_RETREAT_Y = MARKER_SAFE_Y_OFFSET_MM + 150.0 
            
            move_and_verify(
                arm, sx, sy + DEEP_RETREAT_Y, sz, 
                c_roll, c_pitch, c_yaw, DRAW_TRAVEL_SPEED
            )
            
            # 2. Twist the wrist 90 degrees while safely hovering in the bent-elbow zone
            print("    -> Twisting wrist in safe zone...")
            move_and_verify(
                arm, sx, sy + DEEP_RETREAT_Y, sz, 
                roll, pitch, yaw, DRAW_TRAVEL_SPEED
            )
            
            # 3. Push forward into the normal 30mm hover position
            move_and_verify(
                arm, sx, sy + MARKER_SAFE_Y_OFFSET_MM, sz, 
                roll, pitch, yaw, DRAW_TRAVEL_SPEED
            )
            is_first_stroke = False
        else:
            # Normal hover between strokes (wrist is already in the correct orientation)
            arm.set_position(
                sx, sy + MARKER_SAFE_Y_OFFSET_MM, sz, 
                roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
            )
        
        # Plunge (Pen Down to make contact)
        arm.set_position(
            sx, sy + MARKER_CONTACT_Y_OFFSET_MM, sz, 
            roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_STROKE_SPEED
        )
        
        # Keep track of where the marker currently is
        last_x, last_y, last_z = sx, sy, sz
        
        # Trace line, but skip points that are too close together
        for pt in path[1:]:
            tx, ty, tz = map_to_3d(pt[0], pt[1])
            
            # Calculate 3D physical distance from the last commanded point
            dist = math.sqrt((tx - last_x)**2 + (ty - last_y)**2 + (tz - last_z)**2)
            
            if dist >= MIN_DRAW_DIST_MM:
                arm.set_position(
                    tx, ty + MARKER_CONTACT_Y_OFFSET_MM, tz, 
                    roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_STROKE_SPEED
                )
                last_x, last_y, last_z = tx, ty, tz
        
        # Guarantee we hit the very last point of the stroke
        final_x, final_y, final_z = map_to_3d(path[-1][0], path[-1][1])
        arm.set_position(
            final_x, final_y + MARKER_CONTACT_Y_OFFSET_MM, final_z, 
            roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_STROKE_SPEED
        )
        
        # Retreat (Pen Up safely)
        arm.set_position(
            final_x, final_y + MARKER_SAFE_Y_OFFSET_MM, final_z, 
            roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
        )


def smart_erase_lines(arm, pixel_paths, board_poly, board_bounds):
    print("  [Action] Multi-pass drag erasing along ink lines...")
    if not pixel_paths or board_poly is None:
        return

    # 1. Setup Perspective Transform (Homography) from Pixel Space to 3D Space
    src_pts = board_poly.reshape((4, 2)).astype(np.float32)
    dst_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32) 
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    tl, tr, br, bl = board_bounds
    roll, pitch, yaw = tl[3:]

    def map_to_3d(u, v, extra_x=0.0, extra_z=0.0):
        """Transforms a 2D camera pixel directly into physical 3D plane bounds with dynamic offsets"""
        vec = np.array([u, v, 1.0])
        mapped = M @ vec
        nx, ny = mapped[0] / mapped[2], mapped[1] / mapped[2]

        SCALE_FACTOR = 1.0

        nx = ((nx - 0.5) * SCALE_FACTOR) + 0.5
        ny = ((ny - 0.5) * SCALE_FACTOR) + 0.5
        
        nx, ny = max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))
        
        def interp(val_idx):
            top_edge = tl[val_idx] + nx * (tr[val_idx] - tl[val_idx])
            bottom_edge = bl[val_idx] + nx * (br[val_idx] - bl[val_idx])
            return top_edge + ny * (bottom_edge - top_edge)
        
        # Calculate base coordinates
        base_x = interp(0)
        base_y = interp(1)
        base_z = interp(2)
        
        # Apply the fine-tuning translations PLUS the current pass offset
        final_x = base_x + ERASER_OFFSET_X_MM + extra_x
        final_y = base_y # Y is already handled by ERASER_CONTACT_Y_OFFSET_MM
        final_z = base_z + ERASER_OFFSET_Z_MM + extra_z
        
        return final_x, final_y, final_z

    # --- THE FIX: SCRUB PASSES ---
    # We trace the drawing multiple times, shifted slightly in different 
    # directions to create a "thick" eraser brush effect.
    SCRUB_PASSES = [
        (0.0, 0.0),       # Pass 1: Dead center
        (15.0, 15.0),     # Pass 2: Shifted Up-Right
        (-15.0, -15.0),   # Pass 3: Shifted Down-Left
        
        (15.0, -15.0),  # Pass 4: Shifted Down-Right
        (-15.0, 15.0),   # Pass 5: Shifted Up-Left
        (-15.0, 0.0),     # Pass 6: Shifted Left
        (0.0, 15.0),     # Pass 7: Shifted Right
        (15.0, 0.0),     # Pass 8: Shifted Up
        (0.0, -15.0)     # Pass 9: Shifted Down

        #go again
        # (0.0, 0.0),       # Pass 1: Dead center
        # (15.0, 15.0),     # Pass 2: Shifted Up-Right
        # (-15.0, -15.0),   # Pass 3: Shifted Down-Left
        
        # # Uncomment below if you find it STILL misses edges!
        # (15.0, -15.0),  # Pass 4: Shifted Down-Right
        # (-15.0, 15.0)   # Pass 5: Shifted Up-Left

    ]

    MIN_WIPE_DIST_MM = 10.0

    for pass_idx, (pass_x_offset, pass_z_offset) in enumerate(SCRUB_PASSES):
        print(f"    -> Erase Pass {pass_idx + 1}/{len(SCRUB_PASSES)} (Offset X:{pass_x_offset}, Z:{pass_z_offset})...")

        # Grab the very first point of the very first path to start
        first_path = pixel_paths[0]
        sx, sy, sz = map_to_3d(first_path[0][0], first_path[0][1], pass_x_offset, pass_z_offset)
        
        # --- 1. INITIAL APPROACH ---
        # Hover safely over the start point
        arm.set_position(
            sx, sy + ERASER_SAFE_Y_OFFSET_MM, sz, 
            roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
        )
        
        # Plunge ONCE to make contact with the board
        arm.set_position(
            sx, sy + ERASER_CONTACT_Y_OFFSET_MM, sz, 
            roll, pitch, yaw, is_radian=False, wait=True, speed=ERASER_APPROACH_SPEED
        )

        # --- 2. CONTINUOUS DRAG ---
        for path in pixel_paths:
            if len(path) < 1: 
                continue
                
            # Drag the eraser to the START of the next line (Maintains contact!)
            start_x, start_y, start_z = map_to_3d(path[0][0], path[0][1], pass_x_offset, pass_z_offset)
            arm.set_position(
                start_x, start_y + ERASER_CONTACT_Y_OFFSET_MM, start_z, 
                roll, pitch, yaw, is_radian=False, wait=True, speed=ERASER_WIPE_SPEED
            )
            
            # Keep track of where the robot currently is
            last_x, last_y, last_z = start_x, start_y, start_z
            
            # Trace the ink line, but skip points that are too close together
            for pt in path[1:]:
                tx, ty, tz = map_to_3d(pt[0], pt[1], pass_x_offset, pass_z_offset)
                
                # Calculate 3D physical distance from the last commanded point
                dist = math.sqrt((tx - last_x)**2 + (ty - last_y)**2 + (tz - last_z)**2)
                
                if dist >= MIN_WIPE_DIST_MM:
                    arm.set_position(
                        tx, ty + ERASER_CONTACT_Y_OFFSET_MM, tz, 
                        roll, pitch, yaw, is_radian=False, wait=True, speed=ERASER_WIPE_SPEED
                    )
                    # Update our tracker to the new position
                    last_x, last_y, last_z = tx, ty, tz
                    
            # Always make sure we hit the very last point of the stroke!
            final_x, final_y, final_z = map_to_3d(path[-1][0], path[-1][1], pass_x_offset, pass_z_offset)
            arm.set_position(
                final_x, final_y + ERASER_CONTACT_Y_OFFSET_MM, final_z, 
                roll, pitch, yaw, is_radian=False, wait=True, speed=ERASER_WIPE_SPEED
            )
                
        # --- 3. RETREAT BEFORE NEXT PASS ---
        # Pull the eraser away safely before looping back to start the next offset pass
        last_path = pixel_paths[-1]
        lx, ly, lz = map_to_3d(last_path[-1][0], last_path[-1][1], pass_x_offset, pass_z_offset)
        
        arm.set_position(
            lx, ly + ERASER_SAFE_Y_OFFSET_MM, lz, 
            roll, pitch, yaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED
        )
        
# Converts cleaned binary drawing blobs into a 1-pixel centerline skeleton.
def skeletonize_binary(binary_img):
    skeleton = np.zeros(binary_img.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = binary_img.copy()

    while True:
        eroded = cv2.erode(temp_img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img = eroded.copy()

        if cv2.countNonZero(temp_img) == 0:
            break

    return skeleton


# Removes tiny connected components so noise is not traced as strokes.
def remove_small_components(binary_img, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    cleaned = np.zeros_like(binary_img)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


# Traces a 1-pixel skeleton graph into ordered drawing paths.
def trace_skeleton_paths(skeleton, epsilon_factor=0.003):
    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape

    def neighbors(x, y):
        pts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                    pts.append((nx, ny))
        return pts

    pixels = np.argwhere(skel > 0)
    if len(pixels) == 0:
        return []

    degree = {}
    for y, x in pixels:
        degree[(x, y)] = len(neighbors(x, y))

    visited_edges = set()
    paths = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def walk_path(start, nxt):
        path = [list(start)]
        prev = start
        curr = nxt

        while True:
            path.append([curr[0], curr[1]])
            visited_edges.add(edge_key(prev, curr))

            nbrs = neighbors(curr[0], curr[1])
            nbrs = [p for p in nbrs if p != prev]

            if len(nbrs) == 0:
                break

            if degree[curr] != 2:
                break

            next_candidates = [p for p in nbrs if edge_key(curr, p) not in visited_edges]
            if not next_candidates:
                break

            prev, curr = curr, next_candidates[0]

        return path

    for p, deg in degree.items():
        if deg != 2:
            for nbr in neighbors(p[0], p[1]):
                if edge_key(p, nbr) not in visited_edges:
                    path = walk_path(p, nbr)
                    if len(path) >= 2:
                        paths.append(path)

    for p in degree.keys():
        for nbr in neighbors(p[0], p[1]):
            if edge_key(p, nbr) not in visited_edges:
                path = walk_path(p, nbr)
                if len(path) >= 3:
                    if path[0] != path[-1]:
                        path.append(path[0])
                    paths.append(path)

    simplified = []
    for path in paths:
        cnt = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
        is_closed = len(path) >= 3 and path[0] == path[-1]
        eps = epsilon_factor * cv2.arcLength(cnt, is_closed)
        approx = cv2.approxPolyDP(cnt, eps, is_closed)
        simp = [pt[0].tolist() for pt in approx]

        if is_closed and len(simp) >= 3 and simp[0] != simp[-1]:
            simp.append(simp[0])

        if len(simp) >= 2:
            simplified.append(simp)

    return simplified


# new 423 Reorders strokes to reduce pen-up travel distance between paths.
def order_paths_for_drawing(paths):
    if not paths:
        return []

    remaining = [list(path) for path in paths]
    ordered = [remaining.pop(0)]

    while remaining:
        current_end = np.array(ordered[-1][-1], dtype=float)

        best_idx = 0
        best_dist = float("inf")
        best_reverse = False

        for i, path in enumerate(remaining):
            start = np.array(path[0], dtype=float)
            end = np.array(path[-1], dtype=float)
            closed = len(path) >= 3 and path[0] == path[-1]

            d_start = np.linalg.norm(current_end - start)
            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_reverse = False

            if not closed:
                d_end = np.linalg.norm(current_end - end)
                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = i
                    best_reverse = True

        chosen = remaining.pop(best_idx)
        if best_reverse:
            chosen = list(reversed(chosen))

        ordered.append(chosen)

    return ordered






# new 423 Completes 2D pixel stroke conversion into 3D world paths using depth or board-plane fallback.
def transform_paths_to_world(pixel_paths, t_cam_robot, zed_point_cloud, camera_intrinsic=None):
    world_paths = []

    plane_x = None
    if zed_point_cloud is not None:
        valid_x = []

        for path in pixel_paths:
            for u, v in path:
                uu = int(round(u))
                vv = int(round(v))

                if vv < 0 or uu < 0 or vv >= zed_point_cloud.shape[0] or uu >= zed_point_cloud.shape[1]:
                    continue

                xyz = zed_point_cloud[vv, uu]
                if xyz is None:
                    continue

                xyz = np.asarray(xyz, dtype=float).flatten()

                if xyz.shape[0] < 3:
                    continue
                if not np.isfinite(xyz[:3]).all():
                    continue

                depth = np.linalg.norm(xyz[:3])

                if MIN_ZED_VALID_DEPTH_M <= depth <= MAX_ZED_VALID_DEPTH_M:
                    valid_x.append(float(xyz[0]))

        if len(valid_x) > 20:
            plane_x = float(np.median(valid_x))

    for path in pixel_paths:
        world_path = []

        for u, v in path:
            uu = int(round(u))
            vv = int(round(v))
            point_world = None

            if zed_point_cloud is not None:
                if 0 <= vv < zed_point_cloud.shape[0] and 0 <= uu < zed_point_cloud.shape[1]:
                    xyz = zed_point_cloud[vv, uu]

                    if xyz is not None:
                        xyz = np.asarray(xyz, dtype=float).flatten()

                        if xyz.shape[0] >= 3 and np.isfinite(xyz[:3]).all():
                            depth = np.linalg.norm(xyz[:3])

                            if MIN_ZED_VALID_DEPTH_M <= depth <= MAX_ZED_VALID_DEPTH_M:
                                point_world = xyz[:3].tolist()

            if point_world is None and camera_intrinsic is not None and plane_x is not None:
                fx = camera_intrinsic[0, 0]
                fy = camera_intrinsic[1, 1]
                cx = camera_intrinsic[0, 2]
                cy = camera_intrinsic[1, 2]

                ray_cam = np.array([
                    (u - cx) / fx,
                    (v - cy) / fy,
                    1.0
                ], dtype=float)

                ray_cam /= np.linalg.norm(ray_cam)

                R = t_cam_robot[:3, :3]
                t = t_cam_robot[:3, 3]

                ray_world = R @ ray_cam

                if abs(ray_world[0]) > 1e-8:
                    scale = (plane_x - t[0]) / ray_world[0]

                    if scale > 0:
                        p = t + scale * ray_world
                        point_world = p.tolist()

            if point_world is not None:
                world_path.append(point_world)

        if len(world_path) >= 2:
            world_paths.append(world_path)

    return world_paths


def teach_robot_positions(arm):
    """
    Puts the arm into manual mode so the user can physically guide it to
    the 4 board corners and the tool staging locations.
    """
    print("\n" + "="*40)
    print("           TEACHING MODE ACTIVATED")
    print("="*40)
    print("The robot will now go into manual mode (freedrive).")
    print("Physically move the arm to the requested positions.\n")

    # Enable manual teaching mode
    arm.set_mode(2)
    arm.set_state(0)

    taught_data = {}

    def get_pose(prompt_text):
        input(f"[Action Required] Move robot to {prompt_text} and press ENTER...")
        code, pose = arm.get_position(is_radian=True)
        if code == 0:
            print(f"Recorded pose: {pose}\n")
            return pose
        else:
            print(f"Failed to get pose, error code: {code}\n")
            return None

    # 1. Teach Board Bounds
    print("--- TEACHING BOARD BOUNDS ---")
    taught_data['board_top_left'] = get_pose("Board TOP-LEFT corner")
    taught_data['board_top_right'] = get_pose("Board TOP-RIGHT corner")
    taught_data['board_bottom_right'] = get_pose("Board BOTTOM-RIGHT corner")
    taught_data['board_bottom_left'] = get_pose("Board BOTTOM-LEFT corner")

    # 2. Teach Tool Staging
    print("--- TEACHING TOOL LOCATIONS ---")
    taught_data['eraser_staging'] = get_pose("ERASER pickup location")
    taught_data['marker_staging'] = get_pose("MARKER pickup location")

    # Disable manual mode, return to position control
    print("Teaching complete. Locking robot position...")
    arm.set_mode(0)
    arm.set_state(0)

    return taught_data

CONFIG_FILE = 'robot_positions.json'

def save_positions(taught_data):
    """Saves the taught dictionary to a local JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(taught_data, f, indent=4)
    print(f"  💾 Positions saved locally to {CONFIG_FILE}")

def load_positions():
    """Loads the taught dictionary from the local JSON file if it exists."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None


def return_marker(arm, marker_pose):
    print("  [Action] Returning marker to staging area...")
    mx, my, mz, mr, mp, myaw = marker_pose
    
    # 1. Hover safely above the marker staging spot
    arm.set_position(mx, my, mz + 60, mr, mp, myaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    
    # 2. Lower down to the table/holder
    arm.set_position(mx, my, mz, mr, mp, myaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)
    
    # 3. Open the gripper to release the marker
    arm.open_lite6_gripper()
    time.sleep(1.0)
    
    # 4. Lift the arm safely back up
    arm.set_position(mx, my, mz + 60, mr, mp, myaw, is_radian=False, wait=True, speed=DRAW_TRAVEL_SPEED)



# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
def main():
    hardware = {}
    
    def boot_robot():
        print("Connecting to XArm (Lite 6)...")
        arm = XArmAPI(ROBOT_IP)
        arm.connect()
        
        # 1. Clear any lingering hardware errors
        arm.clean_error()
        arm.clean_gripper_error()
        time.sleep(0.5)
        
        # 2. Enable main arm motion and set offsets
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
        
        # 3. Set Mode and State
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        
        hardware['arm'] = arm
        print(" Lite 6 Ready!")

    def boot_camera():
        print("Booting ZED Camera...")
        try:
            zed = ZedCamera()
            hardware['zed'] = zed
            hardware['camera_intrinsic'] = zed.camera_intrinsic
            hardware['perception'] = DrawingPerception()
            print("ZED Camera Ready!")
        except Exception as e:
            print("ZED CAMERA FAILED:", e)

    #t_robot = threading.Thread(target=boot_robot)
    #t_cam = threading.Thread(target=boot_camera)
    #t_robot.start()
    #t_cam.start()
    #t_robot.join()
    #t_cam.join()

    boot_robot()
    boot_camera()

    arm = hardware['arm']
    if 'zed' not in hardware:
        raise RuntimeError("ZED camera failed to initialize. Check connection / logs.")

    zed = hardware['zed']
    camera_intrinsic = hardware['camera_intrinsic']
    perception = hardware['perception']

    try:
        # ---------------------------------------------------------
        # Optional Teaching Phase
        # ---------------------------------------------------------
        global MARKER_STAGING_POSE, ERASER_STAGING_POSE
        board_bounds = []
        taught_poses = None

        if USE_HARDCODED_POSITIONS:
            print("Using hardcoded positions from the script.")
            taught_poses = HARDCODED_POSITIONS
        else:
            do_teach = input("Do you want to manually teach the robot the bounds and tool locations? (y/n): ")
            
            if do_teach.lower() == 'y':
                # Teach and save to file
                taught_poses = teach_robot_positions(arm)
                save_positions(taught_poses)
            else:
                # Try to load from file
                taught_poses = load_positions()
                if taught_poses is None:
                    print("No saved positions found! You MUST teach the robot first or enable hardcoded positions.")
                    return # Abort the script so it doesn't crash later
                else:
                    print("Loaded saved robot positions from previous session.")

        # Apply the chosen poses to our variables
        MARKER_STAGING_POSE = taught_poses['marker_staging']
        ERASER_STAGING_POSE = taught_poses['eraser_staging']
        
        eraser_bounds = [
            taught_poses['eraser_top_left'],
            taught_poses['eraser_top_right'],
            taught_poses['eraser_bottom_right'],
            taught_poses['eraser_bottom_left']
        ]
        
        marker_bounds = [
            taught_poses['marker_top_left'],
            taught_poses['marker_top_right'],
            taught_poses['marker_bottom_right'],
            taught_poses['marker_bottom_left']
        ]
        print("Session positions successfully applied!\n")

        # ---------------------------------------------------------
        # 1. Move to Observation Pose & Capture Image
        # ---------------------------------------------------------
        print("Moving to Observation Pose...")
        arm.set_position(71, -1.1, 394.1, 175.9, -38, 3.6, wait=True)
        arm.clean_error()
        time.sleep(1.0)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)


        arm.clean_error()
        arm.set_position(71, -1.1, 394.1, 175.9, -38, 3.6, wait=True)
        arm.clean_error()



        time.sleep(1.0)
        print("Taking Master Snapshot...")
        cv_image = zed.image.copy()
        point_cloud = zed.point_cloud.copy()

        #Calibrate / Align with Board
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("ERROR: Could not find AprilTags to calibrate board location.")
            return

        

        # ---------------------------------------------------------
        # 2. Extract Drawing
        # ---------------------------------------------------------
        print("Isolating whiteboard and extracting drawing...")
        board_poly = get_board_polygon(cv_image)
        
        if board_poly is None:
            print("ERROR: Could not see all 4 AprilTags. Cannot bound the drawing area.")
            return

        pixel_paths, debug_thresh = perception.extract_drawing_paths(cv_image, board_poly)
        
        if not pixel_paths:
            print("No drawing detected on the board.")
            return

            
        print(f"Detected {len(pixel_paths)} individual strokes.")
        arm.move_gohome(wait=True, speed=500.0)

        # Show Debug Window to verify perception is working
        debug_img = perception.draw_debug_paths(cv_image, pixel_paths)
        
        # Draw the blue bounding box so you can see the isolated area
        cv2.polylines(debug_img, [board_poly], isClosed=True, color=(255, 0, 0), thickness=2)
        
        cv2.imshow("debug_paths.png", debug_img)
        # print("Debug image saved to 'debug_paths.png' in your current folder.")
        
        # # --- THE FIX: Terminal prompt instead of cv2.waitKey ---
        # print("\nPlease open 'debug_paths.png' to verify the detected strokes.")
        # user_input = input("Press 'ENTER' to continue to execution, or type 'q' to quit: ")

        
        
        print("Press 'k' on the image window to continue, or 'q' to quit...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('k'):
                print("Continuing to execution...")
                break
            elif key == ord('q'):
                print("Aborting.")
                return
                
        cv2.destroyAllWindows()
        
        # ---------------------------------------------------------
        # 3. Plan Path 
        # ---------------------------------------------------------
        world_paths = transform_paths_to_world(pixel_paths, t_cam_robot, point_cloud)

        # ---------------------------------------------------------
        # 4. Act (Now utilizing taught variables)
        # ---------------------------------------------------------


        grasp_eraser(arm, ERASER_STAGING_POSE)
        
        # Pass in eraser_bounds
        smart_erase_lines(arm, pixel_paths, board_poly, eraser_bounds)
        
        grasp_marker(arm, ERASER_STAGING_POSE, MARKER_STAGING_POSE)
        
        # Pass in marker_bounds
        execute_drawing(arm, pixel_paths, board_poly, marker_bounds)

        return_marker(arm, MARKER_STAGING_POSE)

        print("Robot Artist task complete.")

        arm.move_gohome(wait=True)
        arm.stop_lite6_gripper()

    finally:
        arm.set_state(4) 
        arm.disconnect()
        zed.close()

if __name__ == "__main__":
    main()


"""
Calibración monocular (cam0, cam1, cam2) + estéreo (cam0-cam1, cam0-cam2).
Calcula también la relación cam1-cam2 a través de cam0.

Requiere imágenes en:
  Calibration/cam0_imgs/   — imágenes de sesión 1 Y sesión 2 combinadas
  Calibration/cam1_imgs/   — imágenes de sesión 1
  Calibration/cam2_imgs/   — imágenes de sesión 2

Produce:
  Calibration/cam0_intrinsics.json
  Calibration/cam1_intrinsics.json
  Calibration/cam2_intrinsics.json
  Calibration/stereo_cam0_cam1.json
  Calibration/stereo_cam0_cam2.json
  Calibration/stereo_cam1_cam2.json

Uso: python calib_stereo.py
"""

import cv2 as cv
import numpy as np
import glob
import json
import os
import sys
import config

CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Calibration")

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_object_image_points(files, adict, board):
    """Get object points and image points for stereo calibration."""
    obj_points = []
    img_points = []
    valid      = []
    for fn in files:
        gray = cv.cvtColor(cv.imread(fn), cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, adict)
        if ids is None:
            continue
        ret, ch_c, ch_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not ret or ch_ids is None or len(ch_ids) < config.MIN_CORNERS:
            continue
        obj_pts, img_pts = board.matchImagePoints(ch_c, ch_ids)
        if obj_pts is not None and len(obj_pts) >= config.MIN_CORNERS:
            obj_points.append(obj_pts)
            img_points.append(img_pts)
            valid.append(fn)
    return obj_points, img_points, valid

def stereo_calibrate(K_a, dist_a, files_a, K_b, dist_b, files_b, adict, board, img_size, label):
    """Stereo calibration between two cameras using synchronized image pairs (same filename)."""
    print(f"\n── Stereo {label} ──")

    names_a = {os.path.basename(f): f for f in files_a}
    names_b = {os.path.basename(f): f for f in files_b}
    common  = sorted(set(names_a.keys()) & set(names_b.keys()))
    print(f"Synchronized pairs: {len(common)}")

    if len(common) < 5:
        print(f"ERROR: Need at least 5 pairs for stereo calibration, found {len(common)}")
        sys.exit(1)

    obj_a, img_a, _ = get_object_image_points([names_a[n] for n in common], adict, board)
    obj_b, img_b, _ = get_object_image_points([names_b[n] for n in common], adict, board)

    # Keep only pairs where both cameras detected the board
    obj_stereo, img_stereo_a, img_stereo_b = [], [], []
    for oa, ia, ob, ib in zip(obj_a, img_a, obj_b, img_b):
        if oa is not None and ob is not None:
            obj_stereo.append(oa)
            img_stereo_a.append(ia)
            img_stereo_b.append(ib)

    print(f"Valid pairs for stereo: {len(obj_stereo)}")

    rms, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
    objectPoints  = obj_stereo,
    imagePoints1  = img_stereo_a,
    imagePoints2  = img_stereo_b,
    cameraMatrix1 = K_a,
    distCoeffs1   = dist_a,
    cameraMatrix2 = K_b,
    distCoeffs2   = dist_b,
    imageSize     = img_size,
    flags         = cv.CALIB_FIX_INTRINSIC
    )

    print(f"Stereo RPE {label}: {rms:.4f} px")
    return R, T, E, F, rms

def save_json(path, data):
    json.dump(data, open(path, "w"), indent=2)
    print(f"Saved: {path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cam0_dir = os.path.join(CALIB_DIR, "cam0_imgs")
    cam1_dir = os.path.join(CALIB_DIR, "cam1_imgs")
    cam2_dir = os.path.join(CALIB_DIR, "cam2_imgs")

    has_cam1 = os.path.isdir(cam1_dir) and len(glob.glob(os.path.join(cam1_dir, "*.png"))) > 0
    has_cam2 = os.path.isdir(cam2_dir) and len(glob.glob(os.path.join(cam2_dir, "*.png"))) > 0

    if not has_cam1 and not has_cam2:
        print("ERROR: No images found in cam1_imgs or cam2_imgs.")
        sys.exit(1)

    # ── Monocular cam0 ──
    K0, dist0, _, _, img_size, dict0, valid0, board, adict = config.calibrate_camera(cam0_dir, "cam0")
    save_json(os.path.join(CALIB_DIR, "cam0_intrinsics.json"), {
        "K": K0.tolist(), "dist": dist0.squeeze().tolist(),
        "dict": int(dict0), "SX": config.SX, "SY": config.SY,
        "square_m": config.SQUARE, "marker_m": config.MARKER, "img_size": img_size
    })

    # ── Monocular cam1 + stereo cam0-cam1 ──
    if has_cam1:
        K1, dist1, _, _, img_size1, dict1, valid1, _, _ = config.calibrate_camera(cam1_dir, "cam1")
        save_json(os.path.join(CALIB_DIR, "cam1_intrinsics.json"), {
            "K": K1.tolist(), "dist": dist1.squeeze().tolist(),
            "dict": int(dict1), "SX": config.SX, "SY": config.SY,
            "square_m": config.SQUARE, "marker_m": config.MARKER, "img_size": img_size1
        })

        R01, T01, E01, F01, rms01 = stereo_calibrate(
            K0, dist0, valid0, K1, dist1, valid1, adict, board, img_size, "cam0-cam1")
        save_json(os.path.join(CALIB_DIR, "stereo_cam0_cam1.json"), {
            "R": R01.tolist(), "T": T01.tolist(),
            "E": E01.tolist(), "F": F01.tolist(), "rms": rms01,
            "cam0_K": K0.tolist(), "cam0_dist": dist0.squeeze().tolist(),
            "cam1_K": K1.tolist(), "cam1_dist": dist1.squeeze().tolist(),
            "img_size": img_size
        })

    # ── Monocular cam2 + stereo cam0-cam2 ──
    if has_cam2:
        K2, dist2, _, _, img_size2, dict2, valid2, _, _ = config.calibrate_camera(cam2_dir, "cam2")
        save_json(os.path.join(CALIB_DIR, "cam2_intrinsics.json"), {
            "K": K2.tolist(), "dist": dist2.squeeze().tolist(),
            "dict": int(dict2), "SX": config.SX, "SY": config.SY,
            "square_m": config.SQUARE, "marker_m": config.MARKER, "img_size": img_size2
        })

        R02, T02, E02, F02, rms02 = stereo_calibrate(
            K0, dist0, valid0, K2, dist2, valid2, adict, board, img_size, "cam0-cam2")
        save_json(os.path.join(CALIB_DIR, "stereo_cam0_cam2.json"), {
            "R": R02.tolist(), "T": T02.tolist(),
            "E": E02.tolist(), "F": F02.tolist(), "rms": rms02,
            "cam0_K": K0.tolist(), "cam0_dist": dist0.squeeze().tolist(),
            "cam2_K": K2.tolist(), "cam2_dist": dist2.squeeze().tolist(),
            "img_size": img_size
        })

    # ── cam1-cam2 via cam0 ──
    if has_cam1 and has_cam2:
        print("\n── Computing cam1-cam2 via cam0 ──")
        # R12 = R02 @ R01^T,  T12 = T02 - R12 @ T01
        R12 = R02 @ R01.T
        T12 = T02 - R12 @ T01
        save_json(os.path.join(CALIB_DIR, "stereo_cam1_cam2.json"), {
            "R": R12.tolist(), "T": T12.tolist(),
            "note": "Derived from cam0-cam1 and cam0-cam2, not directly calibrated"
        })
        print(f"Saved: stereo_cam1_cam2.json")

    print("\nCalibration complete.")

if __name__ == "__main__":
    main()

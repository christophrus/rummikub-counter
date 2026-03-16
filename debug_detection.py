"""Debug: Test tile detection strategies on light table image."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
import cv2
import numpy as np
from app.services.tile_detector import (
    detect_tiles, draw_detections, _score_tile_set,
    _detect_by_table_diff, _detect_by_saturation, _detect_by_brightness,
    _detect_by_multi_threshold, _detect_by_local_otsu,
    _detect_by_canny, _detect_by_adaptive_threshold,
    _split_merged_regions, _non_max_suppression
)
from app.utils.image_processing import resize_image

image = cv2.imread("test.jpg")
resized = resize_image(image)
img_area = resized.shape[0] * resized.shape[1]

print("=== Per-strategy results (no CLAHE) ===")
strategies = [
    ("table_diff", _detect_by_table_diff),
    ("saturation", _detect_by_saturation),
    ("brightness", _detect_by_brightness),
    ("multi_thresh", _detect_by_multi_threshold),
    ("local_otsu", _detect_by_local_otsu),
    ("canny", _detect_by_canny),
    ("adaptive", _detect_by_adaptive_threshold),
]
for name, fn in strategies:
    tiles = fn(resized, img_area)
    tiles = _split_merged_regions(tiles, resized, img_area)
    tiles = _non_max_suppression(tiles, overlap_thresh=0.3)
    score = _score_tile_set(tiles, img_area)
    print(f"  {name}: {len(tiles)} tiles, score={score:.1f}")
    rows = {}
    for t in tiles:
        row = t['y'] // 100
        rows[row] = rows.get(row, 0) + 1
    for row in sorted(rows):
        print(f"    Row y~{row*100}: {rows[row]}")

# Final combined result
tiles_pre = detect_tiles(resized)
print(f"\nFinal detect_tiles: {len(tiles_pre)} tiles")
rows = {}
for t in tiles_pre:
    row = t['y'] // 100
    rows[row] = rows.get(row, 0) + 1
for row in sorted(rows):
    print(f"  Row y~{row*100}: {rows[row]}")

# Diagnose: check what the multi_thresh tiles look like before final filter
from app.services.tile_detector import _estimate_single_tile_size
tiles_mt = _detect_by_multi_threshold(resized, img_area)
tiles_mt = _split_merged_regions(tiles_mt, resized, img_area)
tiles_mt = _non_max_suppression(tiles_mt, overlap_thresh=0.3)
est_w, est_h = _estimate_single_tile_size(tiles_mt, img_area)
print(f"\nMulti_thresh pre-filter: {len(tiles_mt)} tiles")
print(f"Estimated tile size: {est_w}x{est_h}")
print(f"Area range: {est_w*est_h*0.25:.0f} - {est_w*est_h*3:.0f}")

# Check what gets filtered
filtered_out = []
for t in tiles_mt:
    area = t["w"] * t["h"]
    aspect = t["w"] / t["h"] if t["h"] > 0 else 0
    min_a = est_w * est_h * 0.25
    max_a = est_w * est_h * 3.0
    reasons = []
    if area < min_a or area > max_a:
        reasons.append(f"area={area} [{min_a:.0f}-{max_a:.0f}]")
    if not (0.25 < aspect < 1.5):
        reasons.append(f"aspect={aspect:.2f}")
    if t["y"] <= est_h * 0.05:
        reasons.append(f"top_edge y={t['y']}")
    if t["y"] + t["h"] >= resized.shape[0] - est_h * 0.05:
        reasons.append(f"bot_edge y+h={t['y']+t['h']}")
    if reasons:
        filtered_out.append((t, reasons))

print(f"\nFiltered out: {len(filtered_out)}")
for t, reasons in filtered_out:
    print(f"  ({t['x']},{t['y']}) {t['w']}x{t['h']} aspect={t['w']/t['h']:.2f}: {', '.join(reasons)}")

# Save debug visualization
debug_img = draw_detections(resized, tiles_pre)
cv2.imwrite("test_result_no_clahe.png", debug_img)
print(f"\nDebug image saved to test_result_no_clahe.png")

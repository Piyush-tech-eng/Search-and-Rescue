import cv2 as cv
import numpy as np
import first as f

# -----------------------------
# Read image
# -----------------------------
h, w = f.output.shape[:2]

# -----------------------------
# Convert to HSV
# -----------------------------
hsv = cv.cvtColor(f.output, cv.COLOR_BGR2HSV)

# -----------------------------
# Loose green detection
# -----------------------------
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])
green_mask = cv.inRange(hsv, lower_green, upper_green)

# -----------------------------
# Morphological cleanup
# -----------------------------
kernel = np.ones((5, 5), np.uint8)
green_mask = cv.morphologyEx(green_mask, cv.MORPH_CLOSE, kernel)
green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)

# -----------------------------
# Connected components
# -----------------------------
num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
    green_mask, connectivity=8
)

# -----------------------------
# Find LAND component
# Rule: large + touches image border
# -----------------------------
land_label = None
max_area = 0

for i in range(1, num_labels):  # skip background
    x, y, w_c, h_c, area = stats[i]

    touches_border = (
        x == 0 or y == 0 or
        x + w_c >= w - 1 or
        y + h_c >= h - 1
    )

    if touches_border and area > max_area:
        max_area = area
        land_label = i

# -----------------------------
# Final land mask
# -----------------------------
land_mask = np.zeros_like(green_mask)
if land_label is not None:
    land_mask[labels == land_label] = 255

ocean_mask = cv.bitwise_not(land_mask)

# -----------------------------
# Overlay colors (balanced)
# -----------------------------
overlay = f.output.copy()

LAND_COLOR  = np.array([0, 255, 255])   # perceptually balanced land
OCEAN_COLOR = np.array([190, 60, 60])    # perceptually balanced ocean

overlay[land_mask > 0] = LAND_COLOR
overlay[ocean_mask > 0] = OCEAN_COLOR

# -----------------------------
# Blend
# -----------------------------
output = cv.addWeighted(overlay, 0.45, f.output, 0.55, 0)


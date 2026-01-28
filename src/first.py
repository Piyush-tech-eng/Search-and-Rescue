import cv2 as cv
import numpy as np

img = cv.imread('data/3.png') 
def rescale_frame(frame,scale=1):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def get_color_name(h, s, v):
    # Very dark → ignore (not in your palette)
    if v < 50:
        return "Unknown"

    # White / Gray (ONLY when saturation is VERY low)
    if s < 20:
        if v > 200:
            return "Gray"
        else:
            return "Gray"

    # Pastel protection: allow color even if saturation is low
    if (h < 10) or (h > 165):
        return "Red"
    elif 20 <= h <= 35:
        return "Yellow"
    elif 40 <= h <= 85:
        return "Green"
    elif 90 <= h <= 130:
        return "Blue"
    elif 135 <= h <= 165:
        return "Pink"
    else:
        return "Unknown"


resized_image=rescale_frame(img)
cv.imshow('resized_img',resized_image)
gray=cv.cvtColor(resized_image,cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (9, 9),1.5)
_, thresh = cv.threshold(blur, 0, 255,cv.THRESH_BINARY + cv.THRESH_OTSU)
contours, _ = cv.findContours(
    thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)
hsv = cv.cvtColor(resized_image, cv.COLOR_BGR2HSV)

center_arr=[]
output = resized_image.copy()
for cnt in contours:
    # Shape approximation
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True) # Keep 0.02 or try 0.03 if needed

    # Get center for text
    M = cv.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    shape = "Unknown"

    # Triangle (Standard rule)
    if len(approx) == 3:
        shape = "Triangle"

    # Square / Rectangle (with Extent check)
    elif 4 <= len(approx) <= 6:
        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # Calculate extent
        area = cv.contourArea(cnt)
        rect_area = w * h
        extent = float(area) / rect_area

        # Only classify as Square if it actually looks like a solid block (high extent)
        if extent > 0.70: 
            if 0.85 <= aspect_ratio <= 1.15:
                shape = "Square"
        # If points are 4-6 but it doesn't fill the box, it's likely a noisy Triangle
        elif extent < 0.65:
            shape = "Triangle"

    # Star (non-convex with many vertices)
    elif len(approx) >= 8 and not cv.isContourConvex(approx):
        shape = "Star"

    # Circle (last and strictest)
    else:
        area = cv.contourArea(cnt)
        peri = cv.arcLength(cnt, True)
        if peri > 0:
            circularity = 4 * np.pi * area / (peri * peri)
            if 0.7 <= circularity <= 1:
                shape = "Circle"
    # Create mask for this contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask = cv.erode(mask, None, iterations=2)

    cv.drawContours(mask, [cnt], -1, 255, -1)

    # Mean HSV value inside contour
    ys, xs = np.where(mask == 255)
    h_vals = hsv[ys, xs, 0]
    s_vals = hsv[ys, xs, 1]
    v_vals = hsv[ys, xs, 2]

    h = int(np.median(h_vals))
    s = int(np.median(s_vals))
    v = int(np.median(v_vals))

    color = get_color_name(h, s, v)
    center_arr.append([shape, color, cx, cy])
    # Draw contour & label
    cv.drawContours(output, [cnt], -1, (0, 255, 0), 2)
    cv.putText(
        output, shape, (cx - 40, cy),
        cv.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 255), 2
    )
    cv.circle(output, (cx, cy), 5, (0, 0, 255), -1)


import cv2 as cv
import numpy as np
import math

# ======================================================
# CONFIGURATION
# ======================================================

CASUALTY_PRIORITY = {
    "Star": 3,
    "Triangle": 2,
    "Square": 1
}

EMERGENCY_PRIORITY = {
    "Red": 3,
    "Yellow": 2,
    "Green": 1
}

CAMP_CAPACITIES = {
    "Blue": 4,
    "Pink": 3,
    "Gray": 2
}

# (Not used in new visualization but kept for safety)
ARROW_COLORS = {
    "Blue": (255, 0, 0),
    "Pink": (203, 192, 255),
    "Gray": (128, 128, 128)
}

ALPHA = 0.6  # 60% Priority, 40% Distance

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def rescale_frame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def get_color_name(h, s, v):
    if v < 50: return "Unknown"
    if s < 20: return "Gray" if v > 200 else "Gray"

    if (h < 10) or (h > 165): return "Red"
    elif 20 <= h <= 35: return "Yellow"
    elif 40 <= h <= 85: return "Green"
    elif 90 <= h <= 130: return "Blue"
    elif 135 <= h <= 165: return "Pink"
    else: return "Unknown"

def get_priority_norm(target):
    raw = CASUALTY_PRIORITY.get(target["shape"], 1) * EMERGENCY_PRIORITY.get(target["color"], 1)
    return (raw - 1) / 8.0

def precompute_target_scores(target, camps, alpha):
    p_norm = get_priority_norm(target)
    distances = [math.dist(target["center"], camp["center"]) for camp in camps]
    
    min_d = min(distances)
    max_d = max(distances)
    range_d = max_d - min_d
    
    camp_scores = {}
    for i, camp in enumerate(camps):
        raw_d = distances[i]
        if range_d == 0:
            d_norm = 1.0 
        else:
            d_norm = (max_d - raw_d) / range_d
        
        score = (alpha * p_norm) + ((1 - alpha) * d_norm)
        camp_scores[camp["color"]] = score
        
    return camp_scores

def parse_objects(center_arr):
    sources = []
    targets = []
    for shape, color, x, y in center_arr:
        obj = {"shape": shape, "color": color, "center": (x, y)}
        if shape == "Circle":
            sources.append(obj)
        else:
            targets.append(obj)
    sources.sort(key=lambda s: s["color"])
    return sources, targets

def sort_targets_descending(targets):
    return sorted(targets, key=lambda t: 
                  CASUALTY_PRIORITY.get(t["shape"],1) * EMERGENCY_PRIORITY.get(t["color"],1), 
                  reverse=True)

def solve_backtracking_max_score(target_idx, sorted_targets, sources, current_capacities):
    if target_idx >= len(sorted_targets):
        return 0.0, []

    current_target = sorted_targets[target_idx]
    possible_scores = precompute_target_scores(current_target, sources, ALPHA)
    
    max_total_score = -1.0
    best_path = []
    assignment_made = False
    
    for camp in sources:
        camp_name = camp["color"]
        if current_capacities[camp_name] > 0:
            assignment_made = True
            step_score = possible_scores[camp_name]
            
            current_capacities[camp_name] -= 1
            remaining_score, remaining_path = solve_backtracking_max_score(
                target_idx + 1, sorted_targets, sources, current_capacities
            )
            current_capacities[camp_name] += 1
            
            current_total = step_score + remaining_score
            if current_total > max_total_score:
                max_total_score = current_total
                this_move = {
                    "camp_color": camp_name,
                    "camp_center": camp["center"],
                    "target_shape": current_target["shape"],
                    "target_color": current_target["color"],
                    "target_center": current_target["center"],
                    "step_score": step_score
                }
                best_path = [this_move] + remaining_path
                
    if not assignment_made:
        return solve_backtracking_max_score(target_idx + 1, sorted_targets, sources, current_capacities)

    return max_total_score, best_path

def visualize_path(base_img, path, total_score):
    vis = base_img.copy()
    red_color = (0, 0, 255)  # All arrows RED
    for item in path:
        c_color = red_color
        cv.arrowedLine(vis, item["camp_center"], item["target_center"], c_color, 2, tipLength=0.03)
        cv.circle(vis, item["target_center"], 5, c_color, -1)
        label = f"{item['camp_color']}"
        cv.putText(vis, label, (item["target_center"][0] + 10, item["target_center"][1]), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, c_color, 1)

    cv.putText(vis, f"Max Total Score: {total_score:.3f}", (20, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis

def compute_camp_priority(final_assignments, scale=10):
    camp_score_sum = {"Blue": 0.0, "Pink": 0.0, "Gray": 0.0}
    for a in final_assignments:
        camp_score_sum[a["camp_color"]] += a["step_score"]

    blue_val = int(round(camp_score_sum["Blue"] * scale))
    pink_val = int(round(camp_score_sum["Pink"] * scale))
    gray_val = int(round(camp_score_sum["Gray"] * scale))
    return [[blue_val, pink_val, gray_val]]


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    
    # 1. Initialize lists to hold data for ALL images
    all_camp_priorities = [] 
    all_priority_ratios = [] 
    image_ratios_paired = []  # <--- FIXED: Now initialized

    # Loop through images 1 to 10
    for i in range(1, 11):
        img_name = f"{i}.png"
        img_path = f"data/{img_name}"
        
        print(f"\n--- Processing {img_path} ---")

        img = cv.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}. Skipping...")
            continue 

        # --------------------------------------------------
        # PHASE 1: PRE-PROCESSING & DETECTION
        # --------------------------------------------------
        resized_image = rescale_frame(img)
        gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (9, 9), 1.5)
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        hsv_img = cv.cvtColor(resized_image, cv.COLOR_BGR2HSV)

        center_arr = []
        contour_img = resized_image.copy()

        for cnt in contours:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            M = cv.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            shape = "Unknown"
            if len(approx) == 3: shape = "Triangle"
            elif 4 <= len(approx) <= 6:
                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = cv.contourArea(cnt)
                rect_area = w * h
                extent = float(area) / rect_area
                if extent > 0.70 and 0.85 <= aspect_ratio <= 1.15: shape = "Square"
                elif extent < 0.65: shape = "Triangle"
            elif len(approx) >= 8 and not cv.isContourConvex(approx): shape = "Star"
            else:
                area = cv.contourArea(cnt)
                peri = cv.arcLength(cnt, True)
                if peri > 0:
                    circularity = 4 * np.pi * area / (peri * peri)
                    if 0.7 <= circularity <= 1: shape = "Circle"

            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask = cv.erode(mask, None, iterations=2)
            cv.drawContours(mask, [cnt], -1, 255, -1)
            ys, xs = np.where(mask == 255)
            if len(ys) > 0:
                h_vals = hsv_img[ys, xs, 0]
                s_vals = hsv_img[ys, xs, 1]
                v_vals = hsv_img[ys, xs, 2]
                color = get_color_name(int(np.median(h_vals)), int(np.median(s_vals)), int(np.median(v_vals)))
            else: color = "Unknown"
            center_arr.append([shape, color, cx, cy])

        # --------------------------------------------------
        # PHASE 2: TERRAIN & OVERLAY
        # --------------------------------------------------
        h_img, w_img = contour_img.shape[:2]
        hsv_terrain = cv.cvtColor(contour_img, cv.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv.inRange(hsv_terrain, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv.morphologyEx(green_mask, cv.MORPH_CLOSE, kernel)
        green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(green_mask, connectivity=8)
        land_label = None
        max_area = 0

        for k in range(1, num_labels):
            x, y, w_c, h_c, area = stats[k]
            touches_border = (x == 0 or y == 0 or x + w_c >= w_img - 1 or y + h_c >= h_img - 1)
            if touches_border and area > max_area:
                max_area = area
                land_label = k

        land_mask = np.zeros_like(green_mask)
        if land_label is not None:
            land_mask[labels == land_label] = 255
        ocean_mask = cv.bitwise_not(land_mask)

        overlay = contour_img.copy()
        LAND_COLOR = np.array([0, 255, 255])
        OCEAN_COLOR = np.array([190, 60, 60])
        overlay[land_mask > 0] = LAND_COLOR
        overlay[ocean_mask > 0] = OCEAN_COLOR
        background_img = cv.addWeighted(overlay, 0.45, contour_img, 0.55, 0)

        # --------------------------------------------------
        # PHASE 3: ALGORITHM & CALCULATIONS
        # --------------------------------------------------
        sources, targets = parse_objects(center_arr)
        sorted_targets = sort_targets_descending(targets)
        initial_caps = CAMP_CAPACITIES.copy()
        
        final_score, optimal_path = solve_backtracking_max_score(0, sorted_targets, sources, initial_caps)

        # 1. Calculate Priority List
        current_priority = compute_camp_priority(optimal_path, scale=10)

        # 2. Append to Master List
        ratio = 0.0
        if len(current_priority) > 0:
            p_list = current_priority[0]
            all_camp_priorities.append(p_list)
            
            numerator = sum(p_list)
            denominator = len(targets)
            
            if denominator > 0:
                ratio = numerator / denominator
            
            print(f"  -> Priority: {p_list}, Sum: {numerator}, Targets: {denominator}, Ratio: {ratio:.3f}")
        else:
            all_camp_priorities.append([0, 0, 0])
            print(f"  -> Priority: [0,0,0], Ratio: 0.000")

        all_priority_ratios.append(ratio)
        
        # FIXED: Append to the sorting list
        image_ratios_paired.append((ratio, img_name))

        # Visualization
        final_img = visualize_path(background_img, optimal_path, final_score)
        cv.imshow("Final Optimal Path", final_img)
        if cv.waitKey(500) == ord('q'): 
            break

    cv.destroyAllWindows()

    # ======================================================
    # FINAL SORTING LOGIC
    # ======================================================
    print("\n==========================================")
    print("FINAL SORTED RESULTS (Highest Priority First)")
    print("==========================================")

    # Sort the paired list based on ratio (x[0]) in Descending order
    sorted_images = sorted(image_ratios_paired, key=lambda x: x[0], reverse=True)

    print(f"{'Image Name':<15} | {'Priority Ratio':<15}")
    print("-" * 35)
    
    for ratio, name in sorted_images:
        print(f"{name:<15} | {ratio:.3f}")

    sorted_image_names = [name for ratio, name in sorted_images]
    print("\nSorted Image List:", sorted_image_names)

    print("\n==========================================")
    print("FINAL RESULTS LISTS")
    print("==========================================")
    print("Camp Priority List:", all_camp_priorities)
    print("Priority Ratio List:", all_priority_ratios)
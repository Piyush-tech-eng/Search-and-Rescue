import cv2 as cv
import numpy as np
import math

# ======================================================
# 1. CONFIGURATION & CONSTANTS
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

# Visualization Colors
ARROW_COLOR = (0, 0, 255) # Red for all arrows
LAND_COLOR  = np.array([0, 255, 255])   # Yellowish
OCEAN_COLOR = np.array([190, 60, 60])   # Blueish

# Algorithm Weights
ALPHA = 0.7 

# ======================================================
# 2. HELPER FUNCTIONS
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

def priority_score(target):
    return (
        CASUALTY_PRIORITY.get(target["shape"], 0) *
        EMERGENCY_PRIORITY.get(target["color"], 0)
    )

def parse_objects(center_arr):
    sources = []
    targets = []

    for shape, color, x, y in center_arr:
        obj = {"shape": shape, "color": color, "center": (x, y)}
        if shape == "Circle":
            sources.append(obj)
        else:
            targets.append(obj)

    # Enforce camp order: Blue -> Pink -> Gray
    color_order = {"Blue": 0, "Pink": 1, "Gray": 2}
    valid_sources = [s for s in sources if s["color"] in color_order]
    valid_sources.sort(key=lambda s: color_order[s["color"]])
    
    return valid_sources, targets

def sort_targets_by_priority(targets):
    return sorted(targets, key=lambda t: -priority_score(t))

def normalize_priority(targets):
    if not targets: return
    scores = [priority_score(t) for t in targets]
    max_p = max(scores) if scores else 1
    
    for t in targets:
        p = priority_score(t)
        t["priority_norm"] = (max_p - p + 1) / max_p

def compute_distances(sources, targets):
    dist = []
    for src in sources:
        for tgt in targets:
            dist.append({
                "camp_color": src["color"],
                "camp_center": src["center"],
                "target_shape": tgt["shape"],
                "target_color": tgt["color"],
                "target_center": tgt["center"],
                "distance": math.dist(src["center"], tgt["center"])
            })
    return dist

def normalize_distances(distances, num_sources, num_targets):
    if num_targets == 0: return
    for i in range(num_sources):
        start = i * num_targets
        end = (i + 1) * num_targets
        
        batch = distances[start:end]
        if not batch: continue

        min_d = min(x["distance"] for x in batch)
        max_d = max(x["distance"] for x in batch)
        
        for j in range(len(batch)):
            idx = start + j
            a = distances[idx]["distance"] - min_d
            b = max_d - min_d
            
            if b != 0:
                distances[idx]["distance_norm"] = 1 - (a / b)
            else:
                distances[idx]["distance_norm"] = 0

def compute_score(p_norm, d_norm, alpha):
    return alpha * p_norm + (1 - alpha) * d_norm

def add_scores(distances, targets, alpha=0.7):
    priority_lookup = {
        (t["shape"], t["color"], t["center"]): t.get("priority_norm", 0)
        for t in targets
    }

    for d in distances:
        key = (d["target_shape"], d["target_color"], d["target_center"])
        p_norm = priority_lookup.get(key, 0)
        d_norm = d.get("distance_norm", 0)
        d["score"] = compute_score(p_norm, d_norm, alpha)

def assign_targets_priority_wise(targets_sorted, distances, capacity):
    final_assignments = []
    current_capacity = capacity.copy()

    for tgt in targets_sorted:
        candidates = []
        for d in distances:
            if (d["target_shape"] == tgt["shape"] and 
                d["target_color"] == tgt["color"] and 
                d["target_center"] == tgt["center"]):
                candidates.append(d)
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        assigned = False
        for c in candidates:
            camp = c["camp_color"]
            if current_capacity.get(camp, 0) > 0:
                final_assignments.append({
                    "camp_color": camp,
                    "target_shape": tgt["shape"],
                    "target_color": tgt["color"],
                    "target_center": tgt["center"],
                    "priority": priority_score(tgt),
                    "score": c["score"]
                })
                current_capacity[camp] -= 1
                assigned = True
                break
            
    return final_assignments

def build_camp_lookup(sources):
    return {s["color"]: s["center"] for s in sources}

def visualize_assignments(output_img, final_assignments, camp_lookup):
    vis = output_img.copy()
    for a in final_assignments:
        camp_color = a["camp_color"]
        camp_center = camp_lookup.get(camp_color)
        if not camp_center: continue
        
        target_center = a["target_center"]
        
        # Red arrows for everyone
        color = ARROW_COLOR 

        cv.arrowedLine(vis, camp_center, target_center, color, thickness=2, tipLength=0.03)
        cv.circle(vis, target_center, 5, color, -1)
        
        label = f"{camp_color}"
        cv.putText(vis, label, (target_center[0] + 5, target_center[1] - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis

def compute_camp_priority_list(final_assignments, scale=10):
    camp_score_sum = {"Blue": 0.0, "Pink": 0.0, "Gray": 0.0}
    for a in final_assignments:
        camp_score_sum[a["camp_color"]] += a["score"] # Using 'score' field from logic

    blue_val = int(round(camp_score_sum["Blue"] * scale))
    pink_val = int(round(camp_score_sum["Pink"] * scale))
    gray_val = int(round(camp_score_sum["Gray"] * scale))
    return [blue_val, pink_val, gray_val]

def build_image_list_output(final_assignments):
    AGE_GROUP = {"Star": 3, "Triangle": 2, "Square": 1}
    MEDICAL_EMERGENCY = {"Red": 3, "Yellow": 2, "Green": 1}
    
    Image_n = {"Blue": [], "Pink": [], "Gray": []}
    
    for a in final_assignments:
        shape = a["target_shape"]
        color = a["target_color"]
        camp = a["camp_color"]
        
        if camp in Image_n:
            Image_n[camp].append([
                AGE_GROUP.get(shape, 0),
                MEDICAL_EMERGENCY.get(color, 0)
            ])
            
    return [Image_n["Blue"], Image_n["Pink"], Image_n["Gray"]]

# ======================================================
# 3. MAIN EXECUTION FLOW
# ======================================================

if __name__ == "__main__":
    
    # Storage for results across all images
    all_camp_priorities = []
    all_priority_ratios = []
    image_ratios_paired = []  # List of tuples (ratio, image_name)

    # LOOP 1 to 10
    for i in range(1, 11):
        img_name = f"{i}.png"
        img_path = f"data/{img_name}"
        
        print(f"\n--- Processing {img_path} ---")

        # --- PHASE 1: READ & DETECT ---
        img = cv.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}. Skipping...")
            continue

        resized_image = rescale_frame(img)
        gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (9, 9), 1.5)
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        hsv = cv.cvtColor(resized_image, cv.COLOR_BGR2HSV)

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
                h_vals = hsv[ys, xs, 0]
                s_vals = hsv[ys, xs, 1]
                v_vals = hsv[ys, xs, 2]
                color = get_color_name(int(np.median(h_vals)), int(np.median(s_vals)), int(np.median(v_vals)))
            else: color = "Unknown"

            center_arr.append([shape, color, cx, cy])
            # Visualization stuff for contour_img
            cv.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
            cv.putText(contour_img, shape, (cx - 40, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)

        # --- PHASE 2: TERRAIN ---
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
        overlay[land_mask > 0] = LAND_COLOR
        overlay[ocean_mask > 0] = OCEAN_COLOR
        terrain_img = cv.addWeighted(overlay, 0.45, contour_img, 0.55, 0)

        # --- PHASE 3: ALGORITHM ---
        sources, targets = parse_objects(center_arr)
        
        if not sources:
            print("No Camps detected! Assigning empty priorities.")
            all_camp_priorities.append([0, 0, 0])
            all_priority_ratios.append(0.0)
            image_ratios_paired.append((0.0, img_name))
        else:
            targets_sorted = sort_targets_by_priority(targets)
            normalize_priority(targets_sorted)
            distances = compute_distances(sources, targets_sorted)
            normalize_distances(distances, num_sources=len(sources), num_targets=len(targets_sorted))
            add_scores(distances, targets_sorted, ALPHA)
            
            final_assignments = assign_targets_priority_wise(targets_sorted, distances, CAMP_CAPACITIES)

            # 1. Image_n list (representation)
            Image_ = build_image_list_output(final_assignments)
            print(f"Image_ Representation: {Image_}")

            # 2. Camp Priority for this image
            current_camp_priority = compute_camp_priority_list(final_assignments, scale=10)
            all_camp_priorities.append(current_camp_priority)
            
            # 3. Priority Ratio
            numerator = sum(current_camp_priority)
            denominator = len(targets)
            ratio = numerator / denominator if denominator > 0 else 0.0
            
            all_priority_ratios.append(ratio)
            image_ratios_paired.append((ratio, img_name))
            
            print(f"Camp Priority: {current_camp_priority}")
            print(f"Ratio: {ratio:.3f}")

            # Visualization
            camp_lookup = build_camp_lookup(sources)
            vis_img = visualize_assignments(terrain_img, final_assignments, camp_lookup)
            cv.imshow("Final Assignments", vis_img)
            
            # Use waitKey(100) to auto-advance, or 0 to pause
            if cv.waitKey(100) == ord('q'):
                break

    cv.destroyAllWindows()

    # ======================================================
    # FINAL OUTPUTS
    # ======================================================
    print("\n\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    # 1. Camp Priorities for all 10 images
    print("\n1. All Camp Priorities (List of Lists):")
    print(all_camp_priorities)

    # 2. Priority Ratios for all 10 images
    print("\n2. All Priority Ratios:")
    print([round(r, 3) for r in all_priority_ratios])

    # 3. Sorting Images by Priority Ratio (Descending)
    print("\n3. Images Sorted by Priority Ratio (Highest First):")
    
    sorted_images = sorted(image_ratios_paired, key=lambda x: x[0], reverse=True)
    
    print(f"{'Image':<10} | {'Ratio':<10}")
    print("-" * 25)
    for ratio, name in sorted_images:
        print(f"{name:<10} | {ratio:.3f}")

    sorted_names_only = [name for ratio, name in sorted_images]
    print("\nSorted Image List: ", sorted_names_only)
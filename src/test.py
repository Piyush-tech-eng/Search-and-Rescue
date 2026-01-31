import cv2 as cv
import numpy as np
import math

import first as f     # contour detection + center_arr
import second as s    # land/ocean overlay -> output image

# ======================================================
# 1. Configuration & Priorities
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

# User parameters
ALPHA = 0.6  # 60% Priority, 40% Distance

# ======================================================
# 2. Helpers: Normalization & Scoring
# ======================================================

def get_priority_norm(target):
    """
    Calculates normalized priority (0.0 to 1.0).
    Max Raw = 3*3=9, Min Raw = 1*1=1.
    """
    raw = CASUALTY_PRIORITY.get(target["shape"], 1) * EMERGENCY_PRIORITY.get(target["color"], 1)
    # Normalize between 0 and 1
    return (raw - 1) / 8.0

def precompute_target_scores(target, camps, alpha):
    """
    Calculates the 'Score' this target would generate for EACH camp.
    Returns a dict: { 'Blue': score_blue, 'Pink': score_pink, ... }
    """
    p_norm = get_priority_norm(target)
    
    # 1. Calculate raw distances to all camps
    distances = []
    for camp in camps:
        d = math.dist(target["center"], camp["center"])
        distances.append(d)
        
    # 2. Normalize distances (Local Normalization)
    # Closer = Higher Reward (1.0). Farther = Lower Reward (0.0)
    min_d = min(distances)
    max_d = max(distances)
    range_d = max_d - min_d
    
    camp_scores = {}
    
    for i, camp in enumerate(camps):
        raw_d = distances[i]
        
        # Avoid division by zero if all camps are equidistant
        if range_d == 0:
            d_norm = 1.0 
        else:
            # Invert: (Max - Current) / Range
            # If Raw == Min (Closest), Result = 1.0
            # If Raw == Max (Farthest), Result = 0.0
            d_norm = (max_d - raw_d) / range_d
            
        # 3. Apply Formula
        # Score = alpha * p_norm + (1 - alpha) * d_norm
        score = (alpha * p_norm) + ((1 - alpha) * d_norm)
        
        camp_scores[camp["color"]] = score
        
    return camp_scores

# ======================================================
# 3. Parsing & Sorting
# ======================================================

def parse_objects(center_arr):
    sources = []
    targets = []

    for shape, color, x, y in center_arr:
        obj = {"shape": shape, "color": color, "center": (x, y)}
        if shape == "Circle":
            sources.append(obj)
        else:
            targets.append(obj)
            
    # Sort sources (optional, but good for consistency)
    sources.sort(key=lambda s: s["color"])
    return sources, targets

def sort_targets_descending(targets):
    # Sort by Raw Priority value
    return sorted(targets, key=lambda t: 
                  CASUALTY_PRIORITY.get(t["shape"],1) * EMERGENCY_PRIORITY.get(t["color"],1), 
                  reverse=True)

# ======================================================
# 4. Backtracking (Maximization)
# ======================================================

def solve_backtracking_max_score(target_idx, sorted_targets, sources, current_capacities):
    """
    DFS to find the assignment with the MAXIMUM Total Score.
    """
    # Base Case: All targets assigned
    if target_idx >= len(sorted_targets):
        return 0.0, []

    current_target = sorted_targets[target_idx]
    
    # Pre-calculate scores for this target against all camps to save time inside loop
    # (In a strict sense, we could do this inside the loop, but this is cleaner)
    possible_scores = precompute_target_scores(current_target, sources, ALPHA)
    
    max_total_score = -1.0
    best_path = []
    assignment_made = False
    
    # Try assigning to every camp
    for camp in sources:
        camp_name = camp["color"]
        
        if current_capacities[camp_name] > 0:
            assignment_made = True
            
            # 1. Get Score for this move
            step_score = possible_scores[camp_name]
            
            # 2. Update Capacity
            current_capacities[camp_name] -= 1
            
            # 3. Recurse
            remaining_score, remaining_path = solve_backtracking_max_score(
                target_idx + 1,
                sorted_targets,
                sources,
                current_capacities
            )
            
            # 4. Backtrack
            current_capacities[camp_name] += 1
            
            # 5. Check Total
            current_total = step_score + remaining_score
            
            # We want to MAXIMIZE score
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
                
    # Handle case where no assignment is possible (e.g. all full)
    if not assignment_made:
        return solve_backtracking_max_score(target_idx + 1, sorted_targets, sources, current_capacities)

    return max_total_score, best_path

# ======================================================
# 5. Visualization
# ======================================================

ARROW_COLORS = {
    "Blue": (255, 0, 0),
    "Pink": (203, 192, 255),
    "Gray": (128, 128, 128)
}

def visualize_path(base_img, path, total_score):
    vis = base_img.copy()
    
    for item in path:
        c_color = ARROW_COLORS.get(item["camp_color"], (0, 255, 0))
        
        cv.arrowedLine(vis, item["camp_center"], item["target_center"], c_color, 2, tipLength=0.03)
        cv.circle(vis, item["target_center"], 5, c_color, -1)
        
        label = f"{item['camp_color']}"
        cv.putText(vis, label, (item["target_center"][0] + 10, item["target_center"][1]), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, c_color, 1)

    # Show Max Score
    cv.putText(vis, f"Max Total Score: {total_score:.3f}", (20, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis

def build_image_list(final_assignments):

    AGE_GROUP = {
        "Star": 3,
        "Triangle": 2,
        "Square": 1
    }

    MEDICAL_EMERGENCY = {
        "Red": 3,
        "Yellow": 2,
        "Green": 1
    }

    Image_n = {
        "Blue": [],
        "Pink": [],
        "Gray": []
    }

    for a in final_assignments:
        Image_n[a["camp_color"]].append([
            AGE_GROUP[a["target_shape"]],
            MEDICAL_EMERGENCY[a["target_color"]]
        ])

    return [
        Image_n["Blue"],
        Image_n["Pink"],
        Image_n["Gray"]
    ]

def compute_camp_priority(final_assignments, scale=10):
    """
    Computes sum of scores for each camp, multiplies by `scale`,
    and returns list in order: [Blue, Pink, Gray]
    
    Output format:
    Camp_priority = [[blue_val, pink_val, gray_val]]
    """

    camp_score_sum = {
        "Blue": 0.0,
        "Pink": 0.0,
        "Gray": 0.0
    }

    # 1. Sum scores per camp
    for a in final_assignments:
        camp = a["camp_color"]
        camp_score_sum[camp] += a["step_score"]

    # 2. Scale scores
    blue_val = int(round(camp_score_sum["Blue"] * scale))
    pink_val = int(round(camp_score_sum["Pink"] * scale))
    gray_val = int(round(camp_score_sum["Gray"] * scale))

    # 3. Required output structure
    Camp_priority = [[blue_val, pink_val, gray_val]]

    return Camp_priority


# ======================================================
# 6. MAIN
# ======================================================

if __name__ == "__main__":
    
    # 1. Parse
    sources, targets = parse_objects(f.center_arr)
    
    # 2. Sort by Priority Descending
    sorted_targets = sort_targets_descending(targets)
    
    print(f"Detected {len(sources)} Camps and {len(targets)} Targets.")
    print(f"Optimizing for Formula: Score = {ALPHA}*P_norm + {1-ALPHA}*D_norm")
    
    # 3. Backtracking
    initial_caps = CAMP_CAPACITIES.copy()
    
    final_score, optimal_path = solve_backtracking_max_score(
        target_idx=0,
        sorted_targets=sorted_targets,
        sources=sources,
        current_capacities=initial_caps
    )

    # 4. Build Image_n
    Image_ = build_image_list(optimal_path)

    print("\n=== Image_ Representation ===")
    print(Image_)

    Camp_priority = compute_camp_priority(optimal_path, scale=10)

    print("\n=== Camp Priority ===")
    print(Camp_priority)



    # 5. Visualize
    final_img = visualize_path(s.output, optimal_path, final_score)
    cv.imshow("Final Optimal Path", final_img)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
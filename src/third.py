import cv2 as cv
import math
import first as f
import second as s


# ======================================================
# 1. Priority mappings
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


def priority_score(target):
    return (
        CASUALTY_PRIORITY[target["shape"]] *
        EMERGENCY_PRIORITY[target["color"]]
    )


# ======================================================
# 2. Parse detected objects
# ======================================================

def parse_objects(center_arr):
    sources = []
    targets = []

    for shape, color, x, y in center_arr:
        obj = {
            "shape": shape,
            "color": color,
            "center": (x, y)
        }

        if shape == "Circle":
            sources.append(obj)
        else:
            targets.append(obj)

    # Enforce camp order: Blue → Pink → Gray
    color_order = {"Blue": 0, "Pink": 1, "Gray": 2}
    sources.sort(key=lambda s: color_order[s["color"]])

    return sources, targets


# ======================================================
# 3. Sort casualties by numeric priority only
# ======================================================

def sort_targets_by_priority(targets):
    return sorted(
        targets,
        key=lambda t: -priority_score(t)
    )


# ======================================================
# 4. Normalize priority (your original formula)
# ======================================================

def normalize_priority(targets):
    max_p = max(priority_score(t) for t in targets)

    for t in targets:
        p = priority_score(t)
        t["priority_norm"] = (max_p - p + 1) / max_p


# ======================================================
# 5. Compute distances
# ======================================================

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


# ======================================================
# 6. Normalize distances per camp (UNCHANGED logic)
# ======================================================

def normalize_distances(distances, num_sources, num_targets):
    for i in range(num_sources):
        start = i * num_targets
        end = (i + 1) * num_targets

        min_d = min(x["distance"] for x in distances[start:end])
        max_d = max(x["distance"] for x in distances[start:end])

        for j in range(num_targets):
            a = distances[start + j]["distance"] - min_d
            b = max_d - min_d

            if b != 0:
                distances[start + j]["distance_norm"] = 1 - (a / b)
            else:
                distances[start + j]["distance_norm"] = 0


# ======================================================
# 7. Score function (NEW, AS REQUESTED)
# ======================================================

def compute_score(p_norm, d_norm, alpha):
    """
    score = alpha * priority_norm + (1 - alpha) * distance_norm
    """
    return alpha * p_norm + (1 - alpha) * d_norm


def add_scores(distances, targets, alpha=0.7):
    """
    Adds final 'score' to each distance record.
    """

    # Lookup for priority_norm
    priority_lookup = {
        (t["shape"], t["color"], t["center"]): t["priority_norm"]
        for t in targets
    }

    for d in distances:
        key = (d["target_shape"], d["target_color"], d["target_center"])
        p_norm = priority_lookup[key]
        d_norm = d["distance_norm"]

        d["score"] = compute_score(p_norm, d_norm, alpha)


# ======================================================
# 8. PRINT HELPERS
# ======================================================

def print_sources(sources):
    print("\n=== RESCUE CAMPS ===")
    for i, s in enumerate(sources):
        print(f"{i} | {s['color']} camp at {s['center']}")


def print_targets(targets):
    print("\n=== TARGETS ===")
    for i, t in enumerate(targets):
        print(
            f"{i} | {t['shape']} ({t['color']}) at {t['center']} | "
            f"priority = {priority_score(t)} | "
            f"priority_norm = {round(t['priority_norm'], 3)}"
        )


def print_scores_by_camp(distances, num_targets):
    print("\n=== SCORES BY CAMP ===")

    for i in range(0, len(distances), num_targets):
        camp = distances[i]["camp_color"]
        print(f"\n--- {camp} camp ---")

        for d in distances[i:i + num_targets]:
            print(
                f"{d['target_shape']} ({d['target_color']}) "
                f"| d_norm = {round(d['distance_norm'], 3)} "
                f"| score = {round(d['score'], 3)}"
            )

def assign_targets_priority_wise(
    targets_sorted,
    distances,
    capacity
):
    """
    Assigns targets one-by-one in descending priority order.
    For each target:
      - compare scores across camps
      - choose highest-score camp with available capacity
    """

    final_assignments = []

    for tgt in targets_sorted:
        # collect all camp-score entries for this target
        candidates = []

        for d in distances:
            if (
                d["target_shape"] == tgt["shape"]
                and d["target_color"] == tgt["color"]
                and d["target_center"] == tgt["center"]
            ):
                candidates.append(d)

        # sort camps by score (descending)
        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        assigned = False

        for c in candidates:
            camp = c["camp_color"]

            if capacity[camp] > 0:
                final_assignments.append({
                    "camp_color": camp,
                    "target_shape": tgt["shape"],
                    "target_color": tgt["color"],
                    "target_center": tgt["center"],
                    "priority": priority_score(tgt),
                    "score": c["score"]
                })

                capacity[camp] -= 1
                assigned = True
                break

        if not assigned:
            print(
                f"WARNING: No capacity left for "
                f"{tgt['shape']} ({tgt['color']}) at {tgt['center']}"
            )

    return final_assignments


def build_camp_lookup(sources):
    """
    Returns:
    {
        "Blue": (x, y),
        "Pink": (x, y),
        "Gray": (x, y)
    }
    """
    return {s["color"]: s["center"] for s in sources}

ARROW_COLORS = {
    "Blue": (0, 0, 255),     # Blue (BGR)
    "Pink": (0, 0, 255), # Pink
    "Gray": (0, 0, 255)  # Gray
}

def visualize_assignments(output_img, final_assignments, camp_lookup):
    vis = output_img.copy()

    for a in final_assignments:
        camp_color = a["camp_color"]
        camp_center = camp_lookup[camp_color]
        target_center = a["target_center"]

        color = ARROW_COLORS.get(camp_color, (255, 255, 255))

        # Draw arrow: camp -> target
        cv.arrowedLine(
            vis,
            camp_center,
            target_center,
            color,
            thickness=2,
            tipLength=0.03
        )

        # Draw target point
        cv.circle(vis, target_center, 5, color, -1)

        # Optional label
        label = f"{camp_color}"
        cv.putText(
            vis,
            label,
            (target_center[0] + 5, target_center[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    return vis


# ======================================================
# 9. MAIN
# ======================================================

if __name__ == "__main__":

    # Parse detected objects
    sources, targets = parse_objects(f.center_arr)

    # Sort targets by priority
    targets_sorted = sort_targets_by_priority(targets)

    # Normalize priority
    normalize_priority(targets_sorted)

    # Compute distances
    distances = compute_distances(sources, targets_sorted)

    # Normalize distances
    normalize_distances(
        distances,
        num_sources=len(sources),
        num_targets=len(targets_sorted)
    )

    # Add final score
    alpha = 0.7
    add_scores(distances, targets_sorted, alpha)

    capacity = {
    "Blue": 4,
    "Pink": 3,
    "Gray": 2
}

    final_assignments = assign_targets_priority_wise(
        targets_sorted,
        distances,
        capacity
    )

    print("\n=== FINAL ASSIGNMENTS ===")
    for a in final_assignments:
        print(
            f"{a['camp_color']} camp -> "
            f"{a['target_shape']} ({a['target_color']}) at {a['target_center']} "
            f"| priority = {a['priority']} | score = {round(a['score'], 3)}"
        )

# -----------------------------
# Display
# -----------------------------
# Build camp lookup
    camp_lookup = build_camp_lookup(sources)

# Visualize on land/ocean output from second module
    vis_img = visualize_assignments(
        s.output,              # from second module
        final_assignments,
        camp_lookup
    )

# Show / save
    cv.imshow("Final Assignments", vis_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
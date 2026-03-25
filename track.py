import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import os

INPUT_VIDEO  = "Come_check_out_the_fastest_way_down_the_mountain._bungee_720P.mp4"
OUTPUT_VIDEO = "tracked_output.mp4"
OUTPUT_PLOT   = "tracking_analysis.png"

# ----------------------------
# Tunable parameters
# ----------------------------
BOOT_FRAMES        = 20      # first pass to identify the right target
SEARCH_TOP_RATIO   = 0.78    # bootstrap search only in upper part of frame
MIN_BOOT_AREA      = 80
MAX_BOOT_AREA      = 1200

MIN_TRACK_AREA     = 20
MAX_TRACK_AREA     = 2000

REDETECT_EVERY     = 6       # periodic correction
MAX_MISSES         = 8
LOCAL_PAD_RATIO    = 5.0     # search window size relative to bbox size
LOCAL_PAD_MIN      = 120
GLOBAL_PAD_MIN     = 200

TRACK_GATE_FACTOR   = 4.0     # how far tracker can jump before rejecting
COLOR_H_MIN_1      = 0
COLOR_H_MAX_1      = 18
COLOR_H_MIN_2      = 165
COLOR_H_MAX_2      = 180
COLOR_S_MIN        = 70
COLOR_V_MIN        = 70

os.makedirs(".", exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def create_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker is not available in this OpenCV build.")

def bbox_center(b):
    x, y, w, h = b
    return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)

def bbox_area(b):
    return max(0, int(b[2])) * max(0, int(b[3]))

def bbox_to_int(b):
    return tuple(int(v) for v in b)

def clamp_bbox(b, W, H):
    x, y, w, h = b
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return (x, y, w, h)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def draw_trail(frame, trail, color=(0, 255, 255)):
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, pts[i - 1], pts[i], c, 2)

def speed_color(speed_px):
    norm = min(speed_px / 40.0, 1.0)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    return (0, g, r)

def warm_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(
        hsv,
        (COLOR_H_MIN_1, COLOR_S_MIN, COLOR_V_MIN),
        (COLOR_H_MAX_1, 255, 255)
    )
    m2 = cv2.inRange(
        hsv,
        (COLOR_H_MIN_2, COLOR_S_MIN, COLOR_V_MIN),
        (COLOR_H_MAX_2, 255, 255)
    )
    mask = cv2.bitwise_or(m1, m2)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask

def motion_mask(prev_gray, gray):
    if prev_gray is None:
        return np.zeros_like(gray, dtype=np.uint8)

    diff = cv2.absdiff(prev_gray, gray)
    _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    th = cv2.dilate(th, None, iterations=1)
    return th

def extract_candidates(frame, roi=None, scale=2.0, prev_gray=None):
    """
    Returns candidate boxes in original-frame coordinates:
    [(x, y, w, h, score, cx, cy), ...]
    Score is only a rough filter; final choice is done later.
    """
    H, W = frame.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))
        crop = frame[y1:y2, x1:x2]
        base_x, base_y = x1, y1
    else:
        crop = frame
        base_x, base_y = 0, 0

    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    mask = warm_mask(crop)

    if prev_gray is not None:
        if roi is not None:
            prev_crop = prev_gray[y1:y2, x1:x2]
        else:
            prev_crop = prev_gray

        if scale != 1.0:
            prev_crop = cv2.resize(prev_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mot = motion_mask(prev_crop, gray)
    else:
        mot = np.zeros(mask.shape, dtype=np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    inv_scale = 1.0 / scale

    for c in contours:
        area_scaled = cv2.contourArea(c)
        if area_scaled < 10:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # map back to original coordinates
        ox = int(x * inv_scale) + base_x
        oy = int(y * inv_scale) + base_y
        ow = max(1, int(w * inv_scale))
        oh = max(1, int(h * inv_scale))

        if ox < 0 or oy < 0 or ox + ow > W or oy + oh > H:
            continue

        area = ow * oh
        if area < MIN_TRACK_AREA or area > MAX_TRACK_AREA:
            continue

        # compute simple color + motion ratios
        m_region = mask[y:y + h, x:x + w]
        mo_region = mot[y:y + h, x:x + w]

        warm_ratio = float(np.count_nonzero(m_region)) / float(max(1, m_region.size))
        motion_ratio = float(np.count_nonzero(mo_region)) / float(max(1, mo_region.size))

        cx = ox + ow / 2.0
        cy = oy + oh / 2.0

        # rough candidate score
        score = (4.0 * warm_ratio) + (2.5 * motion_ratio) + (0.0015 * area)
        candidates.append((ox, oy, ow, oh, score, cx, cy))

    return candidates

def choose_best_candidate(candidates, predicted_center, last_area):
    if not candidates:
        return None

    best = None
    best_score = -1e18

    for x, y, w, h, base_score, cx, cy in candidates:
        area = w * h
        c = np.array([cx, cy], dtype=np.float32)

        if predicted_center is not None:
            d = dist(c, predicted_center)
        else:
            d = 0.0

        # area continuity helps avoid switching to platform people/signs
        if last_area is not None:
            area_ratio = area / float(max(1, last_area))
            size_penalty = abs(np.log(max(1e-6, area_ratio)))
        else:
            size_penalty = 0.0

        # lower is better for distance and size penalty
        final = base_score - (0.02 * d) - (0.6 * size_penalty)

        if final > best_score:
            best_score = final
            best = (x, y, w, h)

    return best

# ----------------------------
# Bootstrap: identify the correct red/orange moving target
# ----------------------------
def bootstrap_target_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, first = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read video.")

    H, W = first.shape[:2]
    top_limit = int(H * SEARCH_TOP_RATIO)

    # tracklets for the first few frames
    tracklets = []
    frame_idx = 0
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    while frame_idx < BOOT_FRAMES:
        if frame_idx == 0:
            frame = first
        else:
            ret, frame = cap.read()
            if not ret:
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = (0, 0, W, top_limit)

        candidates = extract_candidates(frame, roi=roi, scale=2.0, prev_gray=prev_gray)

        # Keep only plausible bootstrap candidates:
        # - moderate area
        # - not too low in the frame
        filtered = []
        for cand in candidates:
            x, y, w, h, score, cx, cy = cand
            area = w * h
            if area < MIN_BOOT_AREA or area > MAX_BOOT_AREA:
                continue
            if cy > top_limit:
                continue
            filtered.append(cand)

        # Greedy association by center proximity and area continuity
        assigned = set()
        for tr in tracklets:
            tr["matched"] = False

        for cand in sorted(filtered, key=lambda t: t[4], reverse=True):
            x, y, w, h, score, cx, cy = cand
            c = np.array([cx, cy], dtype=np.float32)

            best_tr = None
            best_d = 1e18
            for tr in tracklets:
                if tr["matched"]:
                    continue
                last = np.array(tr["history"][-1]["center"], dtype=np.float32)
                d = dist(c, last)
                if d < best_d:
                    best_d = d
                    best_tr = tr

            # a little loose in bootstrapping, but not too loose
            if best_tr is not None and best_d < 90:
                best_tr["history"].append({
                    "frame": frame_idx,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "area": w * h
                })
                best_tr["matched"] = True
                assigned.add((x, y, w, h))
            else:
                tracklets.append({
                    "id": len(tracklets),
                    "history": [{
                        "frame": frame_idx,
                        "bbox": (x, y, w, h),
                        "center": (cx, cy),
                        "area": w * h
                    }],
                    "matched": True
                })
                assigned.add((x, y, w, h))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not tracklets:
        # fallback: choose a top-half warm blob from first frame
        candidates = extract_candidates(first, roi=(0, 0, W, top_limit), scale=2.0, prev_gray=None)
        if not candidates:
            raise RuntimeError("No candidate found for initialization.")
        # choose the most moderate-sized candidate
        candidates = [c for c in candidates if MIN_BOOT_AREA <= c[2] * c[3] <= MAX_BOOT_AREA]
        if not candidates:
            candidates = extract_candidates(first, roi=(0, 0, W, top_limit), scale=2.0, prev_gray=None)
        best = max(candidates, key=lambda c: c[4])
        return bbox_to_int(best[:4])

    # score tracklets: prefer persistent, moving, moderate-size targets
    def score_tracklet(tr):
        hist = tr["history"]
        if len(hist) < 3:
            return -1e9

        centers = np.array([h["center"] for h in hist], dtype=np.float32)
        areas = np.array([h["area"] for h in hist], dtype=np.float32)

        displacement = float(np.linalg.norm(centers[-1] - centers[0]))
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        mean_step = float(np.mean(steps)) if len(steps) else 0.0
        jitter = float(np.std(steps)) if len(steps) else 0.0
        mean_area = float(np.mean(areas))
        area_penalty = 0.0

        # the true target in this clip is not a huge platform blob and not a tiny static sign
        if mean_area < 80:
            area_penalty += 3.0
        if mean_area > 900:
            area_penalty += 2.5

        # persistent motion matters more than raw size
        return (2.2 * len(hist)) + (0.08 * displacement) + (0.35 * mean_step) - (0.12 * jitter) - area_penalty

    best_tracklet = max(tracklets, key=score_tracklet)
    init_bbox = best_tracklet["history"][0]["bbox"]
    return bbox_to_int(init_bbox)

# ----------------------------
# Tracking with local/global re-detection
# ----------------------------
def run_tracker(video_path, init_bbox):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

    ret, first = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise RuntimeError("Could not read the first frame.")

    tracker = create_csrt()
    tracker.init(first, init_bbox)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trail = deque(maxlen=80)
    positions = []
    speeds_px = []

    prev_center = None
    prev_gray = None
    last_good_bbox = init_bbox
    last_area = bbox_area(init_bbox)
    velocity = np.array([0.0, 0.0], dtype=np.float32)
    misses = 0
    frame_idx = 0

    print(f"Tracking {total} frames ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # predicted center from simple velocity model
        if last_good_bbox is not None:
            last_c = bbox_center(last_good_bbox)
            predicted_center = last_c + velocity
        else:
            predicted_center = None

        ok, raw_bbox = tracker.update(frame)
        current_bbox = None

        if ok:
            raw_bbox = clamp_bbox(raw_bbox, W, H)
            curr_center = bbox_center(raw_bbox)

            if predicted_center is not None:
                gate = max(60.0, TRACK_GATE_FACTOR * np.sqrt(max(1.0, last_area)))
                if dist(curr_center, predicted_center) > gate:
                    ok = False
            if ok:
                current_bbox = raw_bbox

        # periodic correction or failure recovery
        need_redetect = (not ok) or (frame_idx % REDETECT_EVERY == 0)

        if need_redetect:
            # local search near predicted location first
            search_bbox = None
            if predicted_center is not None and last_good_bbox is not None:
                lx, ly, lw, lh = last_good_bbox
                pad = int(max(LOCAL_PAD_MIN, LOCAL_PAD_RATIO * np.sqrt(max(1.0, lw * lh))))
                px, py = int(predicted_center[0]), int(predicted_center[1])

                x1 = max(0, px - pad)
                y1 = max(0, py - pad)
                x2 = min(W, px + pad)
                y2 = min(H, py + pad)
                search_bbox = (x1, y1, x2, y2)

            candidates = extract_candidates(frame, roi=search_bbox, scale=2.0, prev_gray=prev_gray)
            best = choose_best_candidate(candidates, predicted_center, last_area)

            # if local search fails, do a global search as fallback
            if best is None:
                candidates = extract_candidates(frame, roi=None, scale=2.0, prev_gray=prev_gray)
                best = choose_best_candidate(candidates, predicted_center, last_area)

            if best is not None:
                tracker = create_csrt()
                tracker.init(frame, bbox_to_int(best))
                current_bbox = bbox_to_int(best)
                ok = True
                misses = 0
            else:
                misses += 1

        if ok and current_bbox is not None:
            x, y, w, h = current_bbox
            cx, cy = int(x + w / 2), int(y + h / 2)

            # velocity update
            if prev_center is not None:
                step = np.array([cx - prev_center[0], cy - prev_center[1]], dtype=np.float32)
                velocity = 0.80 * velocity + 0.20 * step
                speed = float(np.linalg.norm(step))
            else:
                speed = 0.0

            prev_center = (cx, cy)
            last_good_bbox = current_bbox
            last_area = w * h

            trail.append((cx, cy))
            positions.append((frame_idx, cx, cy))
            speeds_px.append(speed)

            draw_trail(frame, trail)
            col = speed_color(speed)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"spd:{speed:.1f}px/f", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        else:
            cv2.putText(frame, "RE-ACQUIRING TARGET...", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)

        cv2.putText(frame, f"Frame {frame_idx:04d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time {frame_idx / fps:.2f}s", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        prev_gray = gray
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames processed")

        if misses > MAX_MISSES:
            # forced reset with global search if we lose the target for too long
            candidates = extract_candidates(frame, roi=None, scale=2.0, prev_gray=prev_gray)
            best = choose_best_candidate(candidates, predicted_center, last_area)
            if best is not None:
                tracker = create_csrt()
                tracker.init(frame, bbox_to_int(best))
                current_bbox = bbox_to_int(best)
                last_good_bbox = current_bbox
                last_area = bbox_area(current_bbox)
                misses = 0

    cap.release()
    writer.release()
    print(f"Saved tracked video -> {OUTPUT_VIDEO}")
    return positions, speeds_px, fps

# ----------------------------
# Analysis plots
# ----------------------------
def plot_analysis(positions, speeds_px, fps):
    if not positions:
        print("No positions to plot.")
        return

    frames, xs, ys = zip(*positions)
    times = [f / fps for f in frames]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Person Tracking Analysis – Mountain Descent", fontsize=15, fontweight="bold")

    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # 1. Trajectory
    sc = axes[0, 0].scatter(xs, ys, c=times, cmap="plasma", s=4, alpha=0.85)
    cb = fig.colorbar(sc, ax=axes[0, 0])
    cb.set_label("Time (s)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Trajectory (coloured by time)")
    axes[0, 0].set_xlabel("X pixel")
    axes[0, 0].set_ylabel("Y pixel")

    # 2. X over time
    axes[0, 1].plot(times, xs, color="#00d4ff", linewidth=1.2)
    axes[0, 1].set_title("Horizontal Position over Time")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("X (px)")

    # 3. Y over time
    axes[1, 0].plot(times, ys, color="#ff6b6b", linewidth=1.2)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Vertical Position over Time (↓ = down)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Y (px)")

    # 4. Speed over time
    sp_times = [frames[i] / fps for i in range(len(speeds_px))]
    axes[1, 1].plot(sp_times, speeds_px, color="#ffd700", linewidth=1.0, alpha=0.7)

    if len(speeds_px) > 10:
        kernel = np.ones(11) / 11
        smooth = np.convolve(speeds_px, kernel, mode="same")
        axes[1, 1].plot(sp_times, smooth, color="#ff4500", linewidth=2, label="Smoothed")
        axes[1, 1].legend(facecolor="#16213e", labelcolor="white")

    axes[1, 1].set_title("Pixel Speed per Frame")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Speed (px/frame)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved analysis plot -> {OUTPUT_PLOT}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("=== Bootstrap: finding the correct target ...")
    bbox = bootstrap_target_bbox(INPUT_VIDEO)
    print(f"Initial bounding box: {bbox}")

    print("=== Tracking with local/global re-detection ...")
    positions, speeds, fps = run_tracker(INPUT_VIDEO, bbox)

    print("=== Generating analysis plots ...")
    plot_analysis(positions, speeds, fps)

    if positions:
        frames, xs, ys = zip(*positions)
        avg_spd = float(np.mean(speeds)) if speeds else 0.0
        max_spd = float(np.max(speeds)) if speeds else 0.0

        print(f"\n{'='*45}")
        print(f"Tracked frames : {len(positions)}")
        print(f"Avg speed      : {avg_spd:.2f} px/frame")
        print(f"Peak speed     : {max_spd:.2f} px/frame")
        print(f"X range        : {min(xs)} – {max(xs)} px")
        print(f"Y range        : {min(ys)} – {max(ys)} px")
        print(f"{'='*45}")

    print("Done.")


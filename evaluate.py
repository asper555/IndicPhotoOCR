import os
import json
import numpy as np
from shapely.geometry import Polygon

# ---------------- CONFIG ----------------
GT_DIR = "val_data/gt"
PRED_DIR = "pred"
IOU_THRESHOLD = 0.5


# ---------------- SAFE POLYGON ----------------
def safe_polygon(poly):
    try:
        p = Polygon(poly)
        if not p.is_valid or p.area == 0:
            return None
        return p
    except:
        return None


# ---------------- IOU ----------------
def compute_iou(poly1, poly2):
    p1 = safe_polygon(poly1)
    p2 = safe_polygon(poly2)

    if p1 is None or p2 is None:
        return 0.0

    inter = p1.intersection(p2).area
    union = p1.union(p2).area

    if union == 0:
        return 0.0

    return inter / union


# ---------------- READ GT ----------------
def read_gt(path):
    polys = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return polys

    if "text instances" in data:
        data = data["text instances"]

    for item in data:
        if not isinstance(item, dict):
            continue

        if item.get("flag", 1) != 1:
            continue

        pts = item.get("points")
        if pts:
            polys.append(pts)

    return polys


# ---------------- READ PRED ----------------
def read_pred(path):
    polys = []

    if not os.path.exists(path):
        return polys

    with open(path, "r") as f:
        for line in f:
            try:
                coords = list(map(int, line.strip().split(",")))
                pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                polys.append(pts)
            except:
                continue

    return polys


# ---------------- IOU MATCHING ----------------
def eval_iou(gt, pred):
    matched_pred = set()
    tp = 0
    ious = []

    for g in gt:
        best_iou = 0
        best_j = -1

        for j, p in enumerate(pred):
            if j in matched_pred:
                continue

            iou = compute_iou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_pred.add(best_j)
            ious.append(best_iou)

    fp = len(pred) - tp
    fn = len(gt) - tp

    return tp, fp, fn, ious


# ---------------- TEDEVAL ----------------
def eval_tedeval(gt, pred):
    gt_used = [False] * len(gt)
    pred_used = [False] * len(pred)

    tp = 0

    for i, g in enumerate(gt):
        p1 = safe_polygon(g)
        if p1 is None:
            continue

        for j, p in enumerate(pred):
            p2 = safe_polygon(p)
            if p2 is None:
                continue

            inter = p1.intersection(p2).area
            if inter == 0:
                continue

            recall = inter / p1.area
            precision = inter / p2.area

            if recall >= 0.5 or precision >= 0.5:
                if not gt_used[i] and not pred_used[j]:
                    tp += 1
                    gt_used[i] = True
                    pred_used[j] = True

    fp = len(pred) - tp
    fn = len(gt) - tp

    return tp, fp, fn


# ---------------- MAIN ----------------
def main():

    total_tp = total_fp = total_fn = 0
    all_ious = []

    t_tp = t_fp = t_fn = 0

    for file in os.listdir(GT_DIR):

        if not file.endswith(".json"):
            continue

        gt_path = os.path.join(GT_DIR, file)
        pred_path = os.path.join(PRED_DIR, file.replace(".json", ".txt"))

        gt = read_gt(gt_path)
        pred = read_pred(pred_path)

        # IoU metrics
        tp, fp, fn, ious = eval_iou(gt, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        # TEDEval
        tp2, fp2, fn2 = eval_tedeval(gt, pred)
        t_tp += tp2
        t_fp += fp2
        t_fn += fn2

    # ---------------- FINAL ----------------
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    avg_iou = np.mean(all_ious) if all_ious else 0

    t_precision = t_tp / (t_tp + t_fp + 1e-6)
    t_recall = t_tp / (t_tp + t_fn + 1e-6)
    t_f1 = 2 * t_precision * t_recall / (t_precision + t_recall + 1e-6)

    print("\n========== FINAL RESULTS ==========")

    print("\nIoU Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg IoU:   {avg_iou:.4f}")

    print("\nTEDEval Metrics:")
    print(f"Precision: {t_precision:.4f}")
    print(f"Recall:    {t_recall:.4f}")
    print(f"F1 Score:  {t_f1:.4f}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
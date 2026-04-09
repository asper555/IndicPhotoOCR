import os
import json
import numpy as np
from shapely.geometry import Polygon

# ---------------- CONFIG ----------------
GT_DIR = "val_data/gt"
PRED_DIR = "pred"
IOU_THRESHOLD = 0.5


# ---------------- IOU FUNCTION ----------------
def compute_iou(poly1, poly2):
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)

        if not p1.is_valid or not p2.is_valid:
            return 0.0

        inter = p1.intersection(p2).area
        union = p1.union(p2).area

        if union == 0:
            return 0.0

        return inter / union

    except:
        return 0.0


# ---------------- READ GT ----------------
def read_gt(json_path):
    polys = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🔥 YOUR FORMAT
    if "text instances" in data:
        data = data["text instances"]

    for word in data:
        if not isinstance(word, dict):
            continue

        # skip ignored text
        if word.get("flag", 1) != 1:
            continue

        points = word.get("points")

        if points:
            polys.append(points)

    return polys


# ---------------- READ PRED ----------------
def read_pred(txt_path):
    polys = []

    with open(txt_path, "r") as f:
        for line in f:
            coords = list(map(int, line.strip().split(",")))
            pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polys.append(pts)

    return polys


# ---------------- MATCHING ----------------
def evaluate_image(gt_polys, pred_polys):

    matched_gt = set()
    matched_pred = set()

    iou_scores = []

    for i, gt in enumerate(gt_polys):
        best_iou = 0
        best_j = -1

        for j, pred in enumerate(pred_polys):
            if j in matched_pred:
                continue

            iou = compute_iou(gt, pred)

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(i)
            matched_pred.add(best_j)
            iou_scores.append(best_iou)

    TP = len(matched_gt)
    FP = len(pred_polys) - TP
    FN = len(gt_polys) - TP

    return TP, FP, FN, iou_scores


# ---------------- MAIN ----------------
total_TP = total_FP = total_FN = 0
all_ious = []

print("\n===== DEBUG MODE =====")

for file in os.listdir(GT_DIR):

    if not file.endswith(".json"):
        continue

    gt_path = os.path.join(GT_DIR, file)
    pred_path = os.path.join(PRED_DIR, file.replace(".json", ".txt"))

    if not os.path.exists(pred_path):
        continue

    gt_polys = read_gt(gt_path)
    pred_polys = read_pred(pred_path)

    print(f"\nFile: {file}")
    print("GT count:", len(gt_polys))
    print("Pred count:", len(pred_polys))

    if len(gt_polys) > 0 and len(pred_polys) > 0:
        print("Sample GT polygon:", gt_polys[0])
        print("Sample Pred polygon:", pred_polys[0])

        print("\n--- Sample IoU values ---")
        for i in range(min(3, len(gt_polys))):
            for j in range(min(3, len(pred_polys))):
                iou = compute_iou(gt_polys[i], pred_polys[j])
                print(f"GT[{i}] vs Pred[{j}] = {iou:.3f}")

    TP, FP, FN, ious = evaluate_image(gt_polys, pred_polys)

    total_TP += TP
    total_FP += FP
    total_FN += FN
    all_ious.extend(ious)

print("\n===== END DEBUG =====")

# ---------------- FINAL METRICS ----------------
precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)
avg_iou = np.mean(all_ious) if all_ious else 0

print("\n===== FINAL RESULTS =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Avg IoU:   {avg_iou:.4f}")
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


# ---------------- STANDARD EVAL ----------------
def eval_standard(gt, pred):
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

            # 🔥 TEDEval condition
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

    if not os.path.exists(GT_DIR):
        print("❌ GT folder not found:", GT_DIR)
        return

    if not os.path.exists(PRED_DIR):
        print("❌ Pred folder not found:", PRED_DIR)
        return

    total_tp = total_fp = total_fn = 0
    all_ious = []

    t_tp = t_fp = t_fn = 0

    files = os.listdir(GT_DIR)

    for file in files:
        if not file.endswith(".json"):
            continue

        gt_path = os.path.join(GT_DIR, file)
        pred_path = os.path.join(PRED_DIR, file.replace(".json", ".txt"))

        gt = read_gt(gt_path)
        pred = read_pred(pred_path)

        # Standard metrics
        tp, fp, fn, ious = eval_standard(gt, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        # TEDEval
        tp2, fp2, fn2 = eval_tedeval(gt, pred)
        t_tp += tp2
        t_fp += fp2
        t_fn += fn2

    # ---------------- RESULTS ----------------
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    avg_iou = np.mean(all_ious) if all_ious else 0

    t_precision = t_tp / (t_tp + t_fp + 1e-6)
    t_recall = t_tp / (t_tp + t_fn + 1e-6)
    t_f1 = 2 * t_precision * t_recall / (t_precision + t_recall + 1e-6)

    print("\n===== FINAL RESULTS =====")

    print("\n--- Standard (IoU) ---")
    print("Precision:", round(precision, 4))
    print("Recall:   ", round(recall, 4))
    print("F1 Score: ", round(f1, 4))
    print("Avg IoU:  ", round(avg_iou, 4))

    print("\n--- TEDEval ---")
    print("Precision:", round(t_precision, 4))
    print("Recall:   ", round(t_recall, 4))
    print("F1 Score: ", round(t_f1, 4))


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
"""픽셀 좌표(xyxy absolute) 기반 greedy Weighted Box Fusion."""

import numpy as np


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and an array of boxes, all xyxy pixel coords."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_box  = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-9
    return inter / union


def fuse_results(
    eo_out: dict,
    ir_out: dict,
    fusion_weight: float,
    iou_thr: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy WBF in pixel coordinates.

    Scores are pre-weighted by modality weight before clustering, then
    the final box is the score-weighted average of cluster members.

    Args:
        eo_out: {'boxes': (N,4) xyxy pixel, 'scores': (N,), 'labels': (N,)}
        ir_out: same format
        fusion_weight: EO weight w; IR weight = 1 - w
        iou_thr: IoU threshold for merging boxes into a cluster

    Returns:
        boxes (M,4), scores (M,), labels (M,)
    """
    w_eo = float(fusion_weight)
    w_ir = 1.0 - w_eo

    all_boxes  = []
    all_scores = []
    all_labels = []

    if len(eo_out["boxes"]):
        all_boxes.append(eo_out["boxes"])
        all_scores.append(eo_out["scores"] * w_eo)
        all_labels.append(eo_out["labels"])

    if len(ir_out["boxes"]):
        all_boxes.append(ir_out["boxes"])
        all_scores.append(ir_out["scores"] * w_ir)
        all_labels.append(ir_out["labels"])

    if not all_boxes:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    boxes  = np.concatenate(all_boxes,  axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)

    # Sort descending by weighted score
    order = np.argsort(-scores)
    boxes  = boxes[order]
    scores = scores[order]
    labels = labels[order]

    used = np.zeros(len(boxes), dtype=bool)
    out_boxes, out_scores, out_labels = [], [], []

    for i in range(len(boxes)):
        if used[i]:
            continue
        ious = _box_iou(boxes[i], boxes)
        cluster = np.where((ious >= iou_thr) & ~used)[0]
        # Include self
        cluster = np.union1d([i], cluster)
        used[cluster] = True

        cluster_scores = scores[cluster]
        cluster_boxes  = boxes[cluster]

        total_w = cluster_scores.sum() + 1e-9
        merged_box   = (cluster_boxes * cluster_scores[:, None]).sum(0) / total_w
        merged_score = cluster_scores.max()
        # Label from highest-scoring member
        merged_label = labels[cluster[np.argmax(cluster_scores)]]

        out_boxes.append(merged_box)
        out_scores.append(merged_score)
        out_labels.append(merged_label)

    if not out_boxes:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    return (
        np.array(out_boxes,  dtype=np.float32),
        np.array(out_scores, dtype=np.float32),
        np.array(out_labels, dtype=np.float32),
    )

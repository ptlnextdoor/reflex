"""Shared eval math: IoU, edge error at source res, classification accuracy."""

import numpy as np

import model as M


def denorm_box(box_norm):
    """[0,1] x0,y0,x1,y1 -> source-res px (512x288)."""
    scale = np.array([M.SRC_W, M.SRC_H, M.SRC_W, M.SRC_H], dtype=np.float32)
    return box_norm * scale


def iou(a, b):
    """Elementwise IoU over batches of boxes in px. a,b: (N,4) x0y0x1y1."""
    x0 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 2], b[:, 2])
    y1 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def edge_err_px(pred_src, gt_src):
    """Mean absolute error per edge in source px. Returns (N,) mean over 4 edges."""
    return np.abs(pred_src - gt_src).mean(axis=1)


def box_metrics(pred_norm, gt_src):
    pred_src = denorm_box(pred_norm)
    ious = iou(pred_src, gt_src)
    edge = edge_err_px(pred_src, gt_src)
    return dict(
        iou_mean=float(ious.mean()),
        iou_med=float(np.median(ious)),
        iou_at70=float((ious > 0.7).mean()),
        edge_mean=float(edge.mean()),
        edge_med=float(np.median(edge)),
        edge_p99=float(np.percentile(edge, 99)),
    )

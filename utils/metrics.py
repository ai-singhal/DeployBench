"""
COCO prediction formatting and mAP computation using pycocotools.
"""

import contextlib
import io


def format_predictions(ultralytics_results, image_id: int, coco_cat_id_map: dict) -> list:
    """
    Convert ultralytics Results to COCO prediction format.

    Args:
        ultralytics_results: list of Results from model(image_path)
        image_id: COCO image ID (int)
        coco_cat_id_map: dict mapping class_name -> COCO category_id

    Returns:
        list of COCO prediction dicts with keys:
            image_id, category_id, bbox ([x, y, w, h]), score
    """
    predictions = []
    result = ultralytics_results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return predictions

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names

    for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        class_name = class_names[cls_id]
        coco_cat_id = coco_cat_id_map.get(class_name)
        if coco_cat_id is None:
            continue

        predictions.append({
            "image_id": image_id,
            "category_id": coco_cat_id,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score),
        })

    return predictions


def compute_mAP(predictions: list, ground_truth: dict) -> float:
    """
    Compute mAP@50 using pycocotools.

    Args:
        predictions: list of COCO format prediction dicts
        ground_truth: COCO annotations dict (with "images", "annotations", "categories")

    Returns:
        mAP@50 as a float (0.0–1.0); returns 0.0 if no predictions
    """
    if not predictions:
        return 0.0

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build COCO ground truth object from the loaded dict
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    # Load predictions into COCO format
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)

    # Run evaluation (suppress verbose output from pycocotools)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # stats[1] is AP @ IoU=0.50 (mAP@50, Pascal VOC metric)
    return float(coco_eval.stats[1])

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


@dataclass
class DetectionResult:
    detections: Dict[str, List[dict]]   # {"houses":[...], "obstacles":[...]}
    annotated: Image.Image              # image with boxes drawn


def load_yolo(model_path: str | Path) -> YOLO:
    return YOLO(str(model_path))


def detect_objects(
    model: YOLO,
    pil_img: Image.Image,
    conf: float = 0.25,
    class_map: Optional[dict] = None,
) -> DetectionResult:
    """
    Returns:
      detections["houses"] = [{x1,y1,x2,y2,conf}, ...]
      detections["obstacles"] = [{x1,y1,x2,y2,conf}, ...]
    """
    if class_map is None:
        class_map = {"house": "house", "obstacle": "obstacle"}

    img_rgb = pil_img.convert("RGB")
    img_np = np.array(img_rgb)

    results = model.predict(img_np, conf=conf, verbose=False)
    r = results[0]

    detections = {"houses": [], "obstacles": []}
    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)

    if r.boxes is None:
        return DetectionResult(detections=detections, annotated=annotated)

    names = r.names  # class_id -> class_name

    for b in r.boxes:
        cls_id = int(b.cls.item())
        cls_name = str(names.get(cls_id, cls_id)).lower()
        conf_v = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())

        mapped = class_map.get(cls_name, "obstacle")
        item = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf_v}

        if mapped == "house":
            detections["houses"].append(item)
            color = (0, 180, 255)   # cyan
        else:
            detections["obstacles"].append(item)
            color = (255, 100, 100) # red

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    return DetectionResult(detections=detections, annotated=annotated)


def count_summary(detections: Dict[str, List[dict]]) -> Dict[str, int]:
    return {
        "total": len(detections.get("houses", [])) + len(detections.get("obstacles", [])),
        "houses": len(detections.get("houses", [])),
        "obstacles": len(detections.get("obstacles", [])),
    }
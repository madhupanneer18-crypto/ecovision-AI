"""
EcoVision AI - Inference Engine
Deep Learning Waste Classification & Multi-Object Detection Pipeline.
Combines YOLOv8 and PyTorch CNN for:
1. Whole-image waste classification across 6 categories:
   ['Plastic', 'Glass', 'Metal', 'Paper', 'Cardboard', 'Trash']
2. Multi-object waste detection with bounding boxes across 5 categories:
   ['Glass', 'Metal', 'Paper', 'Plastic', 'Residual']
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO

from src.config import (
    MODELS_DIR,
    CLASSIFICATION_CLASSES,
    DETECTION_CLASSES,
    CLASS_COLORS,
    WASTE_INFO
)


class EcoVisionEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.det_model_path = MODELS_DIR / "yolov8n.pt"
        self.waste_weights_path = MODELS_DIR / "waste_classifier.pth"
        
        self.det_model = None
        self.cls_model = None
        self.cls_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._load_models()

    def _load_models(self):
        """Loads YOLO detector and PyTorch waste classifier."""
        # 1. Load YOLOv8 Detector
        try:
            if self.det_model_path.exists():
                self.det_model = YOLO(str(self.det_model_path))
        except Exception as e:
            print(f"Detector load warning: {e}")

        # 2. Load PyTorch Waste Classifier
        try:
            if self.waste_weights_path.exists():
                ckpt = torch.load(self.waste_weights_path, map_location=self.device, weights_only=False)
                self.cls_classes = ckpt.get("classes", CLASSIFICATION_CLASSES)
                
                model = models.mobilenet_v2()
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.last_channel, 128),
                    nn.ReLU(),
                    nn.Linear(128, len(self.cls_classes))
                )
                model.load_state_dict(ckpt["state_dict"])
                model.to(self.device)
                model.eval()
                self.cls_model = model
            else:
                self.cls_classes = CLASSIFICATION_CLASSES
        except Exception as e:
            print(f"Classifier load warning: {e}")

    def _predict_crop(self, pil_crop):
        """Runs the PyTorch classifier on a PIL crop."""
        if self.cls_model is None:
            # Fallback uniform
            return {c: 1.0 / len(CLASSIFICATION_CLASSES) for c in CLASSIFICATION_CLASSES}

        try:
            tensor = self.cls_transform(pil_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.cls_model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            return {self.cls_classes[i]: float(probs[i]) for i in range(len(self.cls_classes))}
        except Exception as e:
            return {c: 1.0 / len(CLASSIFICATION_CLASSES) for c in CLASSIFICATION_CLASSES}

    # -------------------------------------------------------------
    # 1. Whole-Image Waste Classification
    # -------------------------------------------------------------
    def classify(self, image_input):
        """
        Classifies a whole waste image across the 6 categories:
        ['Plastic', 'Glass', 'Metal', 'Paper', 'Cardboard', 'Trash']
        """
        if isinstance(image_input, (str, os.PathLike)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        raw_probs = self._predict_crop(pil_img)
        
        # Ensure all 6 classification classes exist
        probs = {}
        for c in CLASSIFICATION_CLASSES:
            probs[c] = round(raw_probs.get(c, 0.001), 4)

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total, 4) for k, v in probs.items()}

        # Find top class
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primary_class, primary_conf = sorted_probs[0]

        # Environmental & recycling data
        info = WASTE_INFO.get(primary_class, WASTE_INFO["Trash"])

        return {
            "class": primary_class,
            "confidence": primary_conf,
            "probabilities": probs,
            "info": info
        }

    # -------------------------------------------------------------
    # 2. Multi-Object Waste Detection
    # -------------------------------------------------------------
    def detect(self, image_input, conf_threshold=0.25, iou_threshold=0.45):
        """
        Detects multiple waste objects and bounding boxes across:
        ['Glass', 'Metal', 'Paper', 'Plastic', 'Residual']
        """
        if isinstance(image_input, (str, os.PathLike)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        w, h = pil_img.size
        img_np = np.array(pil_img)
        detections = []

        # 1. Run YOLO object detector proposals
        if self.det_model is not None:
            try:
                results = self.det_model.predict(
                    pil_img,
                    conf=max(0.12, conf_threshold * 0.70),
                    iou=iou_threshold,
                    verbose=False
                )[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    box_w = x2 - x1
                    box_h = y2 - y1
                    if box_w < 20 or box_h < 20:
                        continue

                    # Crop object region
                    crop = pil_img.crop((x1, y1, x2, y2))
                    cls_probs = self._predict_crop(crop)
                    
                    # Map 6 classes to 5 detection classes:
                    # Glass -> Glass, Metal -> Metal, Paper/Cardboard -> Paper, Plastic -> Plastic, Trash -> Residual
                    det_scores = {
                        "Glass": cls_probs.get("Glass", 0.0),
                        "Metal": cls_probs.get("Metal", 0.0),
                        "Paper": cls_probs.get("Paper", 0.0) + cls_probs.get("Cardboard", 0.0),
                        "Plastic": cls_probs.get("Plastic", 0.0),
                        "Residual": cls_probs.get("Trash", 0.0)
                    }

                    best_det_class = max(det_scores.items(), key=lambda x: x[1])[0]
                    det_conf = det_scores[best_det_class]

                    if det_conf >= conf_threshold:
                        detections.append({
                            "class": best_det_class,
                            "confidence": round(float(det_conf), 3),
                            "box": [x1, y1, x2, y2],
                            "area": box_w * box_h,
                            "guidance": WASTE_INFO.get(best_det_class, WASTE_INFO["Residual"])["guidance"]
                        })
            except Exception as e:
                print(f"YOLO detection error: {e}")

        # 2. Saliency Contour fallback if no boxes found
        if len(detections) == 0:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in sorted_contours[:3]:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw > 35 and ch > 35 and (cw * ch) > (w * h * 0.05):
                    crop = pil_img.crop((cx, cy, cx + cw, cy + ch))
                    cls_probs = self._predict_crop(crop)
                    det_scores = {
                        "Glass": cls_probs.get("Glass", 0.0),
                        "Metal": cls_probs.get("Metal", 0.0),
                        "Paper": cls_probs.get("Paper", 0.0) + cls_probs.get("Cardboard", 0.0),
                        "Plastic": cls_probs.get("Plastic", 0.0),
                        "Residual": cls_probs.get("Trash", 0.0)
                    }
                    best_class, best_score = max(det_scores.items(), key=lambda x: x[1])
                    if best_score >= conf_threshold:
                        detections.append({
                            "class": best_class,
                            "confidence": round(float(best_score), 3),
                            "box": [cx, cy, cx + cw, cy + ch],
                            "area": cw * ch,
                            "guidance": WASTE_INFO.get(best_class, WASTE_INFO["Residual"])["guidance"]
                        })

        # Calculate counts
        counts = {c: 0 for c in DETECTION_CLASSES}
        for d in detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1

        annotated_image = self.draw_bounding_boxes(pil_img, detections)

        return {
            "annotated_image": annotated_image,
            "detections": detections,
            "counts": counts,
            "total_detected": len(detections)
        }

    def draw_bounding_boxes(self, pil_image, detections):
        """Draws sleek HUD bounding boxes with semi-transparent tint and badges."""
        img_np = np.array(pil_image.convert("RGB"))
        overlay = img_np.copy()
        h, w = img_np.shape[:2]

        for det in detections:
            cls_name = det["class"]
            x1, y1, x2, y2 = det["box"]
            color_cfg = CLASS_COLORS.get(cls_name, CLASS_COLORS["Residual"])
            rgb = color_cfg["rgb"]

            # Tint fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), rgb, -1)
            # Box border
            cv2.rectangle(img_np, (x1, y1), (x2, y2), rgb, 3)

            # Modern HUD corner brackets
            clen = min(22, max(8, int(min(x2 - x1, y2 - y1) * 0.2)))
            thick = 4
            cv2.line(img_np, (x1, y1), (x1 + clen, y1), rgb, thick)
            cv2.line(img_np, (x1, y1), (x1, y1 + clen), rgb, thick)
            cv2.line(img_np, (x2, y1), (x2 - clen, y1), rgb, thick)
            cv2.line(img_np, (x2, y1), (x2, y1 + clen), rgb, thick)
            cv2.line(img_np, (x1, y2), (x1 + clen, y2), rgb, thick)
            cv2.line(img_np, (x1, y2), (x1, y2 - clen), rgb, thick)
            cv2.line(img_np, (x2, y2), (x2 - clen, y2), rgb, thick)
            cv2.line(img_np, (x2, y2), (x2, y2 - clen), rgb, thick)

        # Alpha blend (0.16 tint)
        cv2.addWeighted(overlay, 0.16, img_np, 0.84, 0, img_np)

        annotated_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(annotated_pil)

        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, int(w * 0.022)))
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            cls_name = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["box"]
            color_cfg = CLASS_COLORS.get(cls_name, CLASS_COLORS["Residual"])
            rgb = color_cfg["rgb"]
            icon = color_cfg.get("icon", "")

            label = f" {icon} {cls_name} {int(conf * 100)}% "
            try:
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                tw = right - left
                th = bottom - top
            except Exception:
                tw = len(label) * 9
                th = 18

            by1 = max(0, y1 - th - 10)
            by2 = by1 + th + 8
            bx2 = min(w, x1 + tw + 12)

            draw.rectangle([x1, by1, bx2, by2], fill=rgb)
            draw.text((x1 + 4, by1 + 3), label, fill=(255, 255, 255), font=font)

        return annotated_pil


# Singleton instance
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = EcoVisionEngine()
    return _engine

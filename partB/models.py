# models.py
from ultralytics import YOLO
import torch
from PIL import Image
import torchvision.transforms as T

# ------------------- YOLOv8 ------------------- #
class YOLODetector:
    def __init__(self, weights="yolov8n.pt"):
        self.model = YOLO(weights)

    def predict(self, image_path):
        """
        Run YOLO inference on a single image
        Returns list of dicts: [{'bbox': [x1,y1,x2,y2], 'score': float, 'label': int}]
        """
        results = self.model(image_path, conf=0.25, iou=0.5)[0]
        preds = []
        for box in results.boxes:
            preds.append({
                'bbox': box.xyxy[0].tolist(),
                'score': float(box.conf),
                'label': int(box.cls)
            })
        return preds

# ------------------- RT-DETR ------------------- #
class RTDETRDetector:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor()
        ])

    @torch.no_grad()
    def predict(self, image_path):
        """
        Run RT-DETR inference on a single image
        Returns list of dicts: [{'bbox': [x1,y1,x2,y2], 'score': float, 'label': int}]
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        outputs = self.model(tensor)[0]

        preds = []
        for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
            preds.append({
                'bbox': box.tolist(),
                'score': float(score),
                'label': int(label)
            })
        return preds

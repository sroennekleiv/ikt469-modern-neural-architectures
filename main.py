import logging
import torch
import os 

import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.datasets import Kitti
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from sklearn.metrics import classification_report

from dataset.fashion_mnist import FashionMNISTDataset

from partA.cnn_baseline import PlainCNN
from partA.fire import SmallSqueezeNet
from partA.inception import SmallInception
from partA.residual import SmallResNet
from partA.super_network import SuperNet

from partA.training_evaluate import run_experiment

# Logging for debug and command information to monitor
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

CLASSES = 10

MODELS = {
    "Plain CNN": PlainCNN(CLASSES),
    "ResNet-like": SmallResNet(CLASSES),
    "Inception-like": SmallInception(CLASSES),
    "SqueezeNet-like": SmallSqueezeNet(CLASSES),
    "Super Net": SuperNet(CLASSES)
}

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    '''log.info("Start at part A: Image classification with residual + inception + fire")
    fashion_mnist = FashionMNISTDataset(batch_size=64, augment=False, size=(28,28), validation_split=0.2)
    train_loader, val_loader, test_loader = fashion_mnist.get_data()
    target_names = fashion_mnist.get_target_names()
    CLASSES = len(target_names)

    log.debug(f"Number of classes: {CLASSES}")

    results = {}
    for name, model in MODELS.items():
        log.info(f"---- Training {name} -----")
        test_acc = run_experiment(model, train_loader, val_loader, test_loader, epochs=10)
        results[name] = test_acc
'''
    log.info("Start part B: Object detection YOLO vs RT-DETR")
    MAX_IMG = 200

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = Kitti(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    from torch.utils.data import Subset
    dataset = Subset(dataset, list(range(MAX_IMG)))

    yolo = YOLO('yolov8n.pt')

    processor = RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_r18vd')
    rtdetr = RTDetrForObjectDetection.from_pretrained('PekingU/rtdetr_r18vd')

    rtdetr.eval()

    metric_yolo = MeanAveragePrecision(iou_thresholds=[0.5])
    metric_rtdetr = MeanAveragePrecision(iou_thresholds=[0.5])

    yolo_fail = []
    rtdetr_fail = []

    def kitti_target_to_dict(target):
        boxes = []
        labels = []

        for obj in target:
            if obj["type"] == "DontCare":
                continue

            xmin, ymin, xmax, ymax = obj["bbox"]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1) 

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return {"boxes": boxes, "labels": labels}
    
    for idx, (img, target) in enumerate(dataset):
        print(f"Image {idx+1}/{len(dataset)}")

        img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        img_path = f"temp_{idx}.png"
        img_pil.save(img_path)

        gt = kitti_target_to_dict(target)

        # -------- YOLO --------
        yolo_res = yolo(img_path, conf=0.25, verbose=False)[0]

        if len(yolo_res.boxes) > 0:
            y_boxes = yolo_res.boxes.xyxy.cpu()
            y_scores = yolo_res.boxes.conf.cpu()
            y_labels = torch.ones(len(y_boxes), dtype=torch.int64)
        else:
            y_boxes = torch.zeros((0, 4))
            y_scores = torch.zeros(0)
            y_labels = torch.zeros(0, dtype=torch.int64)

        metric_yolo.update(
            [{"boxes": y_boxes, "scores": y_scores, "labels": y_labels}],
            [gt]
        )

        # -------- RT-DETR --------
        inputs = processor(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = rtdetr(**inputs)

        results = processor.post_process_object_detection(
            outputs,
            threshold=0.25,
            target_sizes=[img_pil.size[::-1]]
        )[0]

        if len(results["boxes"]) > 0:
            r_boxes = results["boxes"].cpu()
            r_scores = results["scores"].cpu()
            r_labels = torch.ones(len(r_boxes), dtype=torch.int64)
        else:
            r_boxes = torch.zeros((0, 4))
            r_scores = torch.zeros(0)
            r_labels = torch.zeros(0, dtype=torch.int64)

        metric_rtdetr.update(
            [{"boxes": r_boxes, "scores": r_scores, "labels": r_labels}],
            [gt]
        )

        
        if len(y_boxes) == 0 and len(r_boxes) > 0:
            yolo_fail.append(idx)
        if len(r_boxes) == 0 and len(y_boxes) > 0:
            rtdetr_fail.append(idx)

        os.remove(img_path)

    print("\n===== mAP@0.5 =====")
    print("YOLO:", metric_yolo.compute())
    print("RT-DETR:", metric_rtdetr.compute())

    print("\nYOLO failed, RT-DETR succeeded:")
    print(yolo_fail[:10])

    print("\nRT-DETR failed, YOLO succeeded:")
    print(rtdetr_fail[:10])
# main.py
from dataset import load_voc_subset, get_image_path, parse_voc_target
from models import YOLODetector, RTDETRDetector
from mean_average_precision import MetricBuilder
import json
import torch

# ---------------- LOAD DATASET ---------------- #
subset_size = 200
dataset = load_voc_subset(num_images=subset_size)

# ---------------- INIT MODELS ---------------- #
yolo = YOLODetector(weights="yolov8n.pt")

# For RT-DETR, load your pretrained model
# Example placeholder:
# import torch
# from rtdetr import build_model
# model = build_model(...)
# model.load_state_dict(torch.load("rtdetr.pth"))
# rdetr = RTDETRDetector(model)
rdetr = None  # <-- replace with your RT-DETR wrapper

# ---------------- METRIC SETUP ---------------- #
metric_yolo = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=20)
metric_rdetr = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=20)

# ---------------- EVALUATION ---------------- #
yolo_preds_all = []
rdetr_preds_all = []
gts_all = []

for i in range(len(dataset)):
    img_path = get_image_path(dataset, i)
    target = dataset[i][1]

    gts = parse_voc_target(target)
    gts_all.append(gts)

    yolo_preds = yolo.predict(img_path)
    yolo_preds_all.append(yolo_preds)

    if rdetr:
        rdetr_preds = rdetr.predict(img_path)
        rdetr_preds_all.append(rdetr_preds)

# ---------------- CALCULATE mAP ---------------- #
map_yolo = metric_yolo.value(yolo_preds_all, gts_all)
print("YOLO mAP@0.5:", map_yolo)

if rdetr:
    map_rdetr = metric_rdetr.value(rdetr_preds_all, gts_all)
    print("RT-DETR mAP@0.5:", map_rdetr)

# ---------------- FAILURE CASES ---------------- #
def is_failure(preds, gts, iou_thresh=0.5):
    return len(preds) < len(gts)

yolo_fail = []
rdetr_fail = []

for i in range(len(dataset)):
    if is_failure(yolo_preds_all[i], gts_all[i]) and rdetr and not is_failure(rdetr_preds_all[i], gts_all[i]):
        yolo_fail.append(get_image_path(dataset, i))
    if rdetr and is_failure(rdetr_preds_all[i], gts_all[i]) and len(yolo_preds_all[i]) >= len(gts_all[i]):
        rdetr_fail.append(get_image_path(dataset, i))

# Save 10 representative failures
with open("yolo_fail.json", "w") as f:
    json.dump(yolo_fail[:10], f, indent=2)

with open("rdetr_fail.json", "w") as f:
    json.dump(rdetr_fail[:10], f, indent=2)

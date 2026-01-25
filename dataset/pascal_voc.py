# dataset.py
import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import Subset
import os

def load_voc_subset(root="data", year="2007", image_set="test", num_images=200):
    """
    Load a subset of Pascal VOC dataset.
    """
    dataset = VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=True
    )

    indices = torch.randperm(len(dataset))[:num_images]
    subset = Subset(dataset, indices)

    # store original indices for matching paths
    subset.indices = indices
    subset.dataset = dataset
    return subset

def get_image_path(dataset, idx):
    """
    Get the actual file path of an image in the subset.
    """
    original_idx = dataset.indices[idx]
    filename = dataset.dataset.images[original_idx]
    return filename

def parse_voc_target(target):
    """
    Convert VOC annotation to list of dicts:
    [{'bbox': [xmin, ymin, xmax, ymax], 'label': class_name}, ...]
    """
    objs = target['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    gt_list = []
    for obj in objs:
        b = obj['bndbox']
        gt_list.append({
            'bbox': [
                int(b['xmin']),
                int(b['ymin']),
                int(b['xmax']),
                int(b['ymax'])
            ],
            'label': obj['name']
        })
    return gt_list

import logging
import torch
import os 

import numpy as np
from torch.utils.data import DataLoader

from PIL import Image
from torchvision import transforms
from torchvision.datasets import Kitti
#from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
#from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from sklearn.metrics import classification_report

from dataset.fashion_mnist import FashionMNISTDataset

from Assignment_4.partA.autoencoder_embedding import ConvAutoEncoder
from Assignment_4.partA.contrastive_embedding import EmbeddingNet, train_cont, evaluate_and_visualize
from Assignment_4.partB.CLIP import run_partC
from Assignment_4.partB.Flamingo import run_flamingo_tsne
 
from partA.training_evaluate import run_experiment
from plot import visualize_tsne

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

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    fashion_mnist = FashionMNISTDataset(batch_size=64, augment=False, size=(28,28), validation_split=0.2)
    train_loader, val_loader, test_loader = fashion_mnist.get_data()
    target_names = fashion_mnist.get_target_names()
    CLASSES = len(target_names)
    log.debug(f"Number of classes: {CLASSES}")

    log.info("Start at part A: Learning Image Embedding from Scratch")

    '''auto = ConvAutoEncoder(latent_dim=64)

    log.info(f"---- Training -----")
    embeddings, labels = run_experiment(auto, train_loader, val_loader, test_loader, epochs=3, auto=True)
    visualize_tsne(embeddings, labels)

    contrastive = EmbeddingNet(emb_dim=64)
    train_cont(contrastive, epochs=3, train_loader=train_loader)
    evaluate_and_visualize(contrastive, test_loader)'''
    
    #run_partC(test_loader)
    run_flamingo_tsne(test_loader)
    
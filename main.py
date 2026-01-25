import logging
import kagglehub

from sklearn.metrics import classification_report

from dataset.pascal_voc import PascalVOC
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

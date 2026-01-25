
'''
2026-01-23 12:42:40,942 [INFO] ---- Training Plain CNN -----
Epoch 1 | Loss 0.9556 | Train Acc 65.45 | Val Acc 76.61
Epoch 2 | Loss 0.6260 | Train Acc 77.58 | Val Acc 81.07
Epoch 3 | Loss 0.5240 | Train Acc 81.32 | Val Acc 83.33
Epoch 4 | Loss 0.4639 | Train Acc 83.39 | Val Acc 85.55
Epoch 5 | Loss 0.4219 | Train Acc 84.98 | Val Acc 84.46
Epoch 6 | Loss 0.3923 | Train Acc 86.14 | Val Acc 85.91
Epoch 7 | Loss 0.3709 | Train Acc 86.81 | Val Acc 86.45
Epoch 8 | Loss 0.3516 | Train Acc 87.43 | Val Acc 87.30
Epoch 9 | Loss 0.3379 | Train Acc 87.85 | Val Acc 88.29
Epoch 10 | Loss 0.3239 | Train Acc 88.34 | Val Acc 88.23
2026-01-23 13:00:03,274 [INFO] ---- Training ResNet-like -----
Epoch 1 | Loss 0.5618 | Train Acc 80.72 | Val Acc 80.27
Epoch 2 | Loss 0.3519 | Train Acc 87.35 | Val Acc 86.39
Epoch 3 | Loss 0.2942 | Train Acc 89.30 | Val Acc 87.62
Epoch 4 | Loss 0.2626 | Train Acc 90.53 | Val Acc 89.88
Epoch 5 | Loss 0.2357 | Train Acc 91.31 | Val Acc 89.85
Epoch 6 | Loss 0.2116 | Train Acc 92.32 | Val Acc 90.04
Epoch 7 | Loss 0.1969 | Train Acc 92.73 | Val Acc 88.49
Epoch 8 | Loss 0.1753 | Train Acc 93.57 | Val Acc 90.47
Epoch 9 | Loss 0.1618 | Train Acc 94.01 | Val Acc 89.98
Epoch 10 | Loss 0.1448 | Train Acc 94.84 | Val Acc 89.88
2026-01-23 13:39:00,674 [INFO] ---- Training Inception-like -----
Epoch 1 | Loss 1.2287 | Train Acc 53.53 | Val Acc 64.53
Epoch 2 | Loss 0.8629 | Train Acc 68.76 | Val Acc 72.50
Epoch 3 | Loss 0.7543 | Train Acc 73.64 | Val Acc 73.26
Epoch 4 | Loss 0.6872 | Train Acc 76.10 | Val Acc 77.42
Epoch 5 | Loss 0.6341 | Train Acc 78.04 | Val Acc 78.36
Epoch 6 | Loss 0.5962 | Train Acc 79.27 | Val Acc 80.54
Epoch 7 | Loss 0.5697 | Train Acc 80.08 | Val Acc 80.81
Epoch 8 | Loss 0.5534 | Train Acc 80.71 | Val Acc 79.42
Epoch 9 | Loss 0.5396 | Train Acc 81.25 | Val Acc 82.24
Epoch 10 | Loss 0.5154 | Train Acc 82.04 | Val Acc 83.29
2026-01-23 14:40:20,482 [INFO] ---- Training SqueezeNet-like -----
Epoch 1 | Loss 1.3843 | Train Acc 48.77 | Val Acc 68.77
Epoch 2 | Loss 0.8757 | Train Acc 69.18 | Val Acc 72.87
Epoch 3 | Loss 0.7910 | Train Acc 72.69 | Val Acc 73.85
Epoch 4 | Loss 0.7449 | Train Acc 73.92 | Val Acc 76.77
Epoch 5 | Loss 0.7079 | Train Acc 75.14 | Val Acc 76.02
Epoch 6 | Loss 0.6707 | Train Acc 76.64 | Val Acc 76.74
Epoch 7 | Loss 0.6468 | Train Acc 77.25 | Val Acc 78.17
Epoch 8 | Loss 0.6299 | Train Acc 77.87 | Val Acc 79.03
Epoch 9 | Loss 0.6122 | Train Acc 78.48 | Val Acc 78.12
Epoch 10 | Loss 0.6065 | Train Acc 78.59 | Val Acc 80.30
2026-01-23 14:59:14,177 [INFO] ---- Training Super Net -----
Epoch 1 | Loss 0.6636 | Train Acc 77.97 | Val Acc 54.67
Epoch 2 | Loss 0.3920 | Train Acc 86.19 | Val Acc 79.30
Epoch 3 | Loss 0.3334 | Train Acc 88.20 | Val Acc 75.20
Epoch 4 | Loss 0.3059 | Train Acc 89.10 | Val Acc 85.20
Epoch 5 | Loss 0.2792 | Train Acc 90.12 | Val Acc 88.38
Epoch 6 | Loss 0.2652 | Train Acc 90.70 | Val Acc 87.03
Epoch 7 | Loss 0.2497 | Train Acc 91.13 | Val Acc 88.49
Epoch 8 | Loss 0.2389 | Train Acc 91.61 | Val Acc 89.72
Epoch 9 | Loss 0.2261 | Train Acc 91.97 | Val Acc 85.17
Epoch 10 | Loss 0.2164 | Train Acc 92.25 | Val Acc 90.60    
'''

import matplotlib.pyplot as plt

# ---------------------------
# Data from logs
# ---------------------------
logs = {
    "Plain CNN": {
        "loss": [0.9556, 0.6260, 0.5240, 0.4639, 0.4219, 0.3923, 0.3709, 0.3516, 0.3379, 0.3239],
        "train_acc": [65.45, 77.58, 81.32, 83.39, 84.98, 86.14, 86.81, 87.43, 87.85, 88.34],
        "val_acc": [76.61, 81.07, 83.33, 85.55, 84.46, 85.91, 86.45, 87.30, 88.29, 88.23]
    },
    "ResNet-like": {
        "loss": [0.5618, 0.3519, 0.2942, 0.2626, 0.2357, 0.2116, 0.1969, 0.1753, 0.1618, 0.1448],
        "train_acc": [80.72, 87.35, 89.30, 90.53, 91.31, 92.32, 92.73, 93.57, 94.01, 94.84],
        "val_acc": [80.27, 86.39, 87.62, 89.88, 89.85, 90.04, 88.49, 90.47, 89.98, 89.88]
    },
    "Inception-like": {
        "loss": [1.2287, 0.8629, 0.7543, 0.6872, 0.6341, 0.5962, 0.5697, 0.5534, 0.5396, 0.5154],
        "train_acc": [53.53, 68.76, 73.64, 76.10, 78.04, 79.27, 80.08, 80.71, 81.25, 82.04],
        "val_acc": [64.53, 72.50, 73.26, 77.42, 78.36, 80.54, 80.81, 79.42, 82.24, 83.29]
    },
    "SqueezeNet-like": {
        "loss": [1.3843, 0.8757, 0.7910, 0.7449, 0.7079, 0.6707, 0.6468, 0.6299, 0.6122, 0.6065],
        "train_acc": [48.77, 69.18, 72.69, 73.92, 75.14, 76.64, 77.25, 77.87, 78.48, 78.59],
        "val_acc": [68.77, 72.87, 73.85, 76.77, 76.02, 76.74, 78.17, 79.03, 78.12, 80.30]
    },
    "Super Net": {
        "loss": [0.6636, 0.3920, 0.3334, 0.3059, 0.2792, 0.2652, 0.2497, 0.2389, 0.2261, 0.2164],
        "train_acc": [77.97, 86.19, 88.20, 89.10, 90.12, 90.70, 91.13, 91.61, 91.97, 92.25],
        "val_acc": [54.67, 79.30, 75.20, 85.20, 88.38, 87.03, 88.49, 89.72, 85.17, 90.60]
    }
}

epochs = list(range(1, 11))
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']

# ---------------------------
# Plot Loss
# ---------------------------
plt.figure(figsize=(10,6))
for i, (model, values) in enumerate(logs.items()):
    plt.plot(epochs, values['loss'], color=colors[i], linestyle='--', label=model)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for All Models')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# ---------------------------
# Plot Train & Validation Accuracy
# ---------------------------
plt.figure(figsize=(10,6))


# Plot all models' train and val accuracy
for i, (model, values) in enumerate(logs.items()):
    plt.plot(epochs, values['train_acc'], color=colors[i], linestyle='--', label=f'{model} Train')
    plt.plot(epochs, values['val_acc'], color=colors[i], linestyle='-', label=f'{model} Val')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy for All Models')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig("accuracy_plot_with_info_boxes.png")
plt.show()
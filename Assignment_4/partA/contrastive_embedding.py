import random
import torch

import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def make_pairs(x, y):
    pairs = []
    targets = []

    for i in range(len(x)):
        x1, y1 = x[i], y[i]

        # Same label -> positive
        pos_idx = random.choice((y == y1).nonzero(as_tuple=True)[0])
        pairs.append((x1, x[pos_idx]))

        targets.append(1)
        # Different label -> negative
        neg_idx = random.choice((y != y1).nonzero(as_tuple=True)[0])
        pairs.append((x1, x[neg_idx]))
        targets.append(0)

    return pairs, torch.tensor(targets)

class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, emb_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    # Pull positives togheter and negatives apart
    def forward(self, z1, z2, label):
        dist = torch.norm(z1 - z2, dim=1)
        loss = label * dist.pow(2) + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()
    
def train_cont(model, epochs, train_loader):
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            pairs, labels = make_pairs(x, y)
            labels = labels

            x1 = torch.stack([p[0] for p in pairs])
            x2 = torch.stack([p[1] for p in pairs])

            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate_and_visualize(model, test_loader):
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            z = model(x)
            embeddings.append(z.cpu())
            labels.append(y)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )

    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=6,
        alpha=0.8
    )
    plt.colorbar(scatter)
    plt.title("t-SNE of Contrastive Embedding Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig("contrastive.png")
    plt.show()

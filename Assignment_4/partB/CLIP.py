import clip
import torch
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to CLIP input size
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.to(device)


def extract_clip_embeddings(dataloader):
    embeddings = []
    labels = []

    for x, y in dataloader:
        x = x.to(device)

        # Grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to CLIP expected size
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            z = model.encode_image(x)
        embeddings.append(z.cpu())
        labels.append(y)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    
    return embeddings, labels


def run_partC(test_loader):
    embeddings, labels = extract_clip_embeddings(test_loader)

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
    plt.title("t-SNE of CLIP Image Embeddings")
    plt.tight_layout()
    plt.savefig("contrastive.png")
    plt.show()

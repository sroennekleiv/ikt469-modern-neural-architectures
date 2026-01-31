import torch
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

def train_one_epoch_auto(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, _ in loader:
        optimizer.zero_grad()
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total

@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()

    embeddings = []
    labels = []

    for x, y in loader:
        _, z = model(x)

        embeddings.append(z.cpu())
        labels.append(y)

    return torch.cat(embeddings), torch.cat(labels)

def run_experiment(model, train_loader, val_loader, test_loader, epochs=10, auto=True):
    if auto:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        if auto:
            train_loss = train_one_epoch_auto(
                model, train_loader, optimizer, criterion
                )
            print(f"Epoch {epoch+1} | Recon Loss {train_loss:.4f}")
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion
            )
            val_acc = evaluate(model, val_loader)
            print(
                f"Epoch {epoch+1} | "
                f"Loss {train_loss:.4f} | "
                f"Train Acc {train_acc*100:.2f} | "
                f"Val Acc {val_acc*100:.2f}"
            )

    if auto:
        embeddings, labels = extract_embeddings(model, test_loader)
        return embeddings, labels
    else:
        acc = evaluate(model, test_loader)
        return acc

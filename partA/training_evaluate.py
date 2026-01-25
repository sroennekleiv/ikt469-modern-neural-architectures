import torch
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, criterion,):
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

def run_experiment(model, train_loader, val_loader, test_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Loss {train_loss:.4f} | "
            f"Train Acc {train_acc * 100:.2f} | "
            f"Val Acc {val_acc * 100:.2f}"
        )

    test_acc = evaluate(model, test_loader)
    return test_acc

import torch

def evaluate(model, logits, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

NUM_CLASSES = 67

def get_model(model_name: str, pretrained: bool = True, device: torch.device = None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, NUM_CLASSES),
        )

    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, NUM_CLASSES),
        )

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.vgg19(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "densenet169":
        weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
        model = models.densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "custom_cnn":
        model = CustomCNN(num_classes=NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    return model

class CustomCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_available_models() -> List[str]:
    return [
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b1",
        "vgg16",
        "vgg19",
        "densenet121",
        "densenet169",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "custom_cnn",
    ]

def get_model_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_probs)

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    patience = 7

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Device: {device}")
    print(f"Model params: {get_model_size(model):,}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm=max_grad_norm
        )
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"*** New best model! Val Acc: {val_acc:.4f} ***")

            if save_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_state,
                        "val_acc": best_val_acc,
                        "model_name": model_name,
                    },
                    save_path,
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    result = {
        "model_name": model_name,
        "model": model,
        "best_val_acc": best_val_acc,
        "history": history,
        "num_params": get_model_size(model),
    }

    print(f"\n{'='*60}")
    print(f"Training complete for {model_name}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"{'='*60}")

    return result

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    misclassified = np.where(all_preds != all_labels)[0]
    misclassified_info = []
    for idx in misclassified[:50]:
        misclassified_info.append({
            "index": int(idx),
            "true_label": int(all_labels[idx]),
            "pred_label": int(all_preds[idx]),
            "true_name": class_names.get(int(all_labels[idx]), f"Class {all_labels[idx]}") if class_names else f"Class {all_labels[idx]}",
            "pred_name": class_names.get(int(all_preds[idx]), f"Class {all_preds[idx]}") if class_names else f"Class {all_preds[idx]}",
            "confidence": float(all_probs[idx][int(all_preds[idx])]),
            "true_confidence": float(all_probs[idx][int(all_labels[idx])]),
        })

    avg_inference_time = np.mean(inference_times)
    avg_inference_time_per_sample = avg_inference_time / test_loader.batch_size if test_loader.batch_size else avg_inference_time

    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append({
                    "true_class": i,
                    "pred_class": j,
                    "count": int(cm[i][j]),
                    "true_name": class_names.get(i, f"Class {i}") if class_names else f"Class {i}",
                    "pred_name": class_names.get(j, f"Class {j}") if class_names else f"Class {j}",
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "class_support": class_support,
        "misclassified": misclassified_info[:20],
        "confused_pairs": confused_pairs[:20],
        "avg_inference_time": avg_inference_time,
        "avg_inference_time_per_sample": avg_inference_time_per_sample,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }

def test_robustness(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_samples: int = 200,
) -> Dict[str, Dict[str, float]]:
    from src.data import apply_degradation, get_val_transforms

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    base_transform = get_val_transforms(48)

    degradation_types = ["blur", "noise", "darken", "small"]
    severities = [0.0, 0.2, 0.4, 0.6, 0.8]

    results = {}

    for deg_type in degradation_types:
        results[deg_type] = {}
        for severity in severities:
            correct = 0
            total = 0

            with torch.no_grad():
                for idx in indices:
                    img, label = dataset[idx]

                    degraded = apply_degradation(img, deg_type, severity)
                    degraded = degraded.unsqueeze(0).to(device)

                    output = model(degraded)
                    _, pred = torch.max(output, 1)

                    if pred.item() == label:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0
            results[deg_type][severity] = acc
            print(f"  {deg_type} severity={severity}: acc={acc:.4f}")

    return results

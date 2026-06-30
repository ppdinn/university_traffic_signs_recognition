import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import (
    get_data_loaders,
    get_test_loader,
    get_class_weights,
    TrafficSignDataset,
    get_val_transforms,
)
from src.models import (
    get_model,
    train_model,
    evaluate_model,
    test_robustness,
    get_available_models,
    get_model_size,
)

MODEL_LR_OVERRIDES = {
    "vgg16": 1e-4,
    "vgg19": 1e-4,
}
MODEL_GRAD_CLIP = {
    "vgg16": 2.0,
    "vgg19": 2.0,
}

def train_single_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device = None,
    save_dir: str = "models",
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*60}")
    print(f"# Training {model_name}")
    print(f"{'#'*60}")

    model = get_model(model_name, pretrained=True, device=device)
    num_params = get_model_size(model)
    print(f"Model parameters: {num_params:,}")

    effective_lr = MODEL_LR_OVERRIDES.get(model_name, learning_rate)
    max_grad_norm = MODEL_GRAD_CLIP.get(model_name, None)
    if effective_lr != learning_rate:
        print(f"Using adjusted learning rate for {model_name}: {effective_lr}")
    if max_grad_norm is not None:
        print(f"Using gradient clipping for {model_name}: max_norm={max_grad_norm}")

    save_path = os.path.join(save_dir, f"{model_name}_best.pth")

    start_time = time.time()
    result = train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=effective_lr,
        device=device,
        class_weights=class_weights,
        save_path=save_path,
        max_grad_norm=max_grad_norm,
    )
    training_time = time.time() - start_time

    result["training_time"] = training_time
    result["num_params"] = num_params

    return result

def compare_models(
    model_names: List[str],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
    full_dataset: TrafficSignDataset,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device = None,
    save_dir: str = "models",
    results_dir: str = "results",
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for model_name in model_names:
        try:
            result = train_single_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_dir=save_dir,
            )

            print(f"\nEvaluating {model_name} on validation set...")
            model = result["model"]
            val_metrics = evaluate_model(model, val_loader, device)

            print(f"Testing robustness of {model_name}...")
            robustness = test_robustness(model, full_dataset, device, num_samples=200)

            result["val_metrics"] = {
                "accuracy": float(val_metrics["accuracy"]),
                "precision": float(val_metrics["precision"]),
                "recall": float(val_metrics["recall"]),
                "f1_score": float(val_metrics["f1_score"]),
                "avg_inference_time_ms": float(val_metrics["avg_inference_time_per_sample"] * 1000),
            }
            result["robustness"] = {
                deg: {str(sev): float(acc) for sev, acc in sevs.items()}
                for deg, sevs in robustness.items()
            }

            all_results[model_name] = result

            with open(os.path.join(results_dir, f"{model_name}_results.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "best_val_acc": float(result["best_val_acc"]),
                    "num_params": result["num_params"],
                    "training_time": result["training_time"],
                    "val_metrics": result["val_metrics"],
                    "robustness": result["robustness"],
                    "history": {
                        k: [float(v) for v in vals]
                        for k, vals in result["history"].items()
                    },
                }, f, indent=2)

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    comparison_path = os.path.join(results_dir, "model_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)
    else:
        comparison = {}
    for name, res in all_results.items():
        comparison[name] = {
            "best_val_acc": float(res["best_val_acc"]),
            "num_params": res["num_params"],
            "training_time": res["training_time"],
            "val_accuracy": res["val_metrics"]["accuracy"],
            "val_precision": res["val_metrics"]["precision"],
            "val_recall": res["val_metrics"]["recall"],
            "val_f1": res["val_metrics"]["f1_score"],
            "inference_time_ms": res["val_metrics"]["avg_inference_time_ms"],
        }

    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Val Acc':<10} {'Params':<12} {'Time(s)':<10} {'Inf(ms)':<10}")
    print("-"*80)
    for name, metrics in sorted(comparison.items(), key=lambda x: x[1]["best_val_acc"], reverse=True):
        print(f"{name:<25} {metrics['best_val_acc']:.4f}    {metrics['num_params']:<10,} {metrics['training_time']:<10.1f} {metrics['inference_time_ms']:<10.3f}")
    print("="*80)

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Train traffic sign recognition models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train (default: 5 architectures)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--img-size", type=int, default=48,
                        help="Image size")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.models:
        model_names = args.models
    else:
        model_names = [
            "resnet50",
            "efficientnet_b0",
            "vgg16",
            "densenet121",
            "mobilenet_v3_small",
        ]

    print(f"Models to train: {model_names}")

    print("\nLoading data...")
    train_loader, val_loader, full_dataset = get_data_loaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_ratio=0.2,
    )

    test_loader = get_test_loader(
        batch_size=args.batch_size,
        img_size=args.img_size,
    )

    class_weights = get_class_weights(full_dataset)
    print(f"Class weights computed (min: {class_weights.min():.4f}, max: {class_weights.max():.4f})")

    results = compare_models(
        model_names=model_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        full_dataset=full_dataset,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
    )

    print("\nTraining complete!")

if __name__ == "__main__":
    main()

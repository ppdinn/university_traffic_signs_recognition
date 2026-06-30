import sys, os, json, torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import CLASS_NAMES
from src.models import get_model, evaluate_model
from src.data import get_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

model = get_model('densenet121', pretrained=False, device=device)
ckpt = torch.load('models/densenet121_best.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print('Model loaded')

_, val_loader, _ = get_data_loaders(batch_size=64, img_size=48, val_ratio=0.2)
results = evaluate_model(model, val_loader, device, CLASS_NAMES)
print('Accuracy:', results['accuracy'])
print('Misclassified:', len(results['misclassified']))

out = {
    'accuracy': float(results['accuracy']),
    'precision': float(results['precision']),
    'recall': float(results['recall']),
    'f1': float(results['f1_score']),
    'misclassified': results['misclassified'][:30],
    'confused_pairs': results['confused_pairs'][:20],
}
with open('results/error_analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved results/error_analysis_results.json')

print('\nTop confused pairs:')
for cp in results['confused_pairs'][:10]:
    print(f"  {cp['true_name']} -> {cp['pred_name']}: {cp['count']}")

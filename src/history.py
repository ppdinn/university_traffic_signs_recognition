import os
import json
from datetime import datetime
from typing import Dict, List, Optional

HISTORY_PATH = os.path.join("results", "inference_history.json")

def log_inference(
    model_name: str,
    pred_class: int,
    pred_name: str,
    confidence: float,
    image_name: Optional[str] = None,
    degradation: Optional[str] = None,
    severity: Optional[float] = None,
    top5: Optional[List[Dict]] = None,
    history_path: str = HISTORY_PATH,
) -> Dict:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "image_name": image_name,
        "pred_class": int(pred_class),
        "pred_name": pred_name,
        "confidence": round(float(confidence), 6),
        "degradation": degradation,
        "severity": severity,
        "top5": top5,
    }

    history = load_history(history_path)
    history.append(record)

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return record

def load_history(history_path: str = HISTORY_PATH) -> List[Dict]:
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []

def clear_history(history_path: str = HISTORY_PATH) -> None:
    if os.path.exists(history_path):
        os.remove(history_path)

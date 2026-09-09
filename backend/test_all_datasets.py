"""
OptiCare - Comprehensive Multi-Dataset Evaluation
Tests both models (base + finetuned) across APTOS, IDRiD, and Messidor-2 datasets.
Also tests WITH and WITHOUT CLAHE to detect train/inference mismatch.

Reports:
  - Overall accuracy, per-class accuracy, confusion matrix
  - Sensitivity/Specificity (binary: disease vs no-disease)
  - Error patterns (off-by-1, dangerous misses, false alarms)
  - CLAHE vs no-CLAHE comparison
"""

import os
import sys
import csv
import json
import time
import random
from collections import defaultdict, Counter
import torch, torch.nn as nn, numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2

sys.stdout.reconfigure(encoding='utf-8')

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE = os.path.join(BACKEND_DIR, "dr_trained_model.pth")
MODEL_FINETUNED = os.path.join(BACKEND_DIR, "dr_finetuned_idrid.pth")
TEMPERATURE = 2.5
SEVERITY_WEIGHTS = [1.0, 1.0, 1.0, 1.15, 1.20]

DR = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Prolif"}

# Dataset paths
APTOS_CSV = r"C:\Users\Sahindeep\Documents\Dataset\APTOS\train_1.csv"
APTOS_DIR = r"C:\Users\Sahindeep\Documents\Dataset\APTOS\train_images"

IDRID_CSV = r"C:\Users\Sahindeep\Documents\Dataset\IDRiD\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_DIR = r"C:\Users\Sahindeep\Documents\Dataset\IDRiD\B. Disease Grading\1. Original Images\a. Training Set"

MESSIDOR_CSV = r"C:\Users\Sahindeep\Documents\Dataset\Messidor-2\archive\messidor_data.csv"
MESSIDOR_DIR = r"C:\Users\Sahindeep\Documents\Dataset\Messidor-2\IMAGES.zip\IMAGES"

# Sampling
APTOS_SAMPLES = 200
MESSIDOR_SAMPLES = 200

# Output
RESULTS_FILE = os.path.join(BACKEND_DIR, "test_results.txt")
RESULTS_JSON = os.path.join(BACKEND_DIR, "test_results.json")


# ─── CLAHE PREPROCESSING ───────────────────────────────────────────────────
def apply_clahe(img_path):
    """Same CLAHE as training pipeline (train_model.py lines 80-91)."""
    img = cv2.imread(img_path)
    if img is None:
        return Image.new('RGB', (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def load_image_no_clahe(img_path):
    """Load image WITHOUT CLAHE (what app.py currently does at inference)."""
    return Image.open(img_path).convert('RGB')


# ─── TRANSFORMS ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─── DATASET LOADERS ───────────────────────────────────────────────────────
def load_aptos_labels():
    labels = {}
    with open(APTOS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            img_path = os.path.join(APTOS_DIR, row[0] + '.png')
            if os.path.exists(img_path):
                labels[img_path] = int(row[1])
    return labels


def load_idrid_labels():
    labels = {}
    with open(IDRID_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip().isdigit():
                img_path = os.path.join(IDRID_DIR, row[0].strip() + '.jpg')
                if os.path.exists(img_path):
                    labels[img_path] = int(row[1].strip())
    return labels


def load_messidor_labels():
    labels = {}
    with open(MESSIDOR_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4 and row[3].strip() == '1':  # gradable only
                img_path = os.path.join(MESSIDOR_DIR, row[0].strip())
                if os.path.exists(img_path):
                    labels[img_path] = int(row[1].strip())
    return labels


def sample_balanced(labels_dict, n_per_class=None, total=None):
    """Sample images, trying to balance across classes."""
    by_class = defaultdict(list)
    for path, label in labels_dict.items():
        by_class[label].append(path)

    if n_per_class:
        sampled = []
        for lb in sorted(by_class):
            pool = by_class[lb]
            random.shuffle(pool)
            chosen = pool[:n_per_class]
            for p in chosen:
                sampled.append((p, lb))
        return sampled
    elif total:
        # Proportional sampling
        all_items = list(labels_dict.items())
        random.shuffle(all_items)
        return [(p, lb) for p, lb in all_items[:total]]
    else:
        return list(labels_dict.items())


# ─── EVALUATION ENGINE ──────────────────────────────────────────────────────
def evaluate(model, device, test_set, use_clahe=False, use_weighted=False, label=""):
    """Run inference on test_set and compute all metrics."""
    correct = 0
    total = len(test_set)
    cc = [0] * 5
    ct = [0] * 5
    conf_matrix = [[0] * 5 for _ in range(5)]
    errors = []
    confs_right = []
    confs_wrong = []
    pred_dist = [0] * 5

    t0 = time.time()
    for idx, (img_path, true_label) in enumerate(test_set):
        try:
            if use_clahe:
                img = apply_clahe(img_path)
            else:
                img = load_image_no_clahe(img_path)

            t = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(t)

            scaled = out[0] / TEMPERATURE
            probs = torch.nn.functional.softmax(scaled, dim=0)

            if use_weighted:
                weights = torch.tensor(SEVERITY_WEIGHTS, device=device)
                weighted = probs * weights
                pred = torch.argmax(weighted).item()
            else:
                pred = torch.argmax(probs).item()

            conf = probs[pred].item() * 100
            pred_dist[pred] += 1
        except Exception as e:
            print(f"  [ERROR] Could not process {os.path.basename(img_path)}: {e}")
            continue

        ok = pred == true_label
        correct += int(ok)
        cc[true_label] += int(ok)
        ct[true_label] += 1
        conf_matrix[true_label][pred] += 1

        if ok:
            confs_right.append(conf)
        else:
            confs_wrong.append(conf)
            errors.append({
                'image': os.path.basename(img_path),
                'true': true_label,
                'pred': pred,
                'conf': conf,
                'probs': [probs[i].item() * 100 for i in range(5)]
            })

        if (idx + 1) % 100 == 0:
            print(f"    [{label}] {idx + 1}/{total}...")

    elapsed = time.time() - t0

    # Compute metrics
    acc = 100 * correct / max(total, 1)
    per_class_acc = {}
    for i in range(5):
        if ct[i] > 0:
            per_class_acc[i] = 100 * cc[i] / ct[i]
        else:
            per_class_acc[i] = None

    # Binary metrics: disease (>=1) vs healthy (0)
    tp = sum(conf_matrix[i][j] for i in range(1, 5) for j in range(1, 5))
    fn = sum(conf_matrix[i][0] for i in range(1, 5))
    tn = conf_matrix[0][0]
    fp = sum(conf_matrix[0][j] for j in range(1, 5))
    sensitivity = 100 * tp / max(tp + fn, 1)
    specificity = 100 * tn / max(tn + fp, 1)

    # Error analysis
    off_by_one = sum(1 for e in errors if abs(e['true'] - e['pred']) == 1)
    off_by_two = sum(1 for e in errors if abs(e['true'] - e['pred']) == 2)
    off_by_more = sum(1 for e in errors if abs(e['true'] - e['pred']) > 2)

    dangerous_misses = [e for e in errors if e['true'] >= 2 and e['pred'] == 0]
    false_alarms = [e for e in errors if e['true'] == 0 and e['pred'] >= 3]

    return {
        'label': label,
        'total': total,
        'correct': correct,
        'accuracy': acc,
        'per_class_acc': per_class_acc,
        'per_class_total': {i: ct[i] for i in range(5)},
        'per_class_correct': {i: cc[i] for i in range(5)},
        'confusion_matrix': conf_matrix,
        'pred_distribution': pred_dist,
        'errors': errors,
        'n_errors': len(errors),
        'off_by_one': off_by_one,
        'off_by_two': off_by_two,
        'off_by_more': off_by_more,
        'dangerous_misses': dangerous_misses,
        'false_alarms': false_alarms,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'avg_conf_right': float(np.mean(confs_right)) if confs_right else 0,
        'avg_conf_wrong': float(np.mean(confs_wrong)) if confs_wrong else 0,
        'elapsed': elapsed,
    }


def format_results(r):
    """Format results into a readable string."""
    lines = []
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  {r['label']}")
    lines.append(f"{'=' * 90}")
    lines.append(f"  Overall Accuracy: {r['accuracy']:.1f}% ({r['correct']}/{r['total']})")
    lines.append(f"  Time: {r['elapsed']:.1f}s")
    lines.append("")

    lines.append("  Per-Class Accuracy:")
    for i in range(5):
        t = r['per_class_total'][i]
        c = r['per_class_correct'][i]
        a = r['per_class_acc'][i]
        if a is not None:
            bar = "#" * int(a / 2) + "." * (50 - int(a / 2))
            lines.append(f"    Lv{i} ({DR[i]:10s}): {a:5.1f}% ({c:3d}/{t:3d}) [{bar}]")
        else:
            lines.append(f"    Lv{i} ({DR[i]:10s}): N/A (0 samples)")

    lines.append("")
    lines.append("  Prediction Distribution (how many times each class was predicted):")
    for i in range(5):
        lines.append(f"    Lv{i} ({DR[i]:10s}): {r['pred_distribution'][i]} times")

    lines.append("")
    lines.append("  Confusion Matrix (rows=true, cols=predicted):")
    lines.append("            " + "".join(f"  Lv{i}   " for i in range(5)))
    for i in range(5):
        row = f"    Lv{i}:   " + "".join(f"  {r['confusion_matrix'][i][j]:4d}  " for j in range(5))
        lines.append(row)

    lines.append("")
    lines.append(f"  Binary Disease Detection:")
    lines.append(f"    Sensitivity: {r['sensitivity']:.1f}% (catches disease)")
    lines.append(f"    Specificity: {r['specificity']:.1f}% (clears healthy)")

    lines.append("")
    lines.append(f"  Confidence Calibration:")
    lines.append(f"    Correct predictions avg conf: {r['avg_conf_right']:.1f}%")
    lines.append(f"    Wrong predictions avg conf:   {r['avg_conf_wrong']:.1f}%")

    lines.append("")
    lines.append(f"  Error Distance ({r['n_errors']} total errors):")
    lines.append(f"    Off by 1: {r['off_by_one']} ({100 * r['off_by_one'] / max(r['n_errors'], 1):.0f}%)")
    lines.append(f"    Off by 2: {r['off_by_two']} ({100 * r['off_by_two'] / max(r['n_errors'], 1):.0f}%)")
    lines.append(f"    Off by 3+: {r['off_by_more']} ({100 * r['off_by_more'] / max(r['n_errors'], 1):.0f}%)")

    lines.append("")
    if r['dangerous_misses']:
        lines.append(f"  ⚠ DANGEROUS MISSES (disease>=Moderate classified as No DR): {len(r['dangerous_misses'])}")
        for e in r['dangerous_misses'][:10]:
            lines.append(f"    [CRITICAL] {e['image']}: True {DR[e['true']]} -> Pred No DR ({e['conf']:.1f}%)")
    else:
        lines.append("  ✓ No dangerous misses (no disease>=Moderate classified as No DR)")

    if r['false_alarms']:
        lines.append(f"  ⚠ FALSE ALARMS (healthy classified as Severe/Prolif): {len(r['false_alarms'])}")
        for e in r['false_alarms'][:10]:
            lines.append(f"    [ALARM] {e['image']}: Healthy -> Pred {DR[e['pred']]} ({e['conf']:.1f}%)")
    else:
        lines.append("  ✓ No false alarms (no healthy eyes classified as Severe/Prolif)")

    # Top error patterns
    error_types = defaultdict(int)
    for e in r['errors']:
        key = f"True {DR[e['true']]} -> Pred {DR[e['pred']]}"
        error_types[key] += 1

    if error_types:
        lines.append("")
        lines.append("  Top Error Patterns:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {etype}: {count} errors")

    return "\n".join(lines)


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    output_lines = []

    def log(msg):
        print(msg)
        output_lines.append(msg)

    log("=" * 90)
    log("  OptiCare - Comprehensive Multi-Dataset Evaluation")
    log("=" * 90)
    log(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    # ── Load Models ────────────────────────────────────────────────────────
    models_to_test = {}

    if os.path.exists(MODEL_BASE):
        log(f"  Loading base model: {MODEL_BASE}")
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
        m.load_state_dict(torch.load(MODEL_BASE, map_location=device, weights_only=True))
        m.to(device).eval()
        models_to_test['base'] = m
    else:
        log(f"  [WARN] Base model not found: {MODEL_BASE}")

    if os.path.exists(MODEL_FINETUNED):
        log(f"  Loading finetuned model: {MODEL_FINETUNED}")
        m2 = models.resnet50(weights=None)
        m2.fc = nn.Linear(m2.fc.in_features, 5)
        m2.load_state_dict(torch.load(MODEL_FINETUNED, map_location=device, weights_only=True))
        m2.to(device).eval()
        models_to_test['finetuned'] = m2
    else:
        log(f"  [WARN] Finetuned model not found: {MODEL_FINETUNED}")

    if not models_to_test:
        log("ERROR: No models found to test!")
        return

    # ── Load Datasets ──────────────────────────────────────────────────────
    log("\n  Loading datasets...")

    datasets = {}
    
    if os.path.exists(APTOS_CSV) and os.path.exists(APTOS_DIR):
        aptos_labels = load_aptos_labels()
        log(f"  APTOS: {len(aptos_labels)} images found")
        dist = Counter(aptos_labels.values())
        log(f"    Distribution: {dict(sorted(dist.items()))}")
        # Sample: try to get balanced samples
        n_per_class = APTOS_SAMPLES // 5
        aptos_test = sample_balanced(aptos_labels, n_per_class=n_per_class)
        log(f"    Sampled: {len(aptos_test)} images ({n_per_class} per class target)")
        dist2 = Counter(lb for _, lb in aptos_test)
        log(f"    Sampled dist: {dict(sorted(dist2.items()))}")
        datasets['APTOS'] = aptos_test
    else:
        log("  [WARN] APTOS dataset not found")

    if os.path.exists(IDRID_CSV) and os.path.exists(IDRID_DIR):
        idrid_labels = load_idrid_labels()
        log(f"  IDRiD: {len(idrid_labels)} images found")
        dist = Counter(idrid_labels.values())
        log(f"    Distribution: {dict(sorted(dist.items()))}")
        # Use ALL IDRiD images
        idrid_test = list(idrid_labels.items())
        datasets['IDRiD'] = idrid_test
    else:
        log("  [WARN] IDRiD dataset not found")

    if os.path.exists(MESSIDOR_CSV) and os.path.exists(MESSIDOR_DIR):
        messidor_labels = load_messidor_labels()
        log(f"  Messidor-2: {len(messidor_labels)} images found")
        dist = Counter(messidor_labels.values())
        log(f"    Distribution: {dict(sorted(dist.items()))}")
        n_per_class = MESSIDOR_SAMPLES // 5
        messidor_test = sample_balanced(messidor_labels, n_per_class=n_per_class)
        log(f"    Sampled: {len(messidor_test)} images ({n_per_class} per class target)")
        dist2 = Counter(lb for _, lb in messidor_test)
        log(f"    Sampled dist: {dict(sorted(dist2.items()))}")
        datasets['Messidor-2'] = messidor_test
    else:
        log("  [WARN] Messidor-2 dataset not found")

    if not datasets:
        log("ERROR: No datasets found!")
        return

    # ── Run All Evaluations ────────────────────────────────────────────────
    all_results = {}

    for model_name, model in models_to_test.items():
        for ds_name, test_set in datasets.items():
            # Test 1: Without CLAHE (what app.py currently does)
            label = f"{model_name.upper()} | {ds_name} | No CLAHE (app.py inference mode)"
            log(f"\n  Running: {label}...")
            result = evaluate(model, device, test_set, use_clahe=False, use_weighted=False, label=label)
            res_str = format_results(result)
            log(res_str)
            all_results[f"{model_name}_{ds_name}_no_clahe"] = result

            # Test 2: With CLAHE (matching training preprocessing)
            label2 = f"{model_name.upper()} | {ds_name} | With CLAHE (training mode)"
            log(f"\n  Running: {label2}...")
            result2 = evaluate(model, device, test_set, use_clahe=True, use_weighted=False, label=label2)
            res_str2 = format_results(result2)
            log(res_str2)
            all_results[f"{model_name}_{ds_name}_clahe"] = result2

            # Test 3: With CLAHE + weighted severity (full pipeline as app.py intends)
            label3 = f"{model_name.upper()} | {ds_name} | CLAHE + Weighted Severity"
            log(f"\n  Running: {label3}...")
            result3 = evaluate(model, device, test_set, use_clahe=True, use_weighted=True, label=label3)
            res_str3 = format_results(result3)
            log(res_str3)
            all_results[f"{model_name}_{ds_name}_clahe_weighted"] = result3

    # ── CLAHE Impact Summary ───────────────────────────────────────────────
    log("\n" + "=" * 90)
    log("  CLAHE IMPACT ANALYSIS (Train/Inference Mismatch Detection)")
    log("=" * 90)
    log("  Training used CLAHE preprocessing but app.py inference does NOT.")
    log("  Comparing accuracy WITH vs WITHOUT CLAHE:\n")

    for model_name in models_to_test:
        for ds_name in datasets:
            key_no = f"{model_name}_{ds_name}_no_clahe"
            key_yes = f"{model_name}_{ds_name}_clahe"
            key_weighted = f"{model_name}_{ds_name}_clahe_weighted"
            if key_no in all_results and key_yes in all_results:
                acc_no = all_results[key_no]['accuracy']
                acc_yes = all_results[key_yes]['accuracy']
                acc_w = all_results[key_weighted]['accuracy'] if key_weighted in all_results else None
                diff = acc_yes - acc_no
                marker = "⚠ MISMATCH" if abs(diff) > 2.0 else "✓ Minimal"
                log(f"  {model_name.upper()} | {ds_name}:")
                log(f"    Without CLAHE: {acc_no:.1f}%")
                log(f"    With CLAHE:    {acc_yes:.1f}%")
                if acc_w is not None:
                    log(f"    CLAHE+Weighted:{acc_w:.1f}%")
                log(f"    Delta:         {diff:+.1f}% [{marker}]")
                log("")

    # ── Cross-Dataset Generalization Summary ───────────────────────────────
    log("\n" + "=" * 90)
    log("  CROSS-DATASET GENERALIZATION SUMMARY")
    log("=" * 90)
    log("")

    for model_name in models_to_test:
        log(f"  Model: {model_name.upper()}")
        for ds_name in datasets:
            key = f"{model_name}_{ds_name}_clahe"
            if key in all_results:
                r = all_results[key]
                log(f"    {ds_name:12s}: {r['accuracy']:5.1f}% acc | Sens: {r['sensitivity']:.1f}% | Spec: {r['specificity']:.1f}% | Dangerous: {len(r['dangerous_misses'])} | FalseAlarm: {len(r['false_alarms'])}")
        log("")

    # ── Overall Verdict ────────────────────────────────────────────────────
    log("\n" + "=" * 90)
    log("  OVERALL VERDICT")
    log("=" * 90)

    # Check for prediction collapse (all predictions same class)
    for key, r in all_results.items():
        pred_d = r['pred_distribution']
        max_pred = max(pred_d)
        if max_pred > 0.8 * r['total']:
            dominant = pred_d.index(max_pred)
            log(f"  ⚠ PREDICTION COLLAPSE detected in {key}: {max_pred}/{r['total']} ({100*max_pred/r['total']:.0f}%) predicted as Lv{dominant} ({DR[dominant]})")

    log("\n  TEST COMPLETE")
    log("=" * 90)

    # Save results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    log(f"\n  Results saved to: {RESULTS_FILE}")

    # Save JSON for programmatic access (skip non-serializable parts)
    json_results = {}
    for key, r in all_results.items():
        json_results[key] = {
            'label': r['label'],
            'accuracy': r['accuracy'],
            'total': r['total'],
            'correct': r['correct'],
            'per_class_acc': {str(k): v for k, v in r['per_class_acc'].items()},
            'per_class_total': {str(k): v for k, v in r['per_class_total'].items()},
            'confusion_matrix': r['confusion_matrix'],
            'pred_distribution': r['pred_distribution'],
            'sensitivity': r['sensitivity'],
            'specificity': r['specificity'],
            'n_errors': r['n_errors'],
            'off_by_one': r['off_by_one'],
            'off_by_two': r['off_by_two'],
            'off_by_more': r['off_by_more'],
            'n_dangerous_misses': len(r['dangerous_misses']),
            'n_false_alarms': len(r['false_alarms']),
            'avg_conf_right': r['avg_conf_right'],
            'avg_conf_wrong': r['avg_conf_wrong'],
            'elapsed': r['elapsed'],
        }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"  JSON results saved to: {RESULTS_JSON}")


if __name__ == '__main__':
    main()

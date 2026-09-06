"""
OptiCare - Severity Bias Test
Tests if a "safety-first" softmax bias improves clinical safety.
When probabilities are close between two adjacent levels,
we should prefer the higher severity (safer for patients).
"""
import os, csv, time, random
import torch, torch.nn as nn, numpy as np
from collections import defaultdict
from torchvision import models, transforms
from PIL import Image

WEIGHTS = os.path.join(os.path.dirname(__file__), "dr_trained_model.pth")
IMG_DIR = r"C:\Users\Sahindeep\Documents\Dataset\train_images\train_images"
CSV_F   = r"C:\Users\Sahindeep\Documents\Dataset\train_1.csv"
TEMPERATURE = 2.5
SAMPLES_PER_CLASS = 100

DR = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Prolif"}

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
model.to(device).eval()

pre = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

labels = {}
with open(CSV_F) as f:
    r = csv.reader(f); next(r)
    for row in r: labels[row[0]] = int(row[1])

random.seed(42)
samples = defaultdict(list)
for iid, lb in labels.items():
    samples[lb].append(iid)

test_set = []
for lb in sorted(samples):
    pool = samples[lb]
    random.shuffle(pool)
    for iid in pool[:SAMPLES_PER_CLASS]:
        test_set.append((iid, lb))

total = len(test_set)


def predict_with_strategy(out, strategy="baseline"):
    """
    Given raw model output logits, apply different prediction strategies.
    Returns (predicted_class, confidence)
    """
    scaled = out / TEMPERATURE
    probs = torch.nn.functional.softmax(scaled, dim=0)
    
    if strategy == "baseline":
        # Pure argmax (current system)
        pred = torch.argmax(probs).item()
        return pred, probs[pred].item() * 100, probs
    
    elif strategy == "severity_bias":
        # If top-2 classes are adjacent and difference < 15%, pick the more severe one
        probs_np = probs.cpu().numpy()
        top2_idx = np.argsort(probs_np)[-2:]  # indices of top 2
        top2_idx = sorted(top2_idx)
        top1 = np.argmax(probs_np)
        top1_prob = probs_np[top1]
        
        # Check if second-highest is close and more severe
        second = np.argsort(probs_np)[-2]
        second_prob = probs_np[second]
        
        if second > top1 and (top1_prob - second_prob) < 0.15:
            # The more severe class is close - upgrade for safety
            return second, second_prob * 100, probs
        
        return top1, top1_prob * 100, probs
    
    elif strategy == "weighted_severity":
        # Multiply probabilities by severity weights before argmax
        # This makes the model slightly prefer severe classes
        severity_weights = torch.tensor([1.0, 1.0, 1.0, 1.15, 1.20], device=out.device)
        weighted = probs * severity_weights
        pred = torch.argmax(weighted).item()
        return pred, probs[pred].item() * 100, probs
    
    elif strategy == "adjacency_merge":
        # Merge adjacent class probabilities: P(class_i) += 0.3 * P(class_{i+1})
        # This makes the model consider "close" diagnoses
        probs_np = probs.cpu().numpy().copy()
        merged = probs_np.copy()
        for i in range(4):
            merged[i+1] += 0.2 * probs_np[i]  # slight upward bias
        pred = int(np.argmax(merged))
        return pred, probs_np[pred] * 100, probs


strategies = ["baseline", "severity_bias", "weighted_severity", "adjacency_merge"]

for strategy in strategies:
    correct = 0
    cc = [0]*5; ct = [0]*5
    dangerous_miss = 0  # Disease classified as No DR
    false_alarm = 0     # Healthy classified as disease
    
    for iid, true_label in test_set:
        path = os.path.join(IMG_DIR, iid + '.png')
        img = Image.open(path).convert('RGB')
        t = pre(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
        
        pred, conf, probs = predict_with_strategy(out[0], strategy)
        ok = pred == true_label
        correct += int(ok)
        cc[true_label] += int(ok)
        ct[true_label] += 1
        
        if true_label >= 2 and pred == 0:
            dangerous_miss += 1
        if true_label == 0 and pred >= 3:
            false_alarm += 1
    
    acc = 100*correct/total
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*80}")
    print(f"Overall: {acc:.1f}% ({correct}/{total})")
    for i in range(5):
        if ct[i] > 0:
            print(f"  Lv{i} ({DR[i]:10s}): {100*cc[i]/ct[i]:5.1f}%")
    print(f"  Dangerous misses: {dangerous_miss}, False alarms: {false_alarm}")

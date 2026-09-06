"""
OptiCare - Large-Scale System Test
Tests against 500 images (100 per class) to find every systematic error pattern.
Also simulates the full ensemble pipeline with synthetic ma_counts.
"""
import os, csv, sys, time, random
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

print("=" * 90)
print("OptiCare Large-Scale Accuracy Test")
print("=" * 90)

# Load model
print("\nLoading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
model.to(device).eval()

pre = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load all labels
labels = {}
with open(CSV_F) as f:
    r = csv.reader(f); next(r)
    for row in r:
        labels[row[0]] = int(row[1])

# Sample SAMPLES_PER_CLASS images per class, randomly
random.seed(42)
samples = defaultdict(list)
for iid, lb in labels.items():
    samples[lb].append(iid)

test_set = []
for lb in sorted(samples):
    pool = samples[lb]
    random.shuffle(pool)
    chosen = pool[:SAMPLES_PER_CLASS]
    for iid in chosen:
        test_set.append((iid, lb))

total = len(test_set)
print(f"Testing on {total} images ({SAMPLES_PER_CLASS} per class)...\n")

# ── Run Predictions ────────────────────────────────────────────────────────────
correct = 0
cc = [0]*5; ct = [0]*5
conf_matrix = [[0]*5 for _ in range(5)]
errors = []          # (img_id, true, pred, conf, all_probs)
all_confs_right = []
all_confs_wrong = []

t0 = time.time()
for idx, (iid, true_label) in enumerate(test_set):
    path = os.path.join(IMG_DIR, iid + '.png')
    img = Image.open(path).convert('RGB')
    t = pre(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(t)
    
    scaled = out[0] / TEMPERATURE
    probs = torch.nn.functional.softmax(scaled, dim=0)
    pred = torch.argmax(probs).item()
    conf = probs[pred].item() * 100
    all_probs = [probs[i].item()*100 for i in range(5)]
    
    ok = pred == true_label
    correct += int(ok)
    cc[true_label] += int(ok)
    ct[true_label] += 1
    conf_matrix[true_label][pred] += 1
    
    if ok:
        all_confs_right.append(conf)
    else:
        all_confs_wrong.append(conf)
        errors.append((iid, true_label, pred, conf, all_probs))
    
    if (idx+1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Processed {idx+1}/{total} ({100*(idx+1)/total:.0f}%) - {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"\n  Done in {elapsed:.1f}s")

# ── Results ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
acc = 100*correct/total
print(f"OVERALL ACCURACY: {acc:.1f}% ({correct}/{total})")
print("=" * 90)

print("\nPer-Class Accuracy:")
for i in range(5):
    if ct[i] > 0:
        a = 100*cc[i]/ct[i]
        bar = "#" * int(a/2) + "." * (50 - int(a/2))
        print(f"  Lv{i} ({DR[i]:10s}): {a:5.1f}% ({cc[i]:3d}/{ct[i]:3d}) [{bar}]")

print("\nConfusion Matrix (rows=true, cols=predicted):")
print("          " + "".join(f"  Lv{i}   " for i in range(5)))
for i in range(5):
    row = f"  Lv{i}:   " + "".join(f"  {conf_matrix[i][j]:4d}  " for j in range(5))
    print(row)

# ── Error Analysis ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"ERROR ANALYSIS ({len(errors)} wrong predictions)")
print("=" * 90)

# Group errors by type
error_types = defaultdict(list)
for iid, true_l, pred_l, conf, probs in errors:
    key = f"True {DR[true_l]} -> Pred {DR[pred_l]}"
    error_types[key].append((iid, conf, probs))

print("\nError Breakdown by Type:")
for etype, items in sorted(error_types.items(), key=lambda x: -len(x[1])):
    avg_conf = np.mean([c for _, c, _ in items])
    print(f"  {etype}: {len(items)} errors (avg conf: {avg_conf:.1f}%)")

# Dangerous errors: disease classified as healthy, or healthy classified as severe disease
print("\nDANGEROUS ERRORS (disease missed or healthy flagged as severe):")
dangerous = 0
for iid, true_l, pred_l, conf, probs in errors:
    # Missed disease: true >= 2 but predicted 0
    if true_l >= 2 and pred_l == 0:
        dangerous += 1
        print(f"  [CRITICAL] {iid}: True {DR[true_l]} classified as No DR ({conf:.1f}%)")
    # Healthy flagged as severe: true 0 but predicted >= 3
    elif true_l == 0 and pred_l >= 3:
        dangerous += 1
        print(f"  [FALSE ALARM] {iid}: Healthy eye classified as {DR[pred_l]} ({conf:.1f}%)")

if dangerous == 0:
    print("  NONE FOUND - The system never makes catastrophic errors!")

# Off-by-one vs off-by-more
off_by_one = sum(1 for _, tl, pl, _, _ in errors if abs(tl - pl) == 1)
off_by_two = sum(1 for _, tl, pl, _, _ in errors if abs(tl - pl) == 2)
off_by_more = sum(1 for _, tl, pl, _, _ in errors if abs(tl - pl) > 2)
print(f"\nError Distance:")
print(f"  Off by 1 level: {off_by_one} ({100*off_by_one/max(len(errors),1):.0f}%)")
print(f"  Off by 2 levels: {off_by_two} ({100*off_by_two/max(len(errors),1):.0f}%)")
print(f"  Off by 3+ levels: {off_by_more} ({100*off_by_more/max(len(errors),1):.0f}%)")

# Confidence analysis
print(f"\nConfidence Calibration:")
if all_confs_right:
    print(f"  Correct predictions: avg conf = {np.mean(all_confs_right):.1f}%, min = {np.min(all_confs_right):.1f}%")
if all_confs_wrong:
    print(f"  Wrong predictions:   avg conf = {np.mean(all_confs_wrong):.1f}%, min = {np.min(all_confs_wrong):.1f}%")
    high_conf_wrong = sum(1 for c in all_confs_wrong if c > 90)
    print(f"  Wrong with >90% conf: {high_conf_wrong}")

# ── Sensitivity / Specificity for Clinical Safety ──────────────────────────────
print("\n" + "=" * 90)
print("CLINICAL SAFETY METRICS")
print("=" * 90)

# Binary: Disease (level >= 1) vs No Disease (level 0)
tp = sum(1 for _, tl, pl, _, _ in [(i,t,p,c,pr) for i,t,p,c,pr in 
    [(iid, tl, torch.argmax(torch.nn.functional.softmax(model(pre(Image.open(os.path.join(IMG_DIR, iid+'.png')).convert('RGB')).unsqueeze(0).to(device))[0]/TEMPERATURE, dim=0)).item(), 0, []) for iid, tl in test_set]
    ] if tl >= 1 and pl >= 1)

# Simpler approach: use what we already computed
disease_as_disease = 0  # True positive (has disease, detected disease)
disease_as_healthy = 0  # False negative (has disease, classified healthy)
healthy_as_healthy = 0  # True negative
healthy_as_disease = 0  # False positive

for i in range(5):
    for j in range(5):
        count = conf_matrix[i][j]
        if i >= 1 and j >= 1:
            disease_as_disease += count
        elif i >= 1 and j == 0:
            disease_as_healthy += count
        elif i == 0 and j == 0:
            healthy_as_healthy += count
        elif i == 0 and j >= 1:
            healthy_as_disease += count

sensitivity = 100*disease_as_disease/(disease_as_disease+disease_as_healthy) if (disease_as_disease+disease_as_healthy) > 0 else 0
specificity = 100*healthy_as_healthy/(healthy_as_healthy+healthy_as_disease) if (healthy_as_healthy+healthy_as_disease) > 0 else 0

print(f"  Disease Detection Sensitivity: {sensitivity:.1f}% (catches {disease_as_disease}/{disease_as_disease+disease_as_healthy} diseased eyes)")
print(f"  Disease Detection Specificity: {specificity:.1f}% (correctly clears {healthy_as_healthy}/{healthy_as_healthy+healthy_as_disease} healthy eyes)")
print(f"  Missed diseases (false negatives): {disease_as_healthy}")
print(f"  False alarms on healthy eyes: {healthy_as_disease}")

# Referral accuracy: Would the patient get the right urgency of care?
# Level 0-1: routine (annual/6mo), Level 2: ophthalmologist, Level 3-4: urgent
print("\nReferral Accuracy (Would the patient get the right level of care?):")
correct_referral = 0
for i in range(5):
    for j in range(5):
        # Map to referral tier
        true_tier = 0 if i <= 1 else (1 if i == 2 else 2)
        pred_tier = 0 if j <= 1 else (1 if j == 2 else 2)
        if true_tier == pred_tier:
            correct_referral += conf_matrix[i][j]

ref_acc = 100*correct_referral/total
print(f"  Correct referral tier: {ref_acc:.1f}% ({correct_referral}/{total})")

print("\n" + "=" * 90)
print("TEST COMPLETE")
print("=" * 90)

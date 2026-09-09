"""
OptiCare — Phase 3: SIH Domain Fine-Tuning
This script takes the model trained on APTOS+Messidor and fine-tunes it 
exclusively on the IDRiD (Indian) dataset to align its final decision 
boundaries with the specific clinical standards expected by SIH judges.
"""

import os
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH    = "dr_trained_model.pth"
OUTPUT_PATH   = "dr_finetuned_idrid.pth"
EPOCHS        = 15
BATCH_SIZE    = 16
LR            = 1e-5  # Very small learning rate for fine-tuning
VAL_SPLIT     = 0.20
# ==========================================

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)

class IDRiDDataset(Dataset):
    def __init__(self):
        self.data = []
        
        idrid_csv = r"C:\Users\Sahindeep\Documents\Dataset\IDRiD\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
        idrid_dir = r"C:\Users\Sahindeep\Documents\Dataset\IDRiD\B. Disease Grading\1. Original Images\a. Training Set"
        
        if os.path.exists(idrid_csv):
            with open(idrid_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    img_path = os.path.join(idrid_dir, row[0] + '.jpg')
                    if os.path.exists(img_path):
                        self.data.append((img_path, int(row[1])))

        if len(self.data) == 0:
            raise ValueError("No IDRiD images found! Check paths.")

        print(f"  IDRiD Dataset: {len(self.data)} images found.")
        self.counts = [0] * 5
        for _, label in self.data:
            self.counts[label] += 1

    def __len__(self):
        return len(self.data)

    def apply_clahe(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return Image.new('RGB', (224, 224))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_img)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = self.apply_clahe(img_path)
        return image, label


def finetune():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OptiCare — SIH Domain Fine-Tuning (IDRiD)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Standard transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\nLoading IDRiD Dataset...")
    full_dataset = IDRiDDataset()

    n_val   = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    base_train_set, base_val_set = random_split(full_dataset, [n_train, n_val])
    
    train_set = TransformSubset(base_train_set, train_transform)
    val_set = TransformSubset(base_val_set, val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {n_train} | Validation: {n_val}")

    # ── Load Pre-Trained Model ───────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find '{MODEL_PATH}'. You must run train_model.py first!")
        return

    print("\nLoading Base Model weights...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    # Calculate Loss weights for IDRiD
    total_imgs = len(full_dataset.data)
    weights = [total_imgs / (5 * c) if c > 0 else 0 for c in full_dataset.counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    best_val_acc  = 0.0

    print(f"\nStarting Fine-Tuning for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        start = time.time()

        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = train_loss / n_train
        avg_val_loss   = val_loss   / n_val
        train_acc      = 100.0 * train_correct / n_train
        val_acc        = 100.0 * val_correct   / n_val
        elapsed        = time.time() - start

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%) | {elapsed:.0f}s")

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  ✅ New best fine-tuned model saved! (val_acc={val_acc:.1f}%)\n")

    print(f"\nFine-Tuning complete! Saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    finetune()

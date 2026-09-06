"""
OptiCare — PyTorch Training Pipeline
Fixed loopholes:
  #9: Added 80/20 train/validation split to detect overfitting
  #10: Added ReduceLROnPlateau scheduler for proper convergence
  BONUS: Added ColorJitter augmentation for robustness across different cameras
  BONUS: Added per-class accuracy reporting so you know which disease level is weakest
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# ==========================================
# CONFIGURATION — update these if needed
# ==========================================
CSV_FILE_PATH = r"C:\Users\Sahindeep\Documents\Dataset\train_1.csv"
IMG_DIR_PATH  = r"C:\Users\Sahindeep\Documents\Dataset\train_images\train_images"
OUTPUT_PATH   = "dr_trained_model.pth"
EPOCHS        = 50
BATCH_SIZE    = 16
LR            = 0.0001
VAL_SPLIT     = 0.20  # 20% held-out for validation
# ==========================================


class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        self.data      = []

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                img_path = os.path.join(img_dir, row[0] + '.png')
                if os.path.exists(img_path):
                    self.data.append((row[0], int(row[1])))

        print(f"  Dataset: {len(self.data)} valid images found.")
        counts = [0] * 5
        for _, label in self.data:
            counts[label] += 1
        print("  Class distribution:")
        for i, c in enumerate(counts):
            print(f"    Level {i}: {c} images ({100*c/len(self.data):.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data[idx][0] + '.png')
        image    = Image.open(img_name).convert('RGB')
        label    = self.data[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label


def train():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OptiCare — PyTorch Training Pipeline")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Transforms ──────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        # BONUS FIX: ColorJitter makes the model robust to different fundus camera brands
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset Loading ──────────────────────────────────────────────────────
    print("\nLoading dataset...")
    try:
        full_dataset = APTOSDataset(CSV_FILE_PATH, IMG_DIR_PATH, transform=train_transform)
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        return

    # FIX #9: Proper train/val split
    n_val   = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Apply non-augmented transform to val set
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {n_train} images | Validation: {n_val} images")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nInitializing ResNet-50 (ImageNet pre-trained)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    # ── Loss Function with Class Weights ─────────────────────────────────────
    # Weights inversely proportional to class frequency in APTOS 2019
    # Level 0: 1805 (49.5%), Level 1: 370 (10.1%), Level 2: 999 (27.4%),
    # Level 3: 193 (5.3%),  Level 4: 295 (8.1%)
    class_weights = torch.tensor([0.40, 1.97, 0.73, 3.79, 2.48]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # FIX #10: Learning rate scheduler — halves LR if val_loss plateaus for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    best_val_acc    = 0.0
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Validate — FIX #9
        model.eval()
        val_loss = 0.0
        val_correct = 0
        class_correct = [0] * 5
        class_total   = [0] * 5
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * inputs.size(0)
                preds        = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                for p, l in zip(preds, labels):
                    class_correct[l.item()] += (p == l).item()
                    class_total[l.item()]   += 1

        avg_train_loss = train_loss / n_train
        avg_val_loss   = val_loss   / n_val
        train_acc      = 100.0 * train_correct / n_train
        val_acc        = 100.0 * val_correct   / n_val
        elapsed        = time.time() - start

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%) | "
              f"{elapsed:.0f}s")

        # Per-class accuracy (helps spot which disease level the model struggles with)
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Prolif"]
        class_accs = [f"{class_names[i]}: {100*class_correct[i]/class_total[i]:.0f}%"
                      if class_total[i] > 0 else f"{class_names[i]}: N/A"
                      for i in range(5)]
        print(f"          Per-class val: {' | '.join(class_accs)}")

        # Step scheduler based on val loss
        scheduler.step(avg_val_loss)

        # Save the best model by val_acc
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  ✅ New best model saved! (val_acc={val_acc:.1f}%)\n")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.1f}%")
    print(f"  Saved to: {OUTPUT_PATH}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


if __name__ == '__main__':
    train()

import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from KNN import KNN1, KNN3, NearestCentroid


CLASSIFICATION_ROOT = './classification/'

data_list = [] 
for root, dirs, files in os.walk(CLASSIFICATION_ROOT):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            class_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
            data_list.append({
                'Filename': file,
                'Filepath': os.path.join(root, file),
                'originalClassId': class_name
            })

data_df = pd.DataFrame(data_list)
if data_df.empty:
    print("No images found. Check CLASSIFICATION_ROOT path.")
    exit()

unique_classes = sorted(data_df['originalClassId'].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
data_df['label'] = data_df['originalClassId'].map(class_to_idx)

# Load images
X_data_list = []
print("Processing images:")
for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
    img = Image.open(row['Filepath']).convert('L')  # grayscale
    img_resized = resize(np.array(img), (32, 32))   # 32x32
    X_data_list.append(img_resized)
X_data = np.array(X_data_list)
y_data = data_df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.4, random_state=42, stratify=y_data
)

# Normalize and reshape for CNN
X_train = X_train[:, np.newaxis, :, :]  # (N, 1, 32, 32)
X_test = X_test[:, np.newaxis, :, :]
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ================================
# 2. Define CNN Model
# ================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32x8x8
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ================================
# 3. Train classifiers
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(unique_classes)

classifiers = {
    "1-NN": KNN1(k=1),
    "3-NN": KNN3(k=3),
    "Nearest Centroid": NearestCentroid(),
    "CNN": SimpleCNN(num_classes).to(device)
}

results = {}

# Convert for classical classifiers
X_train_flat = X_train.numpy().reshape(len(X_train), -1)
X_test_flat = X_test.numpy().reshape(len(X_test), -1)
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    start_train = time.time()
    
    if name == "CNN":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(clf.parameters(), lr=0.001)
        clf.train()
        epochs = 10  # you can increase this
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = clf(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")
        
        end_train = time.time()

        # Evaluate
        clf.eval()
        y_pred_list = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = clf(inputs)
                preds = torch.argmax(outputs, dim=1)
                y_pred_list.extend(preds.cpu().numpy())
        y_pred = np.array(y_pred_list)
        end_test = time.time()

    else:
        # Non-CNN models use flattened PCA-free data
        clf.fit(X_train_flat, y_train_np)
        end_train = time.time()
        y_pred = clf.predict(X_test_flat)
        end_test = time.time()

    acc = accuracy_score(y_test_np, y_pred)
    results[name] = {
        "Accuracy": acc,
        "Train Time (s)": end_train - start_train,
        "Test Time (s)": end_test - end_train,
        "Predictions": y_pred
    }
    print(f"{name}: Accuracy={acc:.4f}, Train={end_train-start_train:.2f}s, Test={end_test-end_train:.2f}s")

# ================================
# 4. Show final results
# ================================
results_df = pd.DataFrame.from_dict(results, orient='index')
print("\nFINAL RESULTS")
print(results_df[['Accuracy', 'Train Time (s)', 'Test Time (s)']].to_string(float_format="%.4f"))

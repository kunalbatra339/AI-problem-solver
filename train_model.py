import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from sklearn.model_selection import train_test_split # Import for splitting data

# --- Configuration ---
# Path to your PlantVillage dataset's inner folder containing all class subfolders
# Based on your screenshot, this should be correct:
DATA_DIR = './data/PlantVillage/PlantVillage' 

# Define batch size for training and validation
BATCH_SIZE = 32

# Number of epochs for training (can be adjusted later)
NUM_EPOCHS = 10 

# Device configuration (use GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Transformations ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Load Full Dataset and Split ---
# Load the entire dataset first, without assuming train/val subfolders
full_dataset = datasets.ImageFolder(DATA_DIR, data_transforms['train']) # Use train transform for initial loading

# Get class names from the full dataset
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")
print(f"Total dataset size: {len(full_dataset)}")

# Split the dataset into training and validation sets
# We'll use 80% for training and 20% for validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Apply appropriate transforms after splitting
# This is a common practice to ensure validation set uses only validation transforms
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']


# Create data loaders
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if device.type == 'cuda' else 0),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if device.type == 'cuda' else 0)
}

# Get dataset sizes for the split sets
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

print(f"Training dataset size (after split): {dataset_sizes['train']}")
print(f"Validation dataset size (after split): {dataset_sizes['val']}")

# --- Model Setup (Transfer Learning with ResNet) ---
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("\nModel setup complete. Ready for training.")

# --- Training Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- Main execution block ---
if __name__ == '__main__':
    print("Starting model training...")
    
    os.makedirs('./models', exist_ok=True)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=NUM_EPOCHS)

    model_save_path = './models/plant_disease_resnet18.pth'
    torch.save(model_ft.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    with open('./models/class_names.json', 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to ./models/class_names.json")

    print("Training script finished.")

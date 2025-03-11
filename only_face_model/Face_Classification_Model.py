import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Define grid size
GRID_COLS = 24  # 1920/80 = 24 cells horizontally
GRID_ROWS = 14  # 1080/80 ≈ 14 cells vertically
NUM_CLASSES = GRID_ROWS * GRID_COLS

class EyeGazeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.gaze_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

        # Cell dimensions
        self.cell_width = 1920 / GRID_COLS
        self.cell_height = 1080 / GRID_ROWS

    def __len__(self):
        return len(self.gaze_frame)

    def get_cell_index(self, x, y):
        # Convert x,y coordinates to grid cell index
        col = min(int(x / self.cell_width), GRID_COLS - 1)
        row = min(int(y / self.cell_height), GRID_ROWS - 1)
        return row * GRID_COLS + col

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get face image path
        img_name = os.path.join(self.img_dir, os.path.basename(self.gaze_frame.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')

        # Get coordinates and convert to cell index
        x = self.gaze_frame.iloc[idx]['x']
        y = self.gaze_frame.iloc[idx]['y']
        cell_idx = self.get_cell_index(x, y)

        # Convert to one-hot encoding
        target = torch.zeros(NUM_CLASSES)
        target[cell_idx] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, target

class GazeNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GazeNet, self).__init__()

        # Use EfficientNet as backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Remove the original classifier
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # New classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            _, target_class = torch.max(labels.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target_class.cpu().numpy())

    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_targets, all_preds)

    model.train()
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, schedulers, device, num_epochs=50):
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        num_batches = 0

        # Training phase
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            schedulers.step()

            running_loss += loss.item()
            num_batches += 1

            if i % 10 == 9:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.5f}')
                running_loss = 0.0

        # Validation phase
        val_loss, accuracy = validate_model(model, val_loader, criterion, device)
        train_avg_loss = running_loss / num_batches

        train_losses.append(train_avg_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        epoch_time = time.time() - epoch_start_time

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_avg_loss:.5f}')
        print(f'Validation Loss: {val_loss:.5f}')
        print(f'Accuracy: {accuracy:.2%}')
        print(f'Epoch Time: {epoch_time:.2f} seconds')
        print('-' * 60)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy,
                'grid_cols': GRID_COLS,
                'grid_rows': GRID_ROWS
            }, 'best_gaze_model.pth')

            # Plot current predictions
            if epoch % 5 == 0:
                plot_predictions(model, val_loader, device, epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    return train_losses, val_losses, accuracies

def plot_predictions(model, val_loader, device, epoch):
    model.eval()
    all_preds = []
    all_targets = []

    # Create screen grid
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cell_width = 1920 // GRID_COLS
    cell_height = 1080 // GRID_ROWS

    # Draw grid lines
    for i in range(GRID_ROWS):
        y = i * cell_height
        cv2.line(screen, (0, y), (1920, y), (50, 50, 50), 1)
    for i in range(GRID_COLS):
        x = i * cell_width
        cv2.line(screen, (x, 0), (x, 1080), (50, 50, 50), 1)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, target_class = torch.max(labels.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target_class.cpu().numpy())

    # Plot heatmap of predictions vs targets
    plt.figure(figsize=(20, 8))

    # Plot targets
    plt.subplot(1, 2, 1)
    target_heatmap = np.zeros((GRID_ROWS, GRID_COLS))
    for t in all_targets:
        row = t // GRID_COLS
        col = t % GRID_COLS
        target_heatmap[row, col] += 1
    plt.imshow(target_heatmap, cmap='hot')
    plt.title('Target Gaze Distribution')
    plt.colorbar()

    # Plot predictions
    plt.subplot(1, 2, 2)
    pred_heatmap = np.zeros((GRID_ROWS, GRID_COLS))
    for p in all_preds:
        row = p // GRID_COLS
        col = p % GRID_COLS
        pred_heatmap[row, col] += 1
    plt.imshow(pred_heatmap, cmap='hot')
    plt.title('Predicted Gaze Distribution')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'predictions_epoch_{epoch + 1}.png')
    plt.close()

def main():
    # Set device and seeds for reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = EyeGazeDataset(
        csv_file=r'C:\\Users\\orlan\\OneDrive\\Desktop\\2semestre\\AI COMPUTER VISION\\code\\project\\eye_traking\\data_collection_phase\\data\\only_face.csv',
        img_dir=r'C:\\Users\\orlan\\OneDrive\\Desktop\\2semestre\\AI COMPUTER VISION\\code\\project\\eye_traking\\data_collection_phase\\data\\only_face',
        transform=train_transform
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Override transforms
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model and training components
    model = GazeNet(pretrained=True).to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')  # Good for training with soft targets
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Learning rate scheduling
    num_steps = len(train_loader) * 50
    warmup_steps = len(train_loader) * 5

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps-warmup_steps, eta_min=1e-6)
    schedulers = SequentialLR(optimizer,
                            schedulers=[warmup_scheduler, main_scheduler],
                            milestones=[warmup_steps])

    print("Starting training...")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"Grid size: {GRID_ROWS}x{GRID_COLS} ({NUM_CLASSES} classes)")

    train_losses, val_losses, accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, schedulers, device, num_epochs=50
    )

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Classification Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    print("Training completed! Best model saved as 'best_gaze_model.pth'")

if __name__ == "__main__":
    main()
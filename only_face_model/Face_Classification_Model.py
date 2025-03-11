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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Define screen grid configuration
class ScreenGrid:
    def __init__(self, screen_width=1920, screen_height=1080, n_cols=12, n_rows=7):  # Reduced grid size
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.cell_width = screen_width / n_cols
        self.cell_height = screen_height / n_rows
        self.n_cells = n_cols * n_rows

    def get_cell_index(self, x, y):
        """Convert screen coordinates to cell index"""
        col = min(int(x / self.cell_width), self.n_cols - 1)
        row = min(int(y / self.cell_height), self.n_rows - 1)
        return row * self.n_cols + col

    def get_cell_center(self, cell_idx):
        """Get center coordinates of a cell"""
        row = cell_idx // self.n_cols
        col = cell_idx % self.n_cols
        x = (col + 0.5) * self.cell_width
        y = (row + 0.5) * self.cell_height
        return x, y

    def draw_grid(self):
        """Create a visualization of the grid"""
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Draw vertical lines
        for i in range(self.n_cols):
            x = int(i * self.cell_width)
            cv2.line(screen, (x, 0), (x, self.screen_height), (50, 50, 50), 1)

        # Draw horizontal lines
        for i in range(self.n_rows):
            y = int(i * self.cell_height)
            cv2.line(screen, (0, y), (self.screen_width, y), (50, 50, 50), 1)

        return screen

class GazeDataset(Dataset):
    def __init__(self, csv_file, img_dir, screen_grid, transform=None, train=True):
        self.gaze_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.screen_grid = screen_grid
        self.train = train  # Add train flag for different augmentations

    def __len__(self):
        return len(self.gaze_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load face image
        img_name = os.path.join(self.img_dir, os.path.basename(self.gaze_frame.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')

        # Get gaze coordinates and convert to cell index
        x = self.gaze_frame.iloc[idx]['x']
        y = self.gaze_frame.iloc[idx]['y']
        cell_idx = self.screen_grid.get_cell_index(x, y)

        if self.transform:
            image = self.transform(image)

        # Return cell index directly instead of one-hot encoding
        return image, cell_idx, torch.tensor([x, y])

class GazeClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(GazeClassifier, self).__init__()

        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Modify classifier head with more regularization
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Increased dropout
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if i % 10 == 9:
            print(f'Batch {i + 1}, Loss: {running_loss / 10:.3f}, Accuracy: {100 * correct / total:.2f}%')
            running_loss = 0.0

    return correct / total

def validate(model, val_loader, criterion, device, screen_grid):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    pred_coords = []
    true_coords = []

    with torch.no_grad():
        for inputs, targets, coords in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            _, true_class = torch.max(targets.data, 1)

            # Calculate accuracy
            total += true_class.size(0)
            correct += (predicted == true_class).sum().item()

            # Store predictions and targets for analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(true_class.cpu().numpy())

            # Convert predictions to screen coordinates
            for pred_idx, true_idx in zip(predicted.cpu().numpy(), true_class.cpu().numpy()):
                pred_x, pred_y = screen_grid.get_cell_center(pred_idx)
                pred_coords.append([pred_x, pred_y])
                true_x, true_y = screen_grid.get_cell_center(true_idx)
                true_coords.append([true_x, true_y])

    accuracy = correct / total
    avg_loss = val_loss / len(val_loader)

    return avg_loss, accuracy, np.array(pred_coords), np.array(true_coords)

def plot_results(pred_coords, true_coords, screen_grid, epoch):
    plt.figure(figsize=(15, 10))

    # Plot actual vs predicted points
    plt.subplot(1, 2, 1)
    plt.scatter(true_coords[:, 0], true_coords[:, 1], c='blue', alpha=0.5, label='True', s=10)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', alpha=0.5, label='Predicted', s=10)
    plt.title('Gaze Predictions vs Ground Truth')
    plt.xlabel('Screen Width (pixels)')
    plt.ylabel('Screen Height (pixels)')
    plt.legend()
    plt.grid(True)

    # Plot heatmap of prediction errors
    plt.subplot(1, 2, 2)
    errors = np.sqrt(np.sum((pred_coords - true_coords)**2, axis=1))
    plt.hist(errors, bins=50)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Count')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results_epoch_{epoch}.png')
    plt.close()

def main():
    # Initialize screen grid with coarser resolution
    screen_grid = ScreenGrid(n_cols=12, n_rows=7)  # ~160x154 pixel cells

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = GazeDataset(
        csv_file=r'C:\Users\orlan\OneDrive\Desktop\2semestre\AI COMPUTER VISION\code\project\eye_traking\data_collection_phase\data\only_face.csv',
        img_dir=r'C:\Users\orlan\OneDrive\Desktop\2semestre\AI COMPUTER VISION\code\project\eye_traking\data_collection_phase\data\only_face',
        screen_grid=screen_grid,
        transform=train_transform,
        train=True
    )

    val_dataset = GazeDataset(
        csv_file=r'C:\Users\orlan\OneDrive\Desktop\2semestre\AI COMPUTER VISION\code\project\eye_traking\data_collection_phase\data\only_face.csv',
        img_dir=r'C:\Users\orlan\OneDrive\Desktop\2semestre\AI COMPUTER VISION\code\project\eye_traking\data_collection_phase\data\only_face',
        screen_grid=screen_grid,
        transform=val_transform,
        train=False
    )

    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders with larger batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize model
    model = GazeClassifier(num_classes=screen_grid.n_cells, pretrained=True).to(device)

    # Use CrossEntropyLoss instead of KLDivLoss
    criterion = nn.CrossEntropyLoss()

    # Modified optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Training loop
    num_epochs = 50
    best_accuracy = 0
    patience = 8  # Reduced patience for faster feedback
    patience_counter = 0

    print(f"Starting training... Grid size: {screen_grid.n_cols}x{screen_grid.n_rows} ({screen_grid.n_cells} classes)")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, pred_coords, true_coords = validate(model, val_loader, criterion, device, screen_grid)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Plot results every 5 epochs
        if epoch % 5 == 0:
            plot_results(pred_coords, true_coords, screen_grid, epoch)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'grid_config': {
                    'n_cols': screen_grid.n_cols,
                    'n_rows': screen_grid.n_rows
                }
            }, 'best_gaze_classifier.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    print("Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
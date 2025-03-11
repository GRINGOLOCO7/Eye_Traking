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
from sklearn.metrics import mean_absolute_error, r2_score

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 640, kernel_size=3, padding=1),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(640, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention

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

        # Calculate screen center
        self.center_x = 1920 / 2
        self.center_y = 1080 / 2

        # Calculate max distance from center for normalization
        self.max_dist = np.sqrt(1920**2 + 1080**2) / 2

    def __len__(self):
        return len(self.gaze_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get face image path
        img_name = os.path.join(self.img_dir, os.path.basename(self.gaze_frame.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')

        # Get coordinates and normalize relative to screen center
        x = self.gaze_frame.iloc[idx]['x']
        y = self.gaze_frame.iloc[idx]['y']

        # Convert to relative coordinates from center
        x_rel = (x - self.center_x) / self.max_dist
        y_rel = (y - self.center_y) / self.max_dist

        gaze_point = torch.FloatTensor([x_rel, y_rel])

        if self.transform:
            image = self.transform(image)

        return image, gaze_point

class ImprovedGazeNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ImprovedGazeNet, self).__init__()

        # Use EfficientNet as backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Remove the original classifier
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Add spatial attention
        self.attention = SpatialAttention()

        # New head for gaze prediction
        self.gaze_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Tanh()  # Output relative coordinates (-1 to 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.gaze_head(x)
        return x

class GazeLoss(nn.Module):
    def __init__(self):
        super(GazeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred, target):
        # Combined loss
        mse_loss = self.mse(pred, target)
        smooth_l1_loss = self.smooth_l1(pred, target)

        # Add directional consistency loss
        direction_pred = torch.nn.functional.normalize(pred, dim=1)
        direction_target = torch.nn.functional.normalize(target, dim=1)
        direction_loss = 1 - torch.mean(torch.sum(direction_pred * direction_target, dim=1))

        return mse_loss + 0.5 * smooth_l1_loss + 0.3 * direction_loss

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

            # Store predictions and targets for analysis
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches
    model.train()

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate additional metrics
    mae = np.mean(np.abs(all_preds - all_targets))

    return avg_loss, mae

def train_model(model, train_loader, val_loader, criterion, optimizer, schedulers, device, num_epochs=50):
    best_val_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_maes = []

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
        val_loss, val_mae = validate_model(model, val_loader, criterion, device)
        train_avg_loss = running_loss / num_batches

        train_losses.append(train_avg_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        epoch_time = time.time() - epoch_start_time

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_avg_loss:.5f}')
        print(f'Validation Loss: {val_loss:.5f}')
        print(f'Validation MAE: {val_mae:.5f}')
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
                'center_x': train_loader.dataset.dataset.center_x if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.center_x,
                'center_y': train_loader.dataset.dataset.center_y if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.center_y,
                'max_dist': train_loader.dataset.dataset.max_dist if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.max_dist
            }, 'best_gaze_model.pth')

            # Plot current predictions
            if epoch % 5 == 0:
                plot_predictions(model, val_loader, device, epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    return train_losses, val_losses, val_maes

def plot_predictions(model, val_loader, device, epoch):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    plt.figure(figsize=(10, 10))
    plt.scatter(all_targets[:, 0], all_targets[:, 1], c='blue', label='Target', alpha=0.5)
    plt.scatter(all_preds[:, 0], all_preds[:, 1], c='red', label='Predicted', alpha=0.5)
    plt.title(f'Predictions vs Targets - Epoch {epoch + 1}')
    plt.legend()
    plt.grid(True)
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
    model = ImprovedGazeNet(pretrained=True).to(device)
    criterion = GazeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Learning rate scheduling
    num_steps = len(train_loader) * 50
    warmup_steps = len(train_loader) * 5

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps-warmup_steps, eta_min=1e-6)
    schedulers = SequentialLR(optimizer,
                            schedulers=[warmup_scheduler, main_scheduler],
                            milestones=[warmup_steps])

    # Train the model
    print("Starting training...")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

    train_losses, val_losses, val_maes = train_model(
        model, train_loader, val_loader, criterion, optimizer, schedulers, device, num_epochs=50
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_maes, label='Validation MAE')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    print("Training completed! Best model saved as 'best_gaze_model.pth'")

if __name__ == "__main__":
    main()
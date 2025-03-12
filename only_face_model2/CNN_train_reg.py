import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os

class GazeDataset(Dataset):
    """
    Custom Dataset for loading gaze tracking data.
    Similar to the approach used in https://github.com/pperle/gaze-tracking-pipeline
    but simplified to work with just face images instead of separate eye patches.

    The dataset loads face images and their corresponding gaze coordinates.
    The coordinates are normalized to [0,1] range to make training more stable.
    """
    def __init__(self, csv_file, transform=None):
        """
        Initialize the dataset.
        Args:
            csv_file (string): Path to the CSV file containing:
                - Column 1: Image file paths
                - Column 2: X coordinates (in screen pixels)
                - Column 3: Y coordinates (in screen pixels)
            transform (callable, optional): Transform to be applied to the images
        """
        self.gaze_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.gaze_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct the full image path
        img_path = os.path.join('data_collection_phase/data/only_face',
                               self.gaze_frame.iloc[idx, 0])

        # Load and convert image to RGB (some images might be grayscale)
        image = Image.open(img_path).convert('RGB')

        # Normalize coordinates to [0,1] range
        # This is similar to the approach in the reference project's utils.py
        # where they normalize coordinates for better training stability
        x = self.gaze_frame.iloc[idx, 1] / 1920.0  # Normalize by screen width
        y = self.gaze_frame.iloc[idx, 2] / 1050.0  # Normalize by screen height

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([x, y], dtype=torch.float32)

class CustomGazeLoss(nn.Module):
    """
    Enhanced loss function for gaze prediction that combines:
    1. Separate MSE for x and y coordinates
    2. L1 loss for absolute differences
    3. Relative position loss to maintain spatial relationships
    """
    def __init__(self):
        super(CustomGazeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Separate coordinate losses
        mse_x = self.mse(pred[:, 0], target[:, 0])  # MSE for x coordinates
        mse_y = self.mse(pred[:, 1], target[:, 1])  # MSE for y coordinates

        # L1 loss for absolute differences
        l1_loss = self.l1(pred, target)

        # Relative position loss
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        relative_loss = self.mse(pred_diff, target_diff)

        # Combine losses with weights
        total_loss = (mse_x + mse_y) * 0.4 + l1_loss * 0.4 + relative_loss * 0.2

        return total_loss

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()

        # Convolutional layers with batch normalization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Match kernel size 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # Match kernel size 5x5
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Attention mechanism (1x1 convolution style)
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),  # Changed to Conv2d to match saved model
            nn.Sigmoid()
        )

        # Regression layers - matching exact saved model structure
        self.regressor = nn.Sequential(
            nn.ReLU(),  # index 0
            nn.Dropout(0.5),  # index 1
            nn.Linear(512, 256),  # index 2
            nn.ReLU(),  # index 3
            nn.Dropout(0.5),  # index 4
            nn.Linear(256, 64),  # index 5
            nn.ReLU(),  # index 6
            nn.Dropout(0.3),  # index 7
            nn.Linear(64, 2),  # index 8
            nn.Sigmoid()  # index 9
        )

    def forward(self, x):
        # Feature extraction
        features = self.conv_layers(x)

        # Apply attention (keeping spatial dimensions)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Global average pooling and flatten
        features = attended_features.view(attended_features.size(0), -1)

        # Regression
        coordinates = self.regressor(features)

        return coordinates

def train_model(model, train_loader, val_loader, num_epochs=100):
    """
    Training function with validation and model checkpointing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = CustomGazeLoss()

    # Use AdamW with a smaller learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

    # Use ReduceLROnPlateau instead of cosine annealing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'only_face_model2/best_reg_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

def main():
    """
    Main function to set up and run the training pipeline.
    """
    # Image transformations
    # Using standard ImageNet normalization values
    transform = transforms.Compose([
        transforms.Resize((190, 190)),  # Resize to match our model's input size
        transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = GazeDataset(csv_file='data_collection_phase/data/only_face.csv',
                         transform=transform)

    # Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders with batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize and train the model
    model = GazeNet()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

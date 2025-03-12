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
    Custom loss function for gaze prediction that combines:
    1. MSE for absolute position accuracy
    2. Cosine similarity for directional accuracy
    3. Relative distance preservation
    """
    def __init__(self):
        super(CustomGazeLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = self.mse(pred, target)

        # Directional loss using cosine similarity
        pred_normalized = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-7)
        target_normalized = target / (torch.norm(target, dim=1, keepdim=True) + 1e-7)
        cos_loss = 1 - torch.mean(torch.sum(pred_normalized * target_normalized, dim=1))

        # Combine losses with weights
        total_loss = mse_loss + 0.5 * cos_loss

        return total_loss

class GazeNet(nn.Module):
    """
    Neural network for gaze prediction.
    Inspired by the architecture in https://github.com/pperle/gaze-tracking
    but modified to work with full face images instead of separate eye patches.

    The network consists of:
    1. Convolutional layers for feature extraction
    2. Batch normalization for training stability
    3. Dropout for regularization
    4. Fully connected layers for coordinate regression
    """
    def __init__(self):
        super(GazeNet, self).__init__()

        # Increase model capacity
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # Larger kernel for better feature capture
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Add residual connections
        self.skip_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.skip_conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.skip_conv3 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.dropout = nn.Dropout(0.5)  # Increased dropout

        # Larger fully connected layers
        self.fc1 = nn.Linear(512 * 12 * 12, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Add tanh for better coordinate prediction

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor of shape (batch_size, 3, 190, 190)
        Returns:
            Predicted gaze coordinates normalized to [0,1] range
        """
        # First conv block
        x1 = self.relu(self.bn1(self.conv1(x)))

        # Second conv block with skip connection
        x2_main = self.conv2(x1)
        x2_skip = self.skip_conv1(x1)
        x2 = self.relu(self.bn2(x2_main + x2_skip))

        # Third conv block with skip connection
        x3_main = self.conv3(x2)
        x3_skip = self.skip_conv2(x2)
        x3 = self.relu(self.bn3(x3_main + x3_skip))

        # Fourth conv block with skip connection
        x4_main = self.conv4(x3)
        x4_skip = self.skip_conv3(x3)
        x4 = self.relu(self.bn4(x4_main + x4_skip))

        # Flatten and fully connected layers
        x = x4.view(x4.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.tanh(self.fc3(x))  # Use tanh to bound outputs to [-1, 1]

        # Scale from [-1, 1] to [0, 1]
        x = (x + 1) / 2

        return x

def train_model(model, train_loader, val_loader, num_epochs=50):
    """
    Training function with validation and model checkpointing.
    Similar to the training approach in the reference project but with added
    learning rate scheduling and validation monitoring.

    Args:
        model: The GazeNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use custom loss function
    criterion = CustomGazeLoss()

    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Cosine annealing scheduler for better optimization
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience = 10
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

            # Gradient clipping to prevent exploding gradients
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

        # Learning rate scheduling
        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # Early stopping with patience
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

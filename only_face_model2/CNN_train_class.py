import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os

# Define grid size
GRID_COLS = 16  # Divides 1920 into ~120px cells
GRID_ROWS = 9   # Divides 1050 into ~117px cells
NUM_CLASSES = GRID_ROWS * GRID_COLS  # Total number of grid cells

class GazeDatasetGrid(Dataset):
    """
    Custom Dataset for loading gaze tracking data with grid-based classification.
    Instead of regressing exact coordinates, we classify which grid cell
    the user is looking at.
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

        # Calculate cell dimensions
        self.cell_width = 1920 // GRID_COLS
        self.cell_height = 1050 // GRID_ROWS

    def __len__(self):
        return len(self.gaze_frame)

    def get_grid_cell(self, x, y):
        """
        Convert screen coordinates to grid cell index.
        Returns the class index (0 to NUM_CLASSES-1) for the given coordinates.
        """
        # Calculate grid position
        col = min(x // self.cell_width, GRID_COLS - 1)
        row = min(y // self.cell_height, GRID_ROWS - 1)

        # Convert to class index
        return int(row * GRID_COLS + col)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = os.path.join('data_collection_phase/data/only_face',
                               self.gaze_frame.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        # Get coordinates and convert to grid cell
        x = self.gaze_frame.iloc[idx, 1]
        y = self.gaze_frame.iloc[idx, 2]
        grid_cell = self.get_grid_cell(x, y)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(grid_cell, dtype=torch.long)

class GazeNetClassification(nn.Module):
    """
    Neural network for gaze grid prediction.
    Similar to the regression model but modified for classification:
    - Changed final layer to output class probabilities
    - Added softmax activation
    - Increased capacity slightly for the classification task
    """
    def __init__(self):
        super(GazeNetClassification, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)    # 190x190 -> 95x95
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # -> 48x48
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # -> 24x24
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # -> 12x12

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)  # Increased dropout for classification

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, NUM_CLASSES)  # Output one score per grid cell

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor of shape (batch_size, 3, 190, 190)
        Returns:
            Class scores for each grid cell
        """
        # Convolutional feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Logits for each class

        return x

def train_model(model, train_loader, val_loader, num_epochs=50):
    """
    Training function with validation and model checkpointing.
    Modified for classification with:
    - CrossEntropyLoss instead of MSE
    - Accuracy metric tracking
    - Learning rate scheduling based on accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Cross Entropy Loss for classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with learning rate scheduling
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler based on accuracy instead of loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Adjust learning rate based on validation accuracy
        scheduler.step(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'only_face_model2/best_class_model.pth')

def main():
    """
    Main function to set up and run the training pipeline.
    """
    transform = transforms.Compose([
        transforms.Resize((190, 190)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = GazeDatasetGrid(csv_file='data_collection_phase/data/only_face.csv',
                            transform=transform)

    # Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = GazeNetClassification()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
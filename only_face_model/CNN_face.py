import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

# Custom Dataset class for eye gaze data
class EyeGazeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.gaze_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Normalize target coordinates to [0, 1] range
        self.x_max = self.gaze_frame['x'].max()
        self.x_min = self.gaze_frame['x'].min()
        self.y_max = self.gaze_frame['y'].max()
        self.y_min = self.gaze_frame['y'].min()

    def __len__(self):
        return len(self.gaze_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get face image path (first column)
        img_name = os.path.join(self.img_dir, os.path.basename(self.gaze_frame.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')

        # Get and normalize coordinates
        x = (self.gaze_frame.iloc[idx]['x'] - self.x_min) / (self.x_max - self.x_min)
        y = (self.gaze_frame.iloc[idx]['y'] - self.y_min) / (self.y_max - self.y_min)
        gaze_point = torch.FloatTensor([x, y])

        if self.transform:
            image = self.transform(image)

        return image, gaze_point

# CNN Model
class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block - input: 190x190x3
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 95x95x32

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 47x47x64

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 23x23x128
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 23 * 23, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Sigmoid()  # To ensure output is between 0 and 1
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}')
                running_loss = 0.0

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((190, 190)),  # Keep original size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader with actual paths
    dataset = EyeGazeDataset(
        csv_file=r'C:\\Users\\orlan\\OneDrive\\Desktop\\2semestre\\AI COMPUTER VISION\\code\\project\\eye_traking\\data_collection_phase\\data\\eye_data.csv',
        img_dir=r'C:\\Users\\orlan\\OneDrive\\Desktop\\2semestre\\AI COMPUTER VISION\\code\\project\\eye_traking\\data_collection_phase\\data\\saved_images',
        transform=transform
    )

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model, criterion, and optimizer
    model = GazeCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    train_model(model, train_loader, criterion, optimizer, device)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'x_min': dataset.x_min,
        'x_max': dataset.x_max,
        'y_min': dataset.y_min,
        'y_max': dataset.y_max
    }, 'gaze_model.pth')

    print("Training completed! Model saved as 'gaze_model.pth'")

if __name__ == "__main__":
    main()

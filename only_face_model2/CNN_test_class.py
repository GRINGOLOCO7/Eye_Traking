import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
from CNN_train_class import GazeNetClassification, GRID_COLS, GRID_ROWS

def load_model(model_path='only_face_model2/best_class_model.pth'):
    """Load the trained classification model."""
    model = GazeNetClassification()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_cell_center(cell_idx):
    """Convert cell index to screen coordinates (cell center)."""
    cell_width = 1920 // GRID_COLS
    cell_height = 1050 // GRID_ROWS

    row = cell_idx // GRID_COLS
    col = cell_idx % GRID_COLS

    x = (col * cell_width) + (cell_width // 2)
    y = (row * cell_height) + (cell_height // 2)

    return x, y

def visualize_predictions(model, test_csv, num_samples=50):
    """
    Visualize actual vs predicted gaze points on the screen.
    Also shows the grid structure and highlights prediction errors.
    """
    # Data loading setup
    transform = transforms.Compose([
        transforms.Resize((190, 190)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load test data
    df = pd.read_csv(test_csv)

    # Randomly sample if more than num_samples
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)

    # Prepare figure
    plt.figure(figsize=(15, 8))

    # Draw grid
    cell_width = 1920 // GRID_COLS
    cell_height = 1050 // GRID_ROWS
    for i in range(GRID_COLS + 1):
        plt.axvline(x=i * cell_width, color='gray', alpha=0.2)
    for i in range(GRID_ROWS + 1):
        plt.axhline(y=i * cell_height, color='gray', alpha=0.2)

    actual_points = []
    predicted_points = []

    # Make predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for _, row in df.iterrows():
        # Load and process image
        img_path = os.path.join('data_collection_phase/data/only_face', row['File Path Face Image'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get actual coordinates
        actual_x, actual_y = row['x'], row['y']
        actual_points.append((actual_x, actual_y))

        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_cell = output.max(1)[1].item()
            pred_x, pred_y = get_cell_center(predicted_cell)
            predicted_points.append((pred_x, pred_y))

            # Draw line connecting actual and predicted points
            plt.plot([actual_x, pred_x], [actual_y, pred_y], 'gray', alpha=0.3)

    # Plot points
    actual_points = np.array(actual_points)
    predicted_points = np.array(predicted_points)

    plt.scatter(actual_points[:, 0], actual_points[:, 1],
               c='green', label='Actual', alpha=0.6)
    plt.scatter(predicted_points[:, 0], predicted_points[:, 1],
               c='red', label='Predicted', alpha=0.6)

    # Calculate and display average error
    errors = np.sqrt(np.sum((actual_points - predicted_points) ** 2, axis=1))
    avg_error = np.mean(errors)
    plt.title(f'Actual vs Predicted Gaze Points on Screen\nAverage Error: {avg_error:.2f} pixels')

    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.savefig('classification_results.png')
    plt.close()

    # Print statistics
    print(f"Average Error: {avg_error:.2f} pixels")
    print(f"Standard Deviation of Error: {np.std(errors):.2f} pixels")
    print(f"Maximum Error: {np.max(errors):.2f} pixels")
    print(f"Minimum Error: {np.min(errors):.2f} pixels")

    # Calculate accuracy within different thresholds
    for threshold in [100, 200, 300]:
        accuracy = np.mean(errors < threshold) * 100
        print(f"Accuracy within {threshold} pixels: {accuracy:.2f}%")

def main():
    # Load model
    model = load_model()

    # Test and visualize
    visualize_predictions(model, 'data_collection_phase/data/only_face.csv')

if __name__ == "__main__":
    main()
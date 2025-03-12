import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
from CNN_train_reg import GazeNet

def load_model(model_path='best_reg_model.pth'):
    """Load the trained regression model."""
    model = GazeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def denormalize_coordinates(x, y):
    """Convert normalized coordinates back to screen coordinates."""
    return x * 1920, y * 1050

def visualize_predictions(model, test_csv, num_samples=50):
    """
    Visualize actual vs predicted gaze points on the screen.
    Shows prediction errors with connecting lines.
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
            pred_x, pred_y = denormalize_coordinates(output[0][0].item(), output[0][1].item())
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
    plt.savefig('regression_results.png')
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

    # Additional regression-specific metrics
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Calculate separate errors for X and Y coordinates
    x_errors = np.abs(actual_points[:, 0] - predicted_points[:, 0])
    y_errors = np.abs(actual_points[:, 1] - predicted_points[:, 1])
    print(f"\nAverage X Error: {np.mean(x_errors):.2f} pixels")
    print(f"Average Y Error: {np.mean(y_errors):.2f} pixels")

def main():
    # Load model
    model = load_model()

    # Test and visualize
    visualize_predictions(model, 'data_collection_phase/data/only_face.csv')

if __name__ == "__main__":
    main()
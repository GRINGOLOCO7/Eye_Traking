import cv2
import numpy as np
import pyautogui
import os
import dlib
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import time
import seaborn as sns
import pandas as pd
import random

# Import your model
import sys
import os
script_dir = os.path.abspath("../..")
sys.path.append(script_dir)
from model.model import *

# Load model
model = ResNetModel(num_classes=100)
model.load_state_dict(torch.load('../../train/resnet_model.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((30, 155)),
    transforms.ToTensor()
])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define screen and grid properties
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
GRID_COLS, GRID_ROWS = 10, 10
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT // GRID_ROWS
CORP_EYE_WIDTH, CORP_EYE_HEIGHT = 155, 30

# Create a black screen-sized image
grid_img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

# Load facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../collect_data/shape_predictor_68_face_landmarks.dat')

# Initialize error collection dictionary
# Structure: {cell_number: [error1, error2]}
errors_collection = {i: [] for i in range(1, 101)}  # Pre-initialize all cells


def draw_square(n,grid_img=grid_img,color=(0, 0, 255)):
    """Draw a square on the grid at position n"""
    row = (n - 1) // GRID_COLS
    col = (n - 1) % GRID_COLS
    x, y = col * GRID_WIDTH, row * GRID_HEIGHT
    img = grid_img.copy()
    cv2.rectangle(img, (x, y), (x + GRID_WIDTH, y + GRID_HEIGHT), color, 2)
    return img


def get_eye_region():
    """Capture and process eye region from webcam"""
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        print("No face detected")
        return None

    # Use the first detected face
    face = faces[0]
    landmarks = predictor(gray, face)

    # Extract eye coordinates
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    # Calculate eye center
    eye_center_x = (left_eye[0] + right_eye[0]) // 2
    eye_center_y = (left_eye[1] + right_eye[1]) // 2

    # Crop eye region
    eyes_region = frame[
        max(0, eye_center_y - 15):min(frame.shape[0], eye_center_y + 15),
        max(0, eye_center_x - 77):min(frame.shape[1], eye_center_x + 77)
    ]

    # Check if we got a valid crop
    if eyes_region.size == 0 or eyes_region.shape[0] == 0 or eyes_region.shape[1] == 0:
        print("Invalid eye region crop")
        return None

    # Resize to expected dimensions
    eyes_region = cv2.resize(eyes_region, (CORP_EYE_WIDTH, CORP_EYE_HEIGHT))

    # Display for debugging
    cv2.imshow("Eyes Region", eyes_region)

    return eyes_region


def get_prediction(eye_image):
    """Get model prediction from eye image"""
    # Convert to RGB (PIL format)
    pil_image = Image.fromarray(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB))

    # Transform and prepare for model
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()


def collect_errors():
    """Run two rounds of error collection for all cells"""
    # Create a list of all cell numbers 1-100 for each round
    cells_round1 = list(range(1, 101))
    cells_round2 = list(range(1, 101))

    # Shuffle to randomize the order
    random.shuffle(cells_round1)
    random.shuffle(cells_round2)

    # Round 1
    print("Starting Round 1...")
    for cell_number in cells_round1:
        collect_single_cell_error(cell_number, 1)

        # Check for early exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Short break between rounds
    print("\nRound 1 completed. Taking a short break before Round 2...\n")
    time.sleep(3)

    # Round 2
    print("Starting Round 2...")
    for cell_number in cells_round2:
        collect_single_cell_error(cell_number, 2)

        # Check for early exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def collect_single_cell_error(cell_number, round_num):
    """Collect error for a single cell"""
    # Display the grid cell
    grid = draw_square(cell_number)
    cv2.imshow("Grid", grid)
    print(f"Round {round_num}: Look at cell {cell_number}")

    # Give time to look at the cell
    time.sleep(2)

    # Capture eye region
    eye_region = get_eye_region()
    if eye_region is None:
        print(f"Failed to capture eye region for cell {cell_number}")
        return

    # Get prediction
    predicted = get_prediction(eye_region)
    grid = draw_square(predicted, grid, (0, 255, 0))
    cv2.imshow("Grid", grid)

    # Calculate error
    error = predicted - cell_number

    # Store error
    errors_collection[cell_number].append(error)

    print(f"Cell: {cell_number}, Predicted: {predicted}, Error: {error}")

    # Short pause before next cell
    time.sleep(2)


def visualize_errors():
    """Create a heatmap visualization of the errors"""
    # Prepare data for visualization
    error_matrix = np.zeros((GRID_ROWS, GRID_COLS))

    for cell in range(1, 101):
        row = (cell - 1) // GRID_COLS
        col = (cell - 1) % GRID_COLS

        # Calculate average error if we have data
        if errors_collection[cell]:
            error_matrix[row][col] = np.mean(errors_collection[cell])

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create heatmap
    ax = sns.heatmap(
        error_matrix,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=.5
    )

    # Add labels and title
    plt.title("Eye Tracking Prediction Errors (Avg of 2 Rounds)")
    plt.xlabel("Column")
    plt.ylabel("Row")

    # Save the visualization
    plt.savefig("eye_tracking_errors2.png")
    plt.show()

    # Also save the raw data
    error_df = pd.DataFrame(errors_collection).T
    error_df.columns = ["Round1", "Round2"]
    error_df["Average"] = error_df.mean(axis=1)
    error_df.to_csv("eye_tracking_errors.csv")
    print("Data saved to eye_tracking_errors.csv")


def main():
    try:
        # Run the error collection
        collect_errors()

        # Close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Visualize the results
        visualize_errors()

        # Print summary
        print("\nError Collection Summary:")
        for cell, errors in errors_collection.items():
            if errors:
                print(f"Cell {cell}: {errors}, Avg: {np.mean(errors):.2f}")

    except KeyboardInterrupt:
        print("Process interrupted by user")
    finally:
        # Ensure resources are released
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
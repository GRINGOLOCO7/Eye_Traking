import cv2
import numpy as np
import pyautogui
import os
import csv
import dlib

# Initialize webcam
cap = cv2.VideoCapture(0)
# Define screen and grid properties
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
GRID_COLS, GRID_ROWS = 10, 10
CORP_EYE_WIDTH, CORP_EYE_HEIGHT = 155, 30
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT // GRID_ROWS
detector = dlib.get_frontal_face_detector() # Load the pre-trained face detector from dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # Load the facial landmarks predictor
# Create a black screen-sized image
grid_img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

# Create output directory
OUTPUT_DIR = "data/saved_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# CSV file setup
CSV_FILE = "data/face.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["face_image_path", "cell"])  # Header row

def highlight_square(x, y):
    """Draw a red rectangle highlighting the given grid square."""
    img = grid_img.copy()
    cv2.rectangle(img, (x, y), (x + GRID_WIDTH, y + GRID_HEIGHT), (0, 0, 255), 2)
    return img

def capture_images(cell_number):
    """Capture 100 images, save them, and record data in CSV."""
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)

        for i in range(100):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            # Save the image
            filename = os.path.join(OUTPUT_DIR, f"cell_{cell_number}_img_{i+100}.png")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                # Predict facial landmarks for each face
                landmarks = predictor(gray, face)
                # Extract the coordinates for the eyes
                left_eye = (landmarks.part(36).x, landmarks.part(36).y)  # Left eye corner
                right_eye = (landmarks.part(45).x, landmarks.part(45).y)  # Right eye corner
                # Calculate the center of the eyes
                eye_center_x = (left_eye[0] + right_eye[0]) // 2
                eye_center_y = (left_eye[1] + right_eye[1]) // 2
                # Crop the region around the eyes, making sure we don't go out of bounds # end up with 30x155
                eyes_region = frame[eye_center_y - 15:eye_center_y + 15, eye_center_x - 77:eye_center_x + 77]  # (30, 154, 3)
                if eyes_region.shape[:2] != (30, 155): # Ensure the correct size
                    eyes_region = cv2.resize(eyes_region, (155, 30))
                # Display the cropped eyes region
                eyes_region = cv2.resize(eyes_region, (CORP_EYE_WIDTH, CORP_EYE_HEIGHT))




                cv2.imwrite(filename, eyes_region)
                # Log in CSV
                writer.writerow([filename, cell_number])

                # Show webcam feed
                cv2.imshow("Webcam", eyes_region)
            if cv2.waitKey(50) & 0xFF == ord('q'):  # Press 'q' to exit early
                return

def main():
    cell_number = 1  # Start cell numbering from 1
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x, y = col * GRID_WIDTH, row * GRID_HEIGHT
            img = highlight_square(x, y)
            cv2.imshow("Grid", img)

            # Wait for 5 sec before capturing images
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                break

            # Capture images for this grid cell
            capture_images(cell_number)
            cell_number += 1  # Move to the next cell

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

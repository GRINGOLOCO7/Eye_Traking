'''
Script to collect images of the eyes region for the CNN model
to thest model on new discrete images
'''


import cv2
import numpy as np
import pyautogui
import os
import dlib

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define screen and grid properties
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
GRID_COLS, GRID_ROWS = 10, 10
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT // GRID_ROWS
CORP_EYE_WIDTH, CORP_EYE_HEIGHT = 155, 30
# Create a black screen-sized image
grid_img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
detector = dlib.get_frontal_face_detector() # Load the pre-trained face detector from dlib
predictor = dlib.shape_predictor('../collect_data/shape_predictor_68_face_landmarks.dat') # Load the facial landmarks predictor
# Create output directory
OUTPUT_DIR = "test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_square(n):
    row = (n - 1) // GRID_COLS
    col = (n - 1) % GRID_COLS
    x, y = col * GRID_WIDTH, row * GRID_HEIGHT
    img = grid_img.copy()
    cv2.rectangle(img, (x, y), (x + GRID_WIDTH, y + GRID_HEIGHT), (0, 0, 255), 2)
    return img

def capture_image(cell_number):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
    # Save the image
    filename = os.path.join(OUTPUT_DIR, f"{cell_number}.png")

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
        cv2.imshow("Eyes Region", eyes_region)#eye_image_resized)


        cv2.imwrite(filename, eyes_region)
        print(f"Image saved to {filename}")
    if cv2.waitKey(50) & 0xFF == ord('q'):  # Press 'q' to exit early
        return


########################################################################
while True:
    # random number btw 1 adn 100
    cell_number = np.random.randint(1, GRID_COLS * GRID_ROWS + 1)
    grid = draw_square(cell_number)
    cv2.imshow("Grid", grid)
    # sleep 3 sec
    cv2.waitKey(3000)

    # Capture image
    capture_image(cell_number)
    cv2.waitKey(1000)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

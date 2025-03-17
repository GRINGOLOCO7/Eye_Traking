import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load model
model = load_model('model.h5')

# Open webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('../data_collection_phase/haarcascade_frontalface_default.xml')

# Screen dimensions and grid
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
GRID_COLS, GRID_ROWS = 2, 2
CELL_WIDTH, CELL_HEIGHT = SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT // GRID_ROWS

def preprocess_face(frame, x, y, w, h):
    """ Extracts and preprocesses the face image to match the model's input format. """
    cx, cy = x + w // 2, y + h // 2
    half_size = 190 // 2
    x1, y1 = max(cx - half_size, 0), max(cy - half_size, 0)
    x2, y2 = x1 + 190, y1 + 190

    cropped = frame[y1:y2, x1:x2]
    if cropped.shape[0] < 190 or cropped.shape[1] < 190:
        return None  # Skip frame if the face region is too small

    # Resize to 190x190
    resized = cv2.resize(cropped, (190, 190))

    # Extract both eyes region ((20, 40, 170, 85))
    eye_region = resized[40:85, 20:170]

    # Normalize and reshape
    img_array = np.array(eye_region, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

def get_gaze_cell(prediction):
    """ Converts model prediction to screen grid coordinates. """
    predicted_class = np.argmax(prediction)  # Get the most likely class (0-3)
    row = predicted_class // GRID_COLS
    col = predicted_class % GRID_COLS
    return col, row

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Use the largest detected face
        img_array = preprocess_face(frame, x, y, w, h)

        if img_array is not None:
            prediction = model.predict(img_array)
            col, row = get_gaze_cell(prediction)

            # Convert grid position to screen coordinates
            gaze_x = int((col + 0.5) * CELL_WIDTH)
            gaze_y = int((row + 0.5) * CELL_HEIGHT)

            # Show gaze point on screen
            screen_display = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            cv2.circle(screen_display, (gaze_x, gaze_y), 40, (0, 0, 255), -1)  # Red dot

            # Show result
            cv2.imshow("Gaze Prediction", screen_display)

    cv2.imshow('Webcam Feed', frame)  # Show webcam feed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

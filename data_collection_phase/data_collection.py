import cv2
import csv
import os
import pyautogui
import numpy as np
import time

# Take a screenshot of the screen
def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

# Draw a point randomly on the screen
def draw_point(image, x, y, color=(0, 0, 255), radius=30):
    cv2.circle(image, (x, y), radius, color, -1)

# Normalize and save the detected region
def normalize_and_save(image, x, y, w, h, target_width, target_height, save_path):
    cx, cy = x + w // 2, y + h // 2
    half_width, half_height = target_width // 2, target_height // 2

    x1, y1 = max(cx - half_width, 0), max(cy - half_height, 0)
    x2, y2 = x1 + target_width, y1 + target_height

    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (target_width, target_height))
    cv2.imwrite(save_path, resized)
    return save_path

def main():
    # Face and eye detection setup
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Directory to save the images
    output_dir = "data/saved_images"
    os.makedirs(output_dir, exist_ok=True)

    # Resume frame count from the last saved image
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("face_image_") and f.endswith(".png")]
    frame_count = max([int(f.split("_")[2].split(".")[0]) for f in existing_files], default=0) + 1

    # CSV file setup
    csv_file_path = 'data/eye_data.csv'
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:  # Append mode
        csv_writer = csv.writer(csvfile)
        if not file_exists:  # Write header if file is new
            csv_writer.writerow(["File Path Face Image", "File Path Left Eye Image", "File Path Right Eye Image", "x", "y"])

        cap = cv2.VideoCapture(0)
        print("Starting in 5 seconds. Please position yourself in front of the camera.")
        time.sleep(5)

        screenshot = take_screenshot()
        base_image = screenshot.copy()
        screen_width, screen_height = screenshot.shape[1], screenshot.shape[0]
        print(f"Screen dimensions: {screen_width}x{screen_height}") # 1920x1080

        while frame_count < 1000:  # Limit to 1000 frames
            x = np.random.randint(0, screen_width)
            y = np.random.randint(0, screen_height)

            screenshot = base_image.copy()
            draw_point(screenshot, x, y)
            cv2.imshow("Interactive Screen", screenshot)

            time.sleep(2.5)  # 2.5 sec to adjust eye to look at the point

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection and saving
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

            face_image_path = None
            for (fx, fy, fw, fh) in faces:
                face_image_path = normalize_and_save(frame, fx, fy, fw, fh, 190, 190,
                                                     os.path.join(output_dir, f"face_image_{frame_count}.png"))
                break

            # Eye detection and saving
            eyes = eye_cascade.detectMultiScale(gray)
            eyes = sorted(eyes, key=lambda e: e[0])  # Sort eyes by x-coordinate for left/right classification

            left_eye_image_path = None
            right_eye_image_path = None
            if len(eyes) >= 2:
                eye1, eye2 = eyes[:2]
                # Sort eyes by x-coordinate to determine left and right
                if eye1[0] < eye2[0]:
                    left_eye, right_eye = eye1, eye2
                else:
                    left_eye, right_eye = eye2, eye1

                lex, ley, lew, leh = left_eye
                rex, rey, rew, reh = right_eye

                left_eye_image_path = normalize_and_save(frame, lex, ley, lew, leh, 45, 45,
                                                         os.path.join(output_dir, f"left_eye_image_{frame_count}.png"))
                right_eye_image_path = normalize_and_save(frame, rex, rey, rew, reh, 45, 45,
                                                          os.path.join(output_dir, f"right_eye_image_{frame_count}.png"))

            # Write to CSV
            if face_image_path and left_eye_image_path and right_eye_image_path:
                csv_writer.writerow([
                    face_image_path.replace('\\', '/'),
                    left_eye_image_path.replace('\\', '/'),
                    right_eye_image_path.replace('\\', '/'),
                    x, y
                ])
                frame_count += 1
            else:
                print("Skipping frame due to missing data.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting on user request.")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

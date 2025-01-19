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
    # transform the image to full white
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    return screenshot

# Draw a point randomly on the screen
def draw_point(image, x, y, color=(0, 0, 255), radius=10):
    cv2.circle(image, (x, y), radius, color, -1)

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
        print("Starting in 10 seconds. Please position yourself in front of the camera.")
        time.sleep(10)

        screenshot = take_screenshot()
        base_image = screenshot.copy()
        screen_width, screen_height = screenshot.shape[1], screenshot.shape[0]
        print(f"Screen dimensions: {screen_width}x{screen_height}")

        while frame_count < 1000:  # Limit to 1000 frames
            x = np.random.randint(0, screen_width)
            y = np.random.randint(0, screen_height)

            screenshot = base_image.copy()
            draw_point(screenshot, x, y)
            cv2.imshow("Interactive Screen", screenshot)

            time.sleep(2)

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Face detection and saving
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

            face_image_path = None
            for (fx, fy, fw, fh) in faces:
                face_image_path = os.path.join(output_dir, f"face_image_{frame_count}.png")
                cv2.imwrite(face_image_path, frame[fy:fy+fh, fx:fx+fw])
                break

            # Eye detection and saving
            eyes = eye_cascade.detectMultiScale(gray)
            eyes = sorted(eyes, key=lambda x: x[2]*x[3], reverse=True)

            left_eye_image_path = None
            right_eye_image_path = None
            count_eye = 0
            for (ex, ey, ew, eh) in eyes:
                if len(eyes) >= 2:
                    if count_eye == 0:
                        left_eye_image_path = os.path.join(output_dir, f"left_eye_image_{frame_count}.png")
                        cv2.imwrite(left_eye_image_path, frame[ey:ey+eh, ex:ex+ew])
                    elif count_eye == 1:
                        right_eye_image_path = os.path.join(output_dir, f"right_eye_image_{frame_count}.png")
                        cv2.imwrite(right_eye_image_path, frame[ey:ey+eh, ex:ex+ew])
                    count_eye += 1
                    if count_eye == 2:
                        break

            # Write to CSV
            csv_writer.writerow([face_image_path, left_eye_image_path, right_eye_image_path, x, y])
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting on user request.")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

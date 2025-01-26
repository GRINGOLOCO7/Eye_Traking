import cv2
import pandas as pd
import os

# Fixed directory path where images are stored
image_dir = "..\\data_collection_phase\\data\\saved_images"

# Load the cleaned CSV containing approved images
csv_file = "cleaned_eye_data.csv"  # Path to your CSV
data = pd.read_csv(csv_file)

# Flatten the data into three lists (faces, left eyes, right eyes)
face_images = data['File Path Face Image'].tolist()
left_eye_images = data['File Path Left Eye Image'].tolist()
right_eye_images = data['File Path Right Eye Image'].tolist()

# Combine all images into one list for sequential display
all_images = [
    ('Face', face_images),
    ('Left Eye', left_eye_images),
    ('Right Eye', right_eye_images)
]

# Start from the first image in the sequence
current_category = 0  # 0 = Faces, 1 = Left Eyes, 2 = Right Eyes
current_index = 0

# Create a single OpenCV window
window_title = "Image Viewer"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

while True:
    # Get the current category and image list
    category_name, image_list = all_images[current_category]

    # Get the current image path
    image_name = image_list[current_index]
    image_path = os.path.join(image_dir, image_name)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        current_index = (current_index + 1) % len(image_list)  # Skip to the next
        continue

    # Resize the window to fit the image size
    cv2.resizeWindow(window_title, image.shape[1], image.shape[0])

    # Display the image with overlayed text for the category and file name
    display_image = image.copy()
    text = f"{category_name} - {image_name} ({current_index + 1}/{len(image_list)})"
    #cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)

    # Update the window with the new image
    cv2.imshow(window_title, display_image)

    # Wait for user input
    key = cv2.waitKey(0) & 0xFF

    if key == ord('l'):  # Move forward
        current_index += 1
        if current_index >= len(image_list):  # Move to the next category
            current_index = 0
            current_category = (current_category + 1) % len(all_images)
    elif key == ord('a'):  # Move backward
        current_index -= 1
        if current_index < 0:  # Move to the previous category
            current_category = (current_category - 1) % len(all_images)
            current_index = len(all_images[current_category][1]) - 1
    elif key == ord('q'):  # Quit
        print("Exiting image viewer.")
        break

# Cleanup
cv2.destroyAllWindows()

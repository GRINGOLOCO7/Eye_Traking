import cv2
import os

# Path to the directory containing images
image_dir = "..\\data_collection_phase\\data\\saved_images"  # Replace with your directory path

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
image_files.sort()  # Sort to maintain order

if not image_files:
    print("No images found in the directory.")
    exit()

current_index = 0  # Start with the first image
approved_face = [] # List to store the approved face images
approved_left_eye = [] # List to store the approved left eye images
approved_right_eye = [] # List to store the approved right eye images
backtrack = 0

while True:
    # Read the current image
    image_path = os.path.join(image_dir, image_files[current_index])
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_files[current_index]}")
        break

    # Display the image
    cv2.imshow("Image Viewer", image)

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        # Quit the program
        break
    elif key == ord('l'):
        # Next image
        current_index = (current_index + 1) % len(image_files)
        approved_face.append(image_files[current_index])
        print(f'approved_face: {approved_face}')
    elif key == ord('a'):
        # Previous image
        current_index = (current_index - 1 + len(image_files)) % len(image_files)
        backtrack += 1
    elif key == ord('k'):
        # delete the image from list
        if image_files[current_index][0] == 'f': # face image
            for _ in range(backtrack):
                print("Deleting: ", approved_face[-1])
                approved_face.remove(approved_face[-1])
            approved_face.remove(approved_face[-1])
            print(f'approved_face: {approved_face}')
            backtrack = 0

# Cleanup
cv2.destroyAllWindows()
print("Approved face images: ", approved_face)

'''
l: Approve the current triple and move forward. (✔️ ➡️)
a: Disapprove the current triple and move forward. (❌ ➡️)
b: Go back to the previous triple. (⬅️)
n: Skip the current triple without approving. (➡️)
s: Disapprove the last approved triple but stay on the current index. (⬅️❌➡️)
q: Quit and save the cleaned CSV.
'''

import cv2
import os
import pandas as pd

# Path to the directory containing images
image_dir = "..\\data_collection_phase\\data\\saved_images"  # Replace with your directory path

# Get a list of image indices based on the face images
image_indices = sorted(
    [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(image_dir) if f.startswith("face_image_") and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
)

if not image_indices:
    print("No images found in the directory.")
    exit()

current_index = 0  # Start with the first set of images
approved = []
undo_stack = []  # Keep track of approvals for undo functionality

while True:
    # Construct image file names
    face_image = f'face_image_{image_indices[current_index]}.png'
    left_eye_image = f'left_eye_image_{image_indices[current_index]}.png'
    right_eye_image = f'right_eye_image_{image_indices[current_index]}.png'

    # Build paths to the images
    face_path = os.path.join(image_dir, face_image)
    left_eye_path = os.path.join(image_dir, left_eye_image)
    right_eye_path = os.path.join(image_dir, right_eye_image)

    # Read images
    face = cv2.imread(face_path)
    left_eye = cv2.imread(left_eye_path)
    right_eye = cv2.imread(right_eye_path)

    # Check if any image failed to load
    if face is None or left_eye is None or right_eye is None:
        print(f"Failed to load images for index: {image_indices[current_index]}")
        current_index = (current_index + 1) % len(image_indices)  # Skip this set
        continue

    # Ensure images have the same height
    target_height = face.shape[0]
    left_eye = cv2.resize(left_eye, (int(left_eye.shape[1] * target_height / left_eye.shape[0]), target_height))
    right_eye = cv2.resize(right_eye, (int(right_eye.shape[1] * target_height / right_eye.shape[0]), target_height))

    # Concatenate images horizontally
    try:
        combined_image = cv2.hconcat([face, left_eye, right_eye])
    except Exception as e:
        print(f"Error concatenating images for index {image_indices[current_index]}: {e}")
        current_index = (current_index + 1) % len(image_indices)
        continue

    # Display the images together
    cv2.imshow("Image Viewer", combined_image)

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # Quit the program
        break
    elif key == ord('l'):  # Approve and move to next
        approved.append([face_image, left_eye_image, right_eye_image])
        undo_stack.append(current_index)
        print(f"Approved: {approved[-1]}")
        current_index = (current_index + 1) % len(image_indices)
    elif key == ord('a'):  # Disapprove and move to next
        print(f"Disapproved: {image_indices[current_index]}")
        current_index = (current_index + 1) % len(image_indices)
    elif key == ord('b'):  # Go back to the previous index
        current_index = (current_index - 1) % len(image_indices)
        print(f"Going back to index {image_indices[current_index]}")
    elif key == ord('n'):  # Skip without approving
        print(f"Skipped index {image_indices[current_index]}")
        current_index = (current_index + 1) % len(image_indices)
    elif key == ord('s'):  # Disapprove the last approved and stay on current
        if undo_stack:
            last_index = undo_stack.pop()
            if approved and [f'face_image_{image_indices[last_index]}.png',
                             f'left_eye_image_{image_indices[last_index]}.png',
                             f'right_eye_image_{image_indices[last_index]}.png'] in approved:
                approved.remove([f'face_image_{image_indices[last_index]}.png',
                                 f'left_eye_image_{image_indices[last_index]}.png',
                                 f'right_eye_image_{image_indices[last_index]}.png'])
            print(f"Disapproved last approved index {image_indices[last_index]}. Staying on current index {image_indices[current_index]}")
        else:
            print("Nothing to disapprove.")

# Cleanup
cv2.destroyAllWindows()
print("Final approved images:", approved)

# Load the CSV data
csv_file = "..\\data_collection_phase\\data\\eye_data.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Convert approved list to a set for faster lookups
approved_set = set(tuple(item) for item in approved)

# Filter the DataFrame to keep only rows that match the approved triples
cleaned_data = data[data.apply(lambda row: (
    row['File Path Face Image'],
    row['File Path Left Eye Image'],
    row['File Path Right Eye Image']) in approved_set, axis=1)]

# Save the cleaned dataset
output_file = "cleaned_eye_data.csv"
cleaned_data.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
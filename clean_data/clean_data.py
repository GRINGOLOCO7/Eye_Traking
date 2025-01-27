import cv2
import os
import pandas as pd

# Path to the directory containing images
image_dir = "..\\data_collection_phase\\data\\saved_images"  # Replace with your directory path

# Define the fixed directory for saving images
fixed_directory = "data/saved_images/"

# Get a list of image indices based on the face images
image_indices = sorted(
    [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(image_dir) if f.startswith("face_image_") and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
)

# Ensure we are starting from the right index (1010 or the desired starting index)
start_index = 1010
if start_index >= len(image_indices):
    print(f"Starting index {start_index} is out of range.")
    exit()

# Set the current index to start from 1010 (or other value)
current_index = start_index
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
        # Add the fixed directory to the image paths
        approved.append([os.path.join(fixed_directory, face_image),
                         os.path.join(fixed_directory, left_eye_image),
                         os.path.join(fixed_directory, right_eye_image)])
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
            if approved and [os.path.join(fixed_directory, f'face_image_{image_indices[last_index]}.png'),
                             os.path.join(fixed_directory, f'left_eye_image_{image_indices[last_index]}.png'),
                             os.path.join(fixed_directory, f'right_eye_image_{image_indices[last_index]}.png')] in approved:
                approved.remove([os.path.join(fixed_directory, f'face_image_{image_indices[last_index]}.png'),
                                 os.path.join(fixed_directory, f'left_eye_image_{image_indices[last_index]}.png'),
                                 os.path.join(fixed_directory, f'right_eye_image_{image_indices[last_index]}.png')])
            print(f"Disapproved last approved index {image_indices[last_index]}. Staying on current index {image_indices[current_index]}")
        else:
            print("Nothing to disapprove.")

# Cleanup
cv2.destroyAllWindows()
print("Final approved images:", approved)

# Load the existing cleaned CSV
cleaned_csv_file = "cleaned_eye_data.csv"  # Replace with your cleaned CSV file path

# Check if the file exists and load it, otherwise create an empty DataFrame
if os.path.exists(cleaned_csv_file):
    cleaned_data = pd.read_csv(cleaned_csv_file)
    print(f"Loaded existing cleaned data from {cleaned_csv_file}")
else:
    cleaned_data = pd.DataFrame(columns=["File Path Face Image", "File Path Left Eye Image", "File Path Right Eye Image", "x", "y"])
    print(f"No existing cleaned data found. Creating new cleaned data.")

# Load the original data
csv_file = "..\\data_collection_phase\\data\\eye_data.csv"  # Replace with your original CSV file path
data = pd.read_csv(csv_file)

# Convert approved list to a set for faster lookups
approved_set = set(tuple(item) for item in approved)

# Filter the DataFrame to keep only rows that match the approved triples
new_data = data[data.apply(lambda row: (
    row['File Path Face Image'],
    row['File Path Left Eye Image'],
    row['File Path Right Eye Image']) in approved_set, axis=1)]

# Check if new_data has any valid records
if not new_data.empty:
    print(f"Found {len(new_data)} new records to add.")
else:
    print("No new records found to add.")

# Show the new data for debugging (make sure they have x and y)
print(f"New records data:\n{new_data}")

# Add the new approved records to the cleaned data
updated_cleaned_data = pd.concat([cleaned_data, new_data], ignore_index=True)

# Check the concatenated result
print(f"Updated cleaned data:\n{updated_cleaned_data}")

# Save the updated dataset
output_file = "cleaned_eye_data.csv"
updated_cleaned_data.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")

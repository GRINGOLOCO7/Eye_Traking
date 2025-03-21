import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import random

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
GRID_COLS = 10
GRID_ROWS = 10
print(f"cell size = {SCREEN_WIDTH/GRID_COLS}x{SCREEN_HEIGHT/GRID_ROWS}")

# load database
df = pd.read_csv('../collect_data/data/face.csv')
print(len(df))
'''
face_image_path	cell
0	cell_1_img_0.png	1
1	cell_1_img_1.png	1
...
'''

# Define paths
data_dir = "../collect_data/data/saved_images"  # Folder with original images
output_dir = "dataset"  # Root folder for train/test dataset
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Create train and test directories with subfolders for each class (1-100)
for split in ["train", "test"]:
    for label in range(1, 101):  # Cells are numbered from 1 to 100
        os.makedirs(os.path.join(output_dir, split, str(label)), exist_ok=True)

# Split data (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["cell"], random_state=42)


'''
✅ Rotated by ±15°

✅ Adjusted for brightness and contrast

✅ Saved in both original and augmented
'''
# Function to apply data augmentation
def augment_image(image):
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    image = image.rotate(angle)

    # Random brightness & contrast adjustment
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.5, 2))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.5, 2))

    return image

# Function to process and save images
def process_and_save_images(df_subset, split):
    for i in range(len(df_subset)):
        img_path = df_subset.iloc[i]['face_image_path']  # Filename
        full_path = os.path.join(data_dir, img_path)  # Full path
        label = str(df_subset.iloc[i]['cell'])  # Keep 1-100 as labels

        if os.path.exists(full_path):  # Ensure file exists
            img = Image.open(full_path)

            # Apply augmentation
            face_augmented = augment_image(img)

            # Save original and augmented versions
            save_path = os.path.join(output_dir, split, label, f"{os.path.basename(img_path)}")
            img.save(save_path)

            augmented_save_path = os.path.join(output_dir, split, label, f"aug_{os.path.basename(img_path)}")
            face_augmented.save(augmented_save_path)

# Process training and testing images
process_and_save_images(train_df, "train")
process_and_save_images(test_df, "test")

print("✅ Dataset successfully created with data augmentation!")
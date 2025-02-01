# Eye Traking System

In this repo we will create a CNN to detect where the user eye are looking in the screen using the computer webcam

**Computer Desktop dimention 1920x1080**

## Repo Structure

- [data_collection_phase](https://github.com/GRINGOLOCO7/Eye_Traking/tree/main/data_collection_phase): Script to collect data
- [clean_data](https://github.com/GRINGOLOCO7/Eye_Traking/tree/main/clean_data): Procedure to extract only good tripes of face, left eye and right eye
- [only_face_model](https://github.com/GRINGOLOCO7/Eye_Traking/tree/main/only_face_model): CNN trained to detect where you are looking on the screen based on face corped image
- [triple_eyeface_model](https://github.com/GRINGOLOCO7/Eye_Traking/tree/main/triple_eyeface_model): CNN trained to detect where you are looking on the screen based on triple (face, right eye and left eye corped images)

## Collect Data (data_collection_phase)

To collect data we prepared a code (`data_collection.py`).

1. The code open a window on the screen as big as Desktop screen. Then gives you 5 seconds to prepare.

2. Random dots are spawned on the screen

3. The user have cupple of second to look at the dot

4. The code take a screenshot of the face, right eye and left eye and saves them in the `data/saved_images` folder. The face and eyes are detected thanks to pretrained models I downloaded (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`). (find more infos [here](https://www.geeksforgeeks.org/opencv-python-program-face-detection/))

5. **Face, Left Eye, Right Eye, X coordinate of the dot and Y coordinate of the dot are saved in a CSV file (`data/eye_data.csv`)**


## Data Processing (clean_data)
1. Normalize image sizes: All images have pixel dimention difference. We need to cut them to be the all the same pixels. (_fixed during data collection_)
1. clean data: sometimes the eyes are missjuged by air, nose or something in the background. **Delete those records**. In `clean_data/clean_data.py` we show each triple (face right and lef eyes) and with keyboards keys we can approve or disapprove the triple. Then, all the triple with good and clean data are saved in `clean_data/cleaned_eye_data.csv` with of course the respective x and y coordinate we where looking at. For 'debuggin' there is another python script to scroll trough the approved images (`scroll_approved_images.py`).
1. data_exploration: check perentage of the screen that was coverd by the _x_ and _y_ poits we looked at.
2. data_exploration: Normalize Coordinates: Scale x and y to a range between 0 and 1 by dividing by the screen width and height (_in my lenovo: 1920x1080_), respectively.

## CNN Structure and Study

#### Input-Output Relationship

**Input:** A combination of three images: the face image, the left eye image, and the right eye image.

**Output:** A pair of continuous values (x, y) representing the gaze coordinates on the screen.

#### Input Representation
Concatenate Images: Process each image type (face, left eye, right eye) separately through different CNN branches and then concatenate their feature embeddings.

#### Model Architecture:
1. Separate CNN Branches: One CNN for the face image + Two separate CNNs for the left eye and right eye images.
2. Feature Fusion: Concatenate the embeddings from all CNN branches.
3. Fully Connected Layers: Pass the concatenated embeddings through fully connected layers to predict the normalized (x, y) coordinates.
<details>
  <summary>Sample code</summary>

    ```python
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Define CNN for processing images
    def create_cnn(input_shape):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
        ])
        return model

    # Input shapes
    face_input = layers.Input(shape=(64, 64, 3))  # Example shape for face
    left_eye_input = layers.Input(shape=(32, 32, 3))  # Example shape for left eye
    right_eye_input = layers.Input(shape=(32, 32, 3))  # Example shape for right eye

    # Create CNN branches
    face_branch = create_cnn((64, 64, 3))(face_input)
    left_eye_branch = create_cnn((32, 32, 3))(left_eye_input)
    right_eye_branch = create_cnn((32, 32, 3))(right_eye_input)

    # Concatenate features
    concatenated = layers.Concatenate()([face_branch, left_eye_branch, right_eye_branch])

    # Fully connected layers
    fc = layers.Dense(256, activation='relu')(concatenated)
    fc = layers.Dense(128, activation='relu')(fc)
    output = layers.Dense(2, activation='linear')(fc)  # Predict (x, y)

    # Build model
    model = models.Model(inputs=[face_input, left_eye_input, right_eye_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model.summary()
    ```

</details>

## Resources

Research Papers:

- "Gaze Estimation via Deep Learning" (e.g., papers from ECCV or CVPR).
- Papers from datasets like MPIIGaze or GazeCapture.

Datasets:

- MPIIGaze, GazeCapture for transfer learning or pretraining.

Tutorials:

- TensorFlow/Keras or PyTorch tutorials on multimodal inputs.
- [Youtube Eye-tracking Mouse Using Convolutional Neural Networks and Webcam](https://www.youtube.com/watch?v=iV9ZkvdsL7I)

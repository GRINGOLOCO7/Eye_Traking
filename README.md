# Eye Traking System

In this repo we will create a CNN to detect where the user eyes are looking in the screen using only computer webcam.

The objective is to create the most:
- simple
- effective
- ready to use
- for everyone to explore
- NOT computational expensive
- ONLY NEED BUILD IN WEBCAM

CNN model

We discretized the problem form a regression model that predict (x, y) coordinate of eye gaze on the screen, to a CNN classifier that predicts eye gaze on a 10x10 grid of the coputer screen.

**Computer Desktop dimention 1920x1080**

This way the model learns faster and we where able to collect much more (quality) data.

<br>

---

## Results

Training and testing results are amzing!! Above 90% accuracy in both!!

While [training](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/train/train_model.ipynb) we can see that loss functions of both train and validation decrease, advising that the model is learning well without overfitting, plus high accuracy of validation (above 90%)

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_training_model.jpg" alt="drawing" width="300"/>

Checking at test set, we also conclude high accuracy and good results! Overall accuracy above 90% on these unseen immages. TEST RESULTS:

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_testing_model.jpg" alt="drawing" width="500"/>

When going to live testing, we use the same technique used when collecting data, and frame by frame feed the image of the 2 eyes to the model. As we can notice in [test_offline.ipynb](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/test_offline.ipynb) or by running [live_testing.py](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/live_testing.py), here the predictions are not even close to the amazing results we had in traain and test. The problem can be caused by different lightning (even if unporbable, given the fact that before training we augmented the immages, adding random brighness, contract adn rotations), OR due to different head positions. We can conclude that the mdoel perform poorly on live data, due to small head moovment never seen by the model.

As I said, same we belive that same eyes pictures, can have different eye focus, depending on hwere the face is... Every time we tryied to replicate and make the image aquisition standard, but even with this attention, the results are not accurate as in testing
I believe the problem is that it is very sensitive to changes in face position and light conditions (I tried to tape the computer on the table and fix my head and chair position for data collecting and testing. Plus I tried to do everything around the same time of the day, but still nothing to do…).


<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/same_eyes_config_different_gaze.jpg" alt="drawing" width="300"/>

Also time of the day differences make the predictions goes grazy (that is wy I ran the data collection phase twice in diffrent ligthing rooms... to make model strongger, but still, the live testing is much MUCH less accurate than test set)

Looking at each point in the 10x10 grid twice, with random order, we get this error pattern:

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/errors_calculation_patterns/eye_tracking_errors.jpg" alt="drawing" width="300"/>

Where each cell tels me how much, in live testing, the `predicted value by the model - real value` is. We can try to find patterns and smooth the eye traking in real time.




<br>

---

## Repo Structure

1. `collect_data/`
    - The script will take about 10 minutes and will record a total of 10k images in one run.
    - Run the script in the same folder
    - The script will pop up a balck screen as big as the computer screen
    - Will divide the screen in a 10x10 grid
    - One by one will show on the screen one square to look
    - Give 3 seconds to adjust eye gaze
    - Start recording and storing each frame
    - In the end we end up with a new folder called `data/` that cotains a `face.csv` file and a `saved_images` folder. The csv file maps the images to the cell we where looking at
2. `dataset/`
    - Torch needs a specific structure for the dataset. It is required a folder for testing and a forlder for training. In each subfolders, as much as the classes we have to predict (in our case 10x10 grid = 100 classes = 100 folders)
    - The script reorganize automticaly the images in to this structure.
    - Plus apply randomly augmentation od the images. Increaseing the dataset dimention and the model strenght, plus avoiding overfitting.
3. `model/`
    - In the python file we difine the CNN structure (as we can notice we use a resNet as base and start trining from there). We unfreze the last 2 layers for better learning.
    - We add dropout of 0.5 to avoid overfitting and last layer of 100 to predict our 100 classes
4. `train/`
    - The train script loads the dataset and the model and start training
    <img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/infinite_trainings.jpg" alt="drawing" width="300"/>
5. `test/`
    - the test scripts are there to play with new, and diverse data inputs
    - These never seen data, images of eyes collected with smae technique of the data collection phase but in different time of day and locations, makes the model brake easily.
    <img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/collection_data_phase.jpg" alt="drawing" width="300"/>

<br>





## Future Improovments
1. Add pitch roll and yaw to triangulate face position and eye gaze precision
2. Combine and concatenate more images for bigger overall model
- Separate CNN Branches: One CNN for the face image + Two separate CNNs for the left eye and right eye images.
- Feature Fusion: Concatenate the embeddings from all CNN branches.
- Fully Connected Layers: Pass the concatenated embeddings through fully connected layers to predict the normalized (x, y) coordinates.
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

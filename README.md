# Eye Traking System

In this repo, we will create a CNN that detects where the user's eyes are looking at in the screen using only a computer webcam.

The objective is to create the most:
- simple
- effective
- ready to use
- for everyone to explore
- NOT computationally expensive
- ONLY NEED BUILD IN WEBCAM

CNN model

We discretized the problem form a regression model that predicts (x, y) coordinate of eye gaze on the screen to a CNN classifier that predicts eye gaze on a 10x10 grid of the computer screen.

Objective:

Use eyes images (taken from webcam) to predict where on the screen the user is looking at.

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/collect_data/data/saved_images/cell_100_img_0.png" alt="drawing" width="150"/>

The screen is discretized into a 10x10 grid, making the problem a classification (and not a regression)

**Computer Desktop dimention 1920x1080**

This way, the model learns faster and we were able to collect much more (quality) data.

<br>

---

## Execution

To execute the Eye Tracking System project, please follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GRINGOLOCO7/Eye_Traking.git
   cd Eye_Traking
   ```

2. **Install Dependencies:**
   Ensure that you have Python and the required libraries installed. You can install the necessary dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Collect Data:**
   Before training the model, you need to collect data using the `collect_data` script. This script will record images of your eyes looking at different parts of the screen.
   ```bash
   python collect_data/data_collection.py
   ```

4. **Prepare Dataset:**
   The collected data needs to be organized into a structure suitable for training. Run the dataset preparation script:
   ```bash
   python dataset/organize_dataset.py
   ```

5. **Train the Model:**
   Train the Convolutional Neural Network (CNN) using the prepared dataset. Execute `   jupyter train/train_model.ipynb`

6. **Live Testing:**
   For live testing, use the same technique used during data collection. The script will capture frames from the webcam and predict the eye gaze on the screen in real-time.
   ```bash
   python test/test_live.py
   ```


## Results

Training and testing results are amazing!! Above 90% accuracy in both!!

While [training](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/train/train_model.ipynb), we can see that loss functions of both train and validation decrease, advising that the model is learning well without overfitting, plus high accuracy of validation (above 90%)

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_training_model.jpg" alt="drawing" width="300"/>

Checking at test set, we also conclude high accuracy and good results! The overall accuracy was above 90% on these unseen images. TEST RESULTS:

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_testing_model.jpg" alt="drawing" width="500"/>

When going to live testing, we use the same technique used when collecting data, and frame by frame feed the image of the 2 eyes to the model. As we can notice in [test_offline.ipynb](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/test_offline.ipynb) or by running [live_testing.py](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/live_testing.py), the predictions are not even close to the amazing results we had in train and test. The problem can be caused by different lightning (even if unporbable, given the fact that before training we augmented the images, adding random brightness, contrast, and rotations) OR due to different head positions. We can conclude that the model performs poorly on live data due to small head movements never seen by the model.

As I said, we believe that the same eyes pictures can have different eye focus, depending on where the face is... Every time we tried to replicate and make the image acquisition standard, but even with this attention, the results are not accurate as in testing
I believe the problem is that it is very sensitive to changes in face position and light conditions (I tried to tape the computer on the table and fix my head and chair position for data collecting and testing. Plus I tried to do everything around the same time of the day, but still nothing to do…).


<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/same_eyes_config_different_gaze.jpg" alt="drawing" width="300"/>

Also, time-of-day differences make the predictions go crazy (that is why we ran the data collection phase twice in different lighting rooms... to make the model stronger, but still, the live testing is much MUCH less accurate than the test set)

Looking at each point in the 10x10 grid twice, with random order, we get this error pattern:

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/errors_calculation_patterns/eye_tracking_errors_model1.jpg" alt="drawing" width="300"/>

Where each cell tells me how much, in live testing, the `predicted value by the model - real value` is. We can try to find patterns and smooth the eye tracking in real time.




<br>

---

## Repo Structure

1. `collect_data/`
    - The script will take about 10 minutes and will record a total of 10k images in one run.
    - Run the script in the same folder
    - The script will pop up a black screen as big as the computer screen
    - Will divide the screen in a 10x10 grid
    - One by one will show on the screen one square to look
    - Give 3 seconds to adjust eye gaze
    - Start recording and storing each frame
    - In the end, we end up with a new folder called `data/` that contains a `face.csv` file and a `saved_images` folder. The csv file maps the images to the cell we were looking at
2. `dataset/`
    - Torch needs a specific structure for the dataset. It requires a folder for testing and a folder for training. In each subfolder, as much as the classes we have to predict (in our case, 10x10 grid = 100 classes = 100 folders)
    - The script reorganizes the images automatically into this structure.
    - Plus aptly random augmentation of the images, increasing the dataset dimensions, model strength, and avoiding overfitting.
3. `model/`
    - In the Python file, we define the CNN structure (as we can notice, we use a resNet as base and start training from there). We unfreeze the last 2 layers for better learning.
    - We add a dropout of 0.5 to avoid overfitting and a last layer of 100 to predict our 100 classes
4. `train/`
    - The train script loads the dataset and the model and starts training
    <img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/infinite_trainings.jpg" alt="drawing" width="300"/>
5. `test/`
    - the test scripts are there to play with new and diverse data inputs
    - These never-seen data images of eyes collected with the same technique of the data collection phase but at different times of day and locations, makes the model break easily.
    <img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/collection_data_phase.jpg" alt="drawing" width="300"/>

#### collect_data

The Python file loads the shape_detector_68 model to detect the user's face and eyes. We are able to crop out a 30x155 image of the 2 eyes of the user with this model.

We do this in parallel with popping a window as big as the screen.

The data collection phase has started... It takes around 10 minutes to collect 10k cleaned images of the eyes of the user.

On the window that pops up, we sequentially show the user a square of the 10x10 grid. The user has 3 seconds to concentrate their gaze, and then we start recording 100 frames/images of the user looking at cell 1. We do this for all the 100 cells.

10x10x100 = 10000 images of the user looking at each cell. Along with collecting images, we prepare a dataset containing the path of the image and the cell we are looking at (this will be helpful in the next step).

=> output: /data -> folder containg 10k images labled and a structured database

#### dataset

Torch requests a specific data structure for the images. A folder for testing and training is required. Plus, in each, we need to have one folder for each class containing the images belonging to that class.

In the python file, we automatically split the data in train and test sets and locate them in the correct folders (eg. /dataset/test/10/ -> an image of my eyes where I was looking at cell 10).

Plus, we apply augmentations to some images. This makes the dataset larger and allows the model to be stronger and generalize better:
    - Rotated by ±15°

    - Adjusted for brightness and contrast

    - Saved in both original and augmented

=> output /dataset -> folder containing a structured dataset coherent with torch requirements (test and train and subfolders for each class)

#### model

Here, we define once the model structure and required functions.

Note that we are using resNet as base, where the last layers are free to be finetuned, plus we define last dense layer (with dropout to avoid overfitting) and ending into a layer fo 100 (number of classes we are classifing)

#### train

In the train folder, we have a notebook that trains the CNN to receive a 30x155 image of the eyes and output the cell from 1 to 100 that we are looking at. (_more info in the notebook itself_)

=> output: it outputs and saves the model (.pth) that we can later use for our tasks

#### test

Contains many scripts for live testing, off-line testing, and error visualizations
    - Discrete testing: Script that collects faces looking at cell on the screen. Then, we can predict one by one the eye gaze that the model output and compare it with the real cell we were looking at

    - Error calculation patterns: Will make user complete a round of data collection, where user looks at each cell twice in total and outputs the error pattern for future adjustments in live testing

    - Then we have python files for live testing


<br>





## Future Improvements
THE MODEL IS PROMISING.

Training and testing are very high. This showcases how the challenging problem of eye gaze detection was successfully tackled by discretizing such problem form regression to a classification and simplifying the model to just take as input the image of the eyes and nothing else.

BUT, due to this simplicity, the model lacks in generalization and can't handle small variation form samples form train/test AND live testing.

1. Collect data, train, and test at the same time of the day, with the same illumination to check accuracy
1. Make the data acquisition setup as a standard. This means that data acquisition pose and live test pose should be EXACTLY the same in terms of light condition but also computer position, head high, chair position, fix position fo computer on the table, etc...
1. Add pitch roll and yaw to triangulate face position and eye gaze precision
2. Combine and concatenate more images for the bigger overall model
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

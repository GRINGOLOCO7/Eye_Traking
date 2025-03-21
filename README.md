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

Objective:

Use eyes images (taken from webcam) to predict where on the screen the user is looking at
<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/collect_data/data/saved_images/cell_100_img_0.png" alt="drawing" width="150"/>
The screen is discretized into a 10x10 grid makeing the problem a classification (an not a regression)

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

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/errors_calculation_patterns/eye_tracking_errors_model1.jpg" alt="drawing" width="300"/>

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

#### collect_data

The pytthin file, load the shape_detector_68 model to detect user face and eyes. We are able, using this model, to corp out a 30x155 image of the 2 eyes of the user.

We do this in parallel with popping a window, as big as the screen.

The data collection phase has started... arround 10 minutes to collect 10k cleaned images of the eyes of the user.

On the window that pop up, we sequentialy show to the user a square of the 10x10 grid, The user have 3 seconds to concentrate the gaze and then we start recording 100 frames/images of the user looking at cell 1. We do this for all the 100 cells.

10x10x100 = 10000 images of the user lookig at each cell. Along with collecting images, we prepare a dataset containg the path of the image and the cell we where looking at (this will be helpfull in the next step).

=> output: /data -> folder containg 10k images labled and a structured database

#### dataset

Torch, request a specific data structure for the images. A folder for testing and training is required. Plus, in each, we need to have one folder for each class containg the images belonging to that class.

In the python file, we automaticaly split in train and test the images and locate them in the correct folder (eg. /dataset/test/10/ -> an image of my eyes where I was looking at cell 10).

Plus we apply Augmentations to soem images. This make the dataset larger and allow the model to be stronger adn generalize better:
    - Rotated by ±15°

    - Adjusted for brightness and contrast

    - Saved in both original and augmented

=> output /dataset -> folder containgg structured dataset coherent with torch requironments (test and train and subfolders for each class)

#### model

Here we define once the model structure and required functions.

Note that we are using resNet as base, where the last layers are free to be finetuned, plus we define last dense layer (with dropout to avoid overfitting) and ending into a layer fo 100 (number of classes we are classifing)

#### train

In train folder we have a notebook that train teh CNN to recive a 30x155 image of the eyes, and output the cell form 1 to 100 that we where looking at. (_more infos in the noteboo itself_)

=> output: it output and save the model (.pth) that we can later use for our tasks

#### test

Contains many scripts for live testing and off-line testing and error visualizations
    - Discrete testing: Script tht collect faces looking at cell on the screen. And then we can predict one by one the eye gaze that the model output and compare with the real cell we where looking at

    - Error calculation patterns: Will make user complete a round of data collection, where user look each cell twice in total and output the error pattern for future adjustments in live testing

    - Then we have python files for live testing


<br>





## Future Improovments
THE MODEL IS PROMESSING.

Trainig and testing are very high. This showcase how the challenging problem of eye gaze detection was succesfuly takled, by discretizing such problem form regression to a classification and simplify the model to just take as input the image of the eyes and nothing else.

BUT, due to this semplicity the mdoel lack in generalization and can't handle small variation form samples form traing/test AND live testing.

1. Collect data, train and test at the same time of the day, with same illumination to check accuracy then
1. Make the data acquisition setup as a standard. Meaning that data aquisition pose and live test pose, should be EXACTLY the same, in therm of light condition but also coputer position, head high, chair position, fix position fo computer on the table, etc...
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

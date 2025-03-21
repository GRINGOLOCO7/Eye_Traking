# Eye Traking System

In this repo we will create a CNN to detect where the user eyes are looking in the screen using only computer webcam.

The objective is to create the most:
- simple
- effective
- ready to use
- for everyone to explore
- NOT computational expensive

CNN model

We discretized the problem form a regression model that predict (x, y) coordinate of eye gaze on the screen, to a CNN classifier that predicts eye gaze on a 10x10 grid of the coputer screen.

**Computer Desktop dimention 1920x1080**

This way the model learns faster and we where able to collect much more (quality) data.

<br>

---

## Results

Training and testing results are amzing!! Above 90% accuracy in both!!

While (training)[https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/train/train_model.ipynb] we can see that loss functions of both train and validation decrease, advising that the model is learning well without overfitting, plus high accuracy of validation (above 90%)

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_training_model.jpg" alt="drawing" width="150"/>

Checking at test set, we also conclude high accuracy and good results! Overall accuracy above 90% on these unseen immages

<img src="https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_testing_model.jpg" alt="drawing" width="150"/>

When going to live testing, we use the same technique used when collecting data, and frame by frame feed the image of the 2 eyes to the model. As we can notice in (test_offline.ipynb)[https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/test_offline.ipynb] or by running (live_testing.py)[https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/test/live_testing.py], here the predictions are not even close to the amazing results we had in traain and test. The problem can be caused by different lightning (even if unporbable, given the fact that before training we augmented the immages, adding random brighness, contract adn rotations), OR due to different head positions. We can conclude that the mdoel perform poorly on live data, due to small head moovment never seen by the model.

[![Watch the video first live testing](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/results_testing_model.jpg)](https://github.com/GRINGOLOCO7/Eye_Traking/blob/master/assets/live_test1.mp4)


<br>

---

## Repo Structure

llllllllllllll

<br>





## Future Improovments
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

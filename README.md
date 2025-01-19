# Eye Traking System

In this repo we will create a CNN to detect where the user eye are looking in the screen using the computer webcam

## Collect Data

To collect data I prepared a code (`collect_data.py`).

1. The code open a window on the screen as big as Desktop screen. Then gives you 10 seconds to prepare.

2. Random dots are spawned on the screen

3. The user have cupple of second to look at the dot

4. The code take a screenshot of the face, right eye and left eye and saves them in the `data/saved_images` folder. The face and eyes are detected thanks to pretrained models I downloaded (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`). (find more infos [here](https://www.geeksforgeeks.org/opencv-python-program-face-detection/))

5. Face, Left Eye, Right Eye, X coordinate of the dot and Y coordinate of the dot are saved in a CSV file (`data/eye_data.csv`)

## Train NN


## Test
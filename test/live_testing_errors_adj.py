'''
Script to test live the eye traker with the CNN model
'''

########################################################################IMPORTS:
import cv2
import dlib
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import pyautogui
from retrive_smooth_direction import *
import random

import sys
import os
script_dir = os.path.abspath("..")  # Go up one level from `train/`
sys.path.append(script_dir)
from model.model import *
########################################################################



########################################################################MODEL:
model = ResNetModel(num_classes=100)
model.load_state_dict(torch.load('../train/resnet_model.pth')) # C:\Users\orlan\OneDrive\Desktop\2semestre\AI COMPUTER VISION\code\project\my_offline\model3_eyes\resnet_model.pth
model.eval()
transform = transforms.Compose([
    transforms.Resize((30, 155)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
########################################################################



########################################################################DEFINE:
cap = cv2.VideoCapture(0) # Initialize the webcam
detector = dlib.get_frontal_face_detector() # Load the pre-trained face detector from dlib
predictor = dlib.shape_predictor('../collect_data/shape_predictor_68_face_landmarks.dat') # Load the facial landmarks predictor
# Get screen width and height
CORP_EYE_WIDTH, CORP_EYE_HEIGHT = 155, 30
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()#1920,1080#pyautogui.size()
GRID_COLS, GRID_ROWS = 10, 10
current_cell_base = 55
current_cell_avg = 55
current_cell_max = 55
current_cell_min = 55
current_cell_weigh = 55
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT // GRID_ROWS
grid_img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Create a black canvas to overlay the grid
########################################################################



########################################################################GRID FUNCTIONS:
def draw_square(n, grid_img=grid_img,color=(0, 0, 255)):
    row = (n - 1) // GRID_COLS
    col = (n - 1) % GRID_COLS
    x, y = col * GRID_WIDTH, row * GRID_HEIGHT
    img = grid_img.copy()
    cv2.rectangle(img, (x, y), (x + GRID_WIDTH, y + GRID_HEIGHT), color, 2)
    return img
error_map = {
1: [11, 32],
2: [73, 51],
3: [69, 60],
4: [70, 71],
5: [66, 9],
6: [54, 63],
7: [16, 25],
8: [4, 63],
9: [52, -5],
10: [67, 19],
11: [50, 58],
12: [58, -9],
13: [1, 19],
14: [46, 63],
15: [-3, 46],
16: [66, 57],
17: [60, 59],
18: [60, 43],
19: [53, -19],
20: [56, 56],
21: [50, 64],
22: [11, 39],
23: [9, 40],
24: [51, 37],
25: [52, 7],
26: [49, 52],
27: [46, 49],
28: [64, -13],
29: [48, 32],
30: [47, 19],
31: [2, 48],
32: [51, -32],
33: [41, 0],
34: [14, 7],
35: [10, 35],
36: [25, -1],
37: [39, -5],
38: [-38, 32],
39: [42, 9],
40: [35, 21],
41: [23, -29],
42: [26, 35],
43: [46, -20],
44: [29, -30],
45: [37, -44],
46: [-10, 33],
47: [31, 30],
48: [26, 13],
49: [14, 1],
50: [14, 21],
51: [26, -51],
52: [-49, 10],
53: [10, -8],
54: [-21, 23],
55: [9, -8],
56: [-3, -22],
57: [20, 32],
58: [15, 3],
59: [5, 1],
60: [12, 12],
61: [16, -15],
62: [-28, -62],
63: [-31, 8],
64: [13, -31],
65: [11, -31],
66: [10, 3],
67: [-16, 10],
68: [-4, -4],
69: [-69, -6],
70: [-8, -10],
71: [1, -18],
72: [-24, 5],
73: [-25, -10],
74: [-18, -3],
75: [-6, -42],
76: [-13, 0],
77: [-65, 8],
78: [-30, -1],
79: [2, -18],
80: [-19, 17],
81: [-70, -49],
82: [-70, -7],
83: [-36, -13],
84: [-10, -43],
85: [-14, -38],
86: [-10, -37],
87: [-86, -54],
88: [-17, -40],
89: [-12, -74],
90: [-29, -27],
91: [-14, -62],
92: [-79, -91],
93: [-61, -38],
94: [-15, -34],
95: [-62, -18],
96: [-26, -40],
96: [-26, -40],
96: [-26, -40],
97: [-65, -20],
98: [-38, -21],
99: [-23, -43],
100:[-34, -52]
}
def adjust_error_max(predicted_class, errors):
    new_prediction = predicted_class
    # Make sure the smaller value comes first in random.randint
    max_error = max(errors[0], errors[1])
    new_prediction -= max_error
    if new_prediction < 1:
        new_prediction = predicted_class
    elif new_prediction > 100:
        new_prediction = predicted_class
    return new_prediction
def adjust_error_avg(predicted_class, errors):
    new_prediction = predicted_class
    new_prediction -= (errors[0] + errors[1]) // 2
    if new_prediction < 1:
        new_prediction = predicted_class
    elif new_prediction > 100:
        new_prediction = predicted_class
    return new_prediction
def adjust_error_min(predicted_class, errors):
    new_prediction = predicted_class
    # Make sure the smaller value comes first in random.randint
    max_error = min(errors[0], errors[1])
    new_prediction -= max_error
    if new_prediction < 1:
        new_prediction = predicted_class
    elif new_prediction > 100:
        new_prediction = predicted_class
    return new_prediction
########################################################################



########################################################################
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # Predict facial landmarks for each face
        landmarks = predictor(gray, face)
        # Extract the coordinates for the eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)  # Left eye corner
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)  # Right eye corner
        # Calculate the center of the eyes
        eye_center_x = (left_eye[0] + right_eye[0]) // 2
        eye_center_y = (left_eye[1] + right_eye[1]) // 2
        # Crop the region around the eyes, making sure we don't go out of bounds # end up with 30x155
        eyes_region = frame[eye_center_y - 15:eye_center_y + 15, eye_center_x - 77:eye_center_x + 77]  # (30, 154, 3)
        # Display the cropped eyes region
        eyes_region = cv2.resize(eyes_region, (CORP_EYE_WIDTH, CORP_EYE_HEIGHT))
        print(eyes_region.shape)
        cv2.imshow("Eyes Region", eyes_region)#eye_image_resized)
        cv2.imshow("Frame", frame)
        time.sleep(0.5)
        ####################################################################



        ####################################################################
        input_image = Image.fromarray(cv2.cvtColor(eyes_region, cv2.COLOR_BGR2RGB))  # Convert to RGB
        input_image = transform(input_image).unsqueeze(0)  # Add batch dimension
        print(f"image shape: {input_image.shape}")
        input_image = input_image.to(device)
        with torch.no_grad():
            output = model(input_image)
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()
            print(f"\n\n-----\n\npredicted desire cell: {predicted_class}")
            ####################################################################



            ####################################################################
            desire_cell = predicted_class
            next_drawn_cell = next_cell(current_cell_base, desire_cell)
            print(f"        1. from current cell: {current_cell_base} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell)
            current_cell_base = next_drawn_cell
            #-------------------------------------------------------------------
            desire_cell = adjust_error_avg(predicted_class, error_map[predicted_class])
            next_drawn_cell = next_cell(current_cell_avg, desire_cell+1)
            print(f"        2. from current cell: {current_cell_avg} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (0, 255, 0))
            current_cell_avg = next_drawn_cell
            #-------------------------------------------------------------------
            desire_cell = adjust_error_max(predicted_class, error_map[predicted_class])
            next_drawn_cell = next_cell(current_cell_max, desire_cell+1)
            print(f"        3. from current cell: {current_cell_max} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (0, 255, 255))
            current_cell_max = next_drawn_cell
            #-------------------------------------------------------------------
            desire_cell = adjust_error_min(predicted_class, error_map[predicted_class])
            next_drawn_cell = next_cell(current_cell_min, desire_cell+1)
            print(f"        4. from current cell: {current_cell_min} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (255, 0, 0))
            current_cell_min = next_drawn_cell
            #-------------------------------------------------------------------
            # weighted prediction
            full_probabilities = F.softmax(output, dim=1)[0]
            top5_prob, top5_catid = torch.topk(full_probabilities, 5)
            desire_cell = top5_catid[0]*top5_prob[0] + top5_catid[1]*top5_prob[1] + top5_catid[2]*top5_prob[2] + top5_catid[3]*top5_prob[3] + top5_catid[4]*top5_prob[4]
            desire_cell = desire_cell.int()
            next_drawn_cell = next_cell(current_cell_weigh, desire_cell+1)
            print(f"        5. from current cell: {current_cell_weigh} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (255, 255, 0))
            current_cell_weigh = next_drawn_cell
            ####################################################################



            cv2.imshow("Grid", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

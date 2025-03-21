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
    1:  [12,10],2:  [-1,11],3:  [19,8],4:  [-3,7],5:  [8,19],6:  [61,18],7:  [49,6],8:  [5,16],9:  [4,15],10: [3,3],11: [13,13],12: [44,12],13: [-12,0],14: [-1,-3],15: [52,-2],16: [40,-3],17: [-4,-4],18: [6,6],19: [5,-6],20: [-7,77],21: [57,-8],22: [2,-9],23: [1,-10],24: [-11,0],25: [-12,-1],26: [-13,-25],27: [-14,-14],28: [39,69],29: [-16,-5],30: [-17,-17],31: [-20,-18],32: [-8,-8],33: [-9,-9],34: [-21,-23],35: [-11,-22],36: [-23,-23],37: [30,-24],38: [-37,-14],39: [-26,-38],40: [38,-27],41: [-17,-28],42: [-18,-18],43: [-30,-19],44: [-31,-43],45: [-32,-32],46: [-33,-33],47: [-23,-46],48: [-26,-35],49: [-38,-25],50: [-26,-49],51: [-27,-38],52: [-28,-51],53: [-42,-29],54: [-43,-41],55: [-42,-31],56: [-32,-43],57: [-46,-46],58: [-34,-47],59: [-35,-46],60: [-36,-36],61: [-48,-37],62: [-49,-49],63: [-50,-39],64: [14,-51],65: [-52,-52],66: [-42,-53],67: [-43,-43],68: [-55,-44],69: [-58,-56],70: [-59,-57],71: [-58,-47],72: [-48,-61],73: [-60,-49],74: [-61,23],75: [-8,-51],76: [-63,-65],77: [-66,-53],78: [-67,-43],79: [-44,-66],80: [-67,-67],81: [-68,-57],82: [-58,-58],83: [-48,-59],84: [-60,-60],85: [-50,-61],86: [-62,-62],87: [-63,-74],88: [-64,-64],89: [-65,-76],90: [-79,-89],91: [-78,-67],92: [-68,-79],93: [-71,4],94: [-70,-81],95: [-82,-82],96: [-83,-72],97: [-73,-73],98: [-74,-1],99: [-75,-86],100:[-87,-87]
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
            predicted_class = predicted_class.item() +1
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
            next_drawn_cell = next_cell(current_cell_avg, desire_cell)
            print(f"        2. from current cell: {current_cell_avg} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (0, 255, 0))
            current_cell_avg = next_drawn_cell
            #-------------------------------------------------------------------
            desire_cell = adjust_error_max(predicted_class, error_map[predicted_class])
            next_drawn_cell = next_cell(current_cell_max, desire_cell)
            print(f"        3. from current cell: {current_cell_max} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (0, 255, 255))
            current_cell_max = next_drawn_cell
            #-------------------------------------------------------------------
            desire_cell = adjust_error_min(predicted_class, error_map[predicted_class])
            next_drawn_cell = next_cell(current_cell_min, desire_cell)
            print(f"        4. from current cell: {current_cell_min} to {desire_cell}. End up in:")
            print(f"           -> next closest in path: {next_drawn_cell}\n")
            img = draw_square(next_drawn_cell, img, (255, 0, 0))
            current_cell_min = next_drawn_cell
            ####################################################################



            cv2.imshow("Grid", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

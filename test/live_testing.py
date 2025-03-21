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
########################################################################



########################################################################MODEL:
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class ResNetModel(ImageClassificationBase):
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, xb):
        return self.resnet(xb)
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
current_cell = 55
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT // GRID_ROWS
grid_img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Create a black canvas to overlay the grid
########################################################################



########################################################################GRID FUNCTIONS:
def draw_square(n):
    row = (n - 1) // GRID_COLS
    col = (n - 1) % GRID_COLS
    x, y = col * GRID_WIDTH, row * GRID_HEIGHT
    img = grid_img.copy()
    cv2.rectangle(img, (x, y), (x + GRID_WIDTH, y + GRID_HEIGHT), (0, 0, 255), 2)
    return img
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
            print(predicted_class)
            ####################################################################



            ####################################################################
            desire_cell = predicted_class + 1
            print(f"from {current_cell} to {desire_cell}. End up in:")
            next_drawn_cell = next_cell(current_cell, desire_cell)
            print(next_drawn_cell)
            img = draw_square(next_drawn_cell)
            cv2.imshow("Grid", img)
            current_cell = next_drawn_cell

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

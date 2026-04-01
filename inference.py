import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import ASLNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = ASLNet(num_classes=25)
model.load_state_dict(torch.load('asl_cnn_model.pth', map_location=device))
model.to(device)
model.eval()

# Transform for the input image (same as test_transform)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_letter(prediction):
    # The dataset maps A=0, B=1... skips J=9, so K=10.
    if prediction >= 9:
        return chr(prediction + 66)
    else:
        return chr(prediction + 65)

# OpenCV Webcam setup
cap = cv2.VideoCapture(0)

# ROI coordinates
top, right, bottom, left = 100, 300, 300, 500

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip frame horizontally for easier usability
    frame = cv2.flip(frame, 1)
    
    # Draw ROI rectangle on the original frame
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Extract ROI
    roi = frame[top:bottom, left:right]
    
    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    
    # Convert to PIL Image then apply transforms
    pil_image = Image.fromarray(resized)
    tensor_image = transform(pil_image).unsqueeze(0).to(device) # shape: (1, 1, 28, 28)
    
    # Model Inference
    with torch.no_grad():
        outputs = model(tensor_image)
        _, predicted = torch.max(outputs.data, 1)
        letter = get_letter(predicted.item())
        
    # Overlay Prediction
    cv2.putText(frame, f"Prediction: {letter}", (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
    cv2.putText(frame, "Place hand in the box", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
    cv2.imshow('Real-Time Sign Language Translator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

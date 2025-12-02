import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import tempfile
import os
from typing import List, Dict

app = FastAPI(title="Guitar Chord Recognition API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Load class mapping
with open('class_mapping_2.json', 'r') as f:
    class_mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    num_classes = len(idx_to_class)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)

checkpoint = torch.load('best_chord_model_2.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Image preprocessing transform (matching training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_strings_angle(image):
    """Detect guitar strings and compute rotation angle using Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    best_lines = None
    
    for threshold in range(30, 150, 10):
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        
        if lines is None:
            continue
        
        # Filter for horizontal-ish lines (guitar strings)
        filtered_lines = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            if 30 < angle_deg < 150:
                filtered_lines.append(line)
        
        if 4 <= len(filtered_lines) <= 8:
            best_lines = filtered_lines
            break
        
        if len(filtered_lines) > 0:
            best_lines = filtered_lines
    
    if best_lines is None or len(best_lines) == 0:
        return 0
    
    # Calculate median angle
    angles = [line[0][1] for line in best_lines]
    median_theta = np.median(angles)
    rotation_angle = np.degrees(median_theta - np.pi/2)
    
    return rotation_angle

def rotate_image(image, angle):
    """Rotate image by angle around center."""
    if abs(angle) < 0.5:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(0, 0, 0))
    
    return rotated

def detect_left_hand_bbox(image):
    """
    Detect LEFT hand (fretting hand) using MediaPipe.
    Returns (x_min, y_min, x_max, y_max) or None if not detected.
    Note: MediaPipe labels are from camera perspective (mirrored).
    For front-facing camera: "Right" label = your left hand
    """
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None
    
    # Find the LEFT hand (fretting hand)
    left_hand_landmarks = None
    for idx, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        
        # For front-facing camera (mirrored):
        # "Right" label means YOUR left hand (fretting hand)
        if label == "Right":
            left_hand_landmarks = results.multi_hand_landmarks[idx]
            break
    
    if left_hand_landmarks is None:
        return None
    
    # Get bounding box from landmarks
    x_coords = [landmark.x * w for landmark in left_hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in left_hand_landmarks.landmark]
    
    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))
    
    # Add 50% padding to capture fretboard and strings
    padding_x = int((x_max - x_min) * 0.5)
    padding_y = int((y_max - y_min) * 0.5)
    
    x_min = max(0, x_min - padding_x)
    x_max = min(w, x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(h, y_max + padding_y)
    
    return (x_min, y_min, x_max, y_max)

def crop_to_hand(image, bbox):
    """Crop image to hand bounding box."""
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    
    if x_max <= x_min or y_max <= y_min:
        return None
    
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def remove_black_borders(image, threshold=10):
    """Remove black borders (from rotation) by finding content bounding box."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rows = np.any(gray > threshold, axis=1)
    cols = np.any(gray > threshold, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return image
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return image[ymin:ymax+1, xmin:xmax+1]

def preprocess_frame(frame):
    """
    Preprocess a single frame following the training pipeline:
    1. Detect strings and rotate entire frame
    2. Detect left hand on rotated frame
    3. Crop to hand region
    4. Remove black borders
    5. Resize to 224x224
    """
    # Step 1: Detect strings and rotate
    angle = detect_strings_angle(frame)
    if abs(angle) > 0.5:
        rotated_frame = rotate_image(frame, angle)
    else:
        rotated_frame = frame
    
    # Step 2: Detect left hand on rotated frame
    bbox = detect_left_hand_bbox(rotated_frame)
    
    if bbox is None:
        return None
    
    # Step 3: Crop to hand region
    cropped_frame = crop_to_hand(rotated_frame, bbox)
    
    if cropped_frame is None or cropped_frame.size == 0:
        return None
    
    # Step 4: Remove black borders
    final_frame = remove_black_borders(cropped_frame)
    
    if final_frame.size == 0:
        return None
    
    return final_frame

def predict_chord(image):
    """Predict chord from preprocessed image. Returns predicted chord, confidence, and all probabilities."""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Apply transform
    tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    chord = idx_to_class[predicted.item()]
    confidence_score = confidence.item()
    
    # Get all probabilities as a dictionary
    all_probs = {}
    for idx, prob in enumerate(probabilities):
        chord_name = idx_to_class[idx]
        all_probs[chord_name] = round(prob.item(), 4)
    
    return chord, confidence_score, all_probs

@app.post("/predict")
async def predict_video(video: UploadFile = File(...)):
    """
    Process video and return chord predictions at 2 fps.
    Returns array of predictions with timestamps.
    """
    # Save uploaded video to temporary file
    # Handle both webm and mp4 formats
    suffix = '.webm' if video.content_type and 'webm' in video.content_type else '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        content = await video.read()
        tmp_file.write(content)
    
    try:
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default to 30 fps if unknown
        
        # Process at 2 fps (every 15 frames at 30fps)
        frame_interval = max(1, int(fps / 2))
        
        predictions = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame_interval frames
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Preprocess frame
                preprocessed = preprocess_frame(frame)
                
                if preprocessed is not None:
                    # Predict chord
                    chord, confidence, all_probs = predict_chord(preprocessed)
                    predictions.append({
                        "timestamp": round(timestamp, 2),
                        "chord": chord,
                        "confidence": round(confidence, 2),
                        "probabilities": all_probs
                    })
                else:
                    # No hand detected
                    predictions.append({
                        "timestamp": round(timestamp, 2),
                        "chord": "No hand detected",
                        "confidence": 0.0,
                        "probabilities": {}
                    })
            
            frame_count += 1
        
        cap.release()
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/")
async def root():
    return {"message": "Guitar Chord Recognition API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


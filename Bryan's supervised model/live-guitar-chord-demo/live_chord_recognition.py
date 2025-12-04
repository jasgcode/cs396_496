#!/usr/bin/env python3
"""
Live Guitar Chord Recognition Application
Real-time chord detection from webcam feed using MobileNetV2
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import time
from collections import deque
from pathlib import Path
import threading
from queue import Queue

# Configuration
# Get script directory and resolve model path relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "guitar-chord-demo", "backend")
MODEL_DIR = os.path.abspath(MODEL_DIR)  # Resolve to absolute path

WINDOW_SIZE = 5  # Last 5 predictions for smoothing
MIN_VOTES = 3  # Minimum votes to change chord
MIN_DISPLAY_TIME = 0.5  # Minimum 0.5 seconds before allowing change
MAX_UPDATE_RATE = 1.0  # Maximum 1 update per second
DEFAULT_CAMERA_INDEX = 0  # Default camera
PROCESS_EVERY_N_FRAMES = 1  # Process every 3rd frame to reduce load

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Load class mapping
class_mapping_path = os.path.join(MODEL_DIR, 'class_mapping_2.json')
if not os.path.exists(class_mapping_path):
    raise FileNotFoundError(
        f"Model file not found: {class_mapping_path}\n"
        f"Please ensure the model files are in: {MODEL_DIR}"
    )

with open(class_mapping_path, 'r') as f:
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

model_path = os.path.join(MODEL_DIR, 'best_chord_model_2.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        f"Please ensure the model files are in: {MODEL_DIR}"
    )

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

print(f"✓ Model directory: {MODEL_DIR}")
print(f"✓ Model loaded from: {model_path}")
print(f"✓ Device: {device}")
print(f"✓ Classes: {num_classes}")

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocessing functions (same as backend)
def detect_strings_angle(image):
    """Detect guitar strings and compute rotation angle using Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    best_lines = None
    
    for threshold in range(30, 150, 10):
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        
        if lines is None:
            continue
        
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

def detect_left_hand_bbox(image, camera_type='user'):
    """Detect LEFT hand (fretting hand) using MediaPipe."""
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None
    
    target_label = "Right" if camera_type == "user" else "Left"
    
    left_hand_landmarks = None
    for idx, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        if label == target_label:
            left_hand_landmarks = results.multi_hand_landmarks[idx]
            break
    
    if left_hand_landmarks is None:
        return None
    
    x_coords = [landmark.x * w for landmark in left_hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in left_hand_landmarks.landmark]
    
    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))
    
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

def preprocess_frame(frame, camera_type='user'):
    """Preprocess a single frame following the training pipeline."""
    angle = detect_strings_angle(frame)
    if abs(angle) > 0.5:
        rotated_frame = rotate_image(frame, angle)
    else:
        rotated_frame = frame
    
    bbox = detect_left_hand_bbox(rotated_frame, camera_type)
    
    if bbox is None:
        return None
    
    cropped_frame = crop_to_hand(rotated_frame, bbox)
    
    if cropped_frame is None or cropped_frame.size == 0:
        return None
    
    final_frame = remove_black_borders(cropped_frame)
    
    if final_frame.size == 0:
        return None
    
    return final_frame

def predict_chord(image):
    """Predict chord from preprocessed image. Returns chord, confidence, and all probabilities."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    chord = idx_to_class[predicted.item()]
    confidence_score = confidence.item()
    
    all_probs = {}
    for idx, prob in enumerate(probabilities):
        chord_name = idx_to_class[idx]
        all_probs[chord_name] = prob.item()
    
    return chord, confidence_score, all_probs

def smooth_predictions(recent_predictions, current_time, current_chord, chord_start_time):
    """
    Apply smoothing logic:
    - Use last WINDOW_SIZE predictions
    - Need MIN_VOTES to change
    - MIN_DISPLAY_TIME temporal persistence
    """
    if not recent_predictions:
        return None, current_chord, chord_start_time
    
    # Filter out "No hand detected"
    valid_predictions = [p for p in recent_predictions if p['chord'] and p['chord'] != 'No hand detected']
    
    if not valid_predictions:
        return None, current_chord, chord_start_time
    
    # Get last WINDOW_SIZE predictions
    window_predictions = valid_predictions[-WINDOW_SIZE:]
    
    # Count votes
    vote_count = {}
    for pred in window_predictions:
        chord = pred['chord']
        vote_count[chord] = vote_count.get(chord, 0) + 1
    
    # Find majority chord
    if not vote_count:
        return None, current_chord, chord_start_time
    
    majority_chord = max(vote_count.items(), key=lambda x: x[1])
    chord_name, votes = majority_chord
    
    # Check if we have enough votes
    if votes < MIN_VOTES:
        # Not enough consensus, keep current chord if it exists
        return None, current_chord, chord_start_time
    
    # Temporal persistence check
    if current_chord == chord_name:
        # Same chord, return latest prediction for this chord
        for pred in reversed(window_predictions):
            if pred['chord'] == chord_name:
                return pred, current_chord, chord_start_time
    
    # Different chord - check minimum display time
    if current_chord is not None:
        time_since_change = current_time - chord_start_time
        if time_since_change < MIN_DISPLAY_TIME:
            # Not enough time has passed, keep current chord
            return None, current_chord, chord_start_time
    
    # Update to new chord
    for pred in reversed(window_predictions):
        if pred['chord'] == chord_name:
            return pred, chord_name, current_time
    
    return None, current_chord, chord_start_time

def get_top_chords(probabilities, top_n=3):
    """Get top N chords by probability."""
    sorted_chords = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_chords[:top_n]

def draw_overlay(frame, chord, confidence, top_chords):
    """Draw chord overlay and side panel on frame."""
    h, w = frame.shape[:2]
    
    # Main chord overlay (top-left)
    if chord and chord != "No hand detected":
        # Background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Chord name (large, bold)
        font_scale = 3.0
        thickness = 5
        text = chord.upper()
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        
        # Gradient-like color (purple)
        color = (102, 126, 234)  # BGR format
        cv2.putText(frame, text, (30, 120), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        
        # Confidence
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # No hand detected
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "No hand detected", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Side panel for top 3 chords
    panel_width = 300
    panel_x = w - panel_width - 20
    panel_y = 20
    panel_height = 200
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w - 20, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Title
    cv2.putText(frame, "Top 3 Chords", (panel_x + 10, panel_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Top chords
    y_offset = panel_y + 60
    for i, (chord_name, prob) in enumerate(top_chords):
        text = f"{i+1}. {chord_name}: {prob:.1%}"
        color = (102, 126, 234) if i == 0 else (200, 200, 200)
        cv2.putText(frame, text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 35
    
    return frame

def main():
    """Main application loop."""
    print("\n" + "="*80)
    print("LIVE GUITAR CHORD RECOGNITION")
    print("="*80)
    print(f"Press 'q' to quit")
    print(f"Press 'c' to change camera")
    print("="*80 + "\n")
    
    # Initialize camera
    camera_index = DEFAULT_CAMERA_INDEX
    print(f"Attempting to open camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        print("Trying camera 1...")
        camera_index = 1
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open any camera")
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Detect camera type (assume front-facing by default)
    camera_type = 'user'
    
    # Shared state for threading
    recent_predictions = deque(maxlen=WINDOW_SIZE * 2)
    display_state = {
        'chord': None,
        'confidence': 0.0,
        'probabilities': {},
        'chord_start_time': 0.0,
        'last_update_time': 0.0
    }
    state_lock = threading.Lock()
    frame_queue = Queue(maxsize=2)  # Small queue to avoid lag
    processing_active = threading.Event()
    processing_active.set()
    
    frame_count = 0
    fps_start_time = time.time()
    process_frame_count = 0
    
    def process_frame_worker():
        """Worker thread to process frames asynchronously."""
        while processing_active.is_set():
            try:
                # Get frame from queue (with timeout to check if we should stop)
                try:
                    frame_data = frame_queue.get(timeout=0.1)
                    frame, current_time = frame_data
                except:
                    continue
                
                # Process frame
                try:
                    preprocessed = preprocess_frame(frame, camera_type)
                except Exception as e:
                    preprocessed = None
                
                if preprocessed is not None:
                    # Predict chord
                    try:
                        chord, confidence, probs = predict_chord(preprocessed)
                    except Exception as e:
                        chord = "Error"
                        confidence = 0.0
                        probs = {}
                    
                    # Add to recent predictions
                    with state_lock:
                        recent_predictions.append({
                            'chord': chord,
                            'confidence': confidence,
                            'probabilities': probs,
                            'time': current_time
                        })
                        
                        # Apply smoothing
                        smoothed_pred, new_chord, new_start_time = smooth_predictions(
                            list(recent_predictions),
                            current_time,
                            display_state['chord'],
                            display_state['chord_start_time']
                        )
                        
                        # Update display state if prediction changed
                        if smoothed_pred is not None:
                            time_since_last_update = current_time - display_state['last_update_time']
                            if new_chord != display_state['chord'] or time_since_last_update >= MAX_UPDATE_RATE:
                                display_state['chord'] = new_chord
                                display_state['confidence'] = smoothed_pred['confidence']
                                display_state['probabilities'] = smoothed_pred['probabilities']
                                display_state['chord_start_time'] = new_start_time
                                display_state['last_update_time'] = current_time
                else:
                    # No hand detected
                    with state_lock:
                        recent_predictions.append({
                            'chord': 'No hand detected',
                            'confidence': 0.0,
                            'probabilities': {},
                            'time': current_time
                        })
                
                frame_queue.task_done()
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_frame_worker, daemon=True)
    processing_thread.start()
    
    print("Starting live recognition...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            current_time = time.time()
            frame_count += 1
            
            # Only process every Nth frame to reduce load
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                process_frame_count += 1
                # Add frame to processing queue (non-blocking, drop if queue full)
                if not frame_queue.full():
                    frame_queue.put((frame.copy(), current_time))
            
            # Get current display state (thread-safe)
            with state_lock:
                current_chord = display_state['chord']
                current_confidence = display_state['confidence']
                current_probabilities = display_state['probabilities'].copy()
            
            # Get top 3 chords
            if current_probabilities:
                top_chords = get_top_chords(current_probabilities, top_n=3)
            else:
                top_chords = []
            
            # Draw overlay (always smooth, no blocking)
            display_frame = draw_overlay(frame, current_chord, current_confidence, top_chords)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame (smooth, no blocking)
            cv2.imshow('Live Guitar Chord Recognition', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Switch camera
                cap.release()
                camera_index = 1 - camera_index
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    print(f"Error: Could not open camera {camera_index}")
                    break
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"Switched to camera {camera_index}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Stop processing thread
        processing_active.clear()
        # Wait for queue to empty
        frame_queue.join()
        processing_thread.join(timeout=1.0)
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n✓ Application closed")

if __name__ == "__main__":
    main()


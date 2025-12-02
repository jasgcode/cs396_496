# Guitar Chord Recognition Demo

A full-stack web application for real-time guitar chord recognition. Record yourself playing chords and see real-time predictions overlaid on video playback.

## Architecture

- **Frontend**: React web application with webcam access
- **Backend**: FastAPI server with PyTorch MobileNetV2 model
- **Model**: Pre-trained model for 13 guitar chord classification (MobileNetV2)

## Quick Start

### Kill Existing Processes (if needed)

If ports 3000 or 8000 are already in use, kill the processes first:

```bash
# Kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000 (frontend)
lsof -ti:3000 | xargs kill -9
```

Or manually:
```bash
# Find process on port 8000
lsof -ti:8000

# Kill it (replace PID with the number from above)
kill -9 <PID>

# Same for port 3000
lsof -ti:3000
kill -9 <PID>
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install  # Only needed first time or when dependencies change
npm start
```

The frontend will open at `http://localhost:3000` and connect to the backend at `http://localhost:8000`.

## Features

- ✅ Webcam recording with MediaRecorder API
- ✅ Real-time video processing at 2 fps
- ✅ MediaPipe hand detection and preprocessing
- ✅ Chord prediction with confidence scores and full probability distributions
- ✅ Video playback with chord overlay (top-left corner)
- ✅ Probabilities panel showing all 13 chord probabilities (bottom panel)
- ✅ Prediction smoothing (majority vote over last 5 predictions)
- ✅ Accuracy tracking with ground truth chord sequences
- ✅ Chord sequence selection (G->D->Am, F->C->D->C)
- ✅ Modern, dark-themed UI

## Supported Chords

A, Am, B, Bm, C, Cm, D, Dm, Em, F, Fm, G, Gm (13 classes - note: "E" is not included in the current model)

## Processing Pipeline

1. Video recorded in browser (WebM format)
2. Sent to backend for processing
3. Frames extracted at 2 fps intervals
4. Each frame preprocessed:
   - Rotate to align guitar strings horizontally
   - Detect left hand (fretting hand) with MediaPipe
   - Crop to hand region with 50% padding
   - Remove black borders
   - Resize to 224x224
   - Normalize with ImageNet statistics
5. Predict chord using MobileNetV2 model
6. Return predictions with timestamps
7. Display overlay during video playback

## File Structure

```
guitar-chord-demo/
├── backend/
│   ├── app.py                    # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── best_chord_model_2.pth    # Current model (13 classes) - IN USE
│   ├── class_mapping_2.json      # Current class mapping (13 classes) - IN USE
│   ├── best_chord_model_final.pth # Old model (14 classes) - kept for reference
│   ├── class_mapping_final.json   # Old class mapping (14 classes) - kept for reference
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CameraRecorder.jsx
│   │   │   ├── VideoPlayer.jsx
│   │   │   ├── ChordOverlay.jsx
│   │   │   └── ProbabilitiesPanel.jsx
│   │   ├── App.jsx
│   │   ├── index.js
│   │   └── index.css
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   └── README.md
└── README.md
```

## Model Files

**Currently Active:**
- `backend/best_chord_model_2.pth` - MobileNetV2 model trained on 13 chord classes
- `backend/class_mapping_2.json` - Maps 13 chord classes (A, Am, B, Bm, C, Cm, D, Dm, Em, F, Fm, G, Gm)

**Legacy Files (kept for reference):**
- `backend/best_chord_model_final.pth` - Previous model with 14 classes (includes "E")
- `backend/class_mapping_final.json` - Previous mapping with 14 classes

## Requirements

- Python 3.8+
- Node.js 14+
- Webcam access
- Modern web browser with MediaRecorder support

## Notes

- Video processing happens at 2 fps (every 0.5 seconds) for performance
- MediaPipe preprocessing matches the training pipeline exactly
- Chord overlay shows "No hand detected" when hand is not found
- Video is mirrored for natural viewing experience
- Predictions are smoothed using majority vote over last 5 predictions (2.5 seconds)
- Minimum display time of 0.5 seconds prevents rapid flickering
- Full probability distributions are returned for all 13 chord classes


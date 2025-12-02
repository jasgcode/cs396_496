# Guitar Chord Recognition Demo

A full-stack web application for real-time guitar chord recognition. Record yourself playing chords and see real-time predictions overlaid on video playback.

## Architecture

- **Frontend**: React web application with webcam access
- **Backend**: FastAPI server with PyTorch MobileNetV2 model
- **Model**: Pre-trained model for 14 guitar chord classification

## Quick Start

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

The frontend will open at `http://localhost:3000` and connect to the backend at `http://localhost:8000`.

## Features

- ✅ Webcam recording with MediaRecorder API
- ✅ Real-time video processing at 2 fps
- ✅ MediaPipe hand detection and preprocessing
- ✅ Chord prediction with confidence scores
- ✅ Video playback with chord overlay
- ✅ Smooth transitions between chord changes
- ✅ Modern, dark-themed UI

## Supported Chords

A, Am, B, Bm, C, Cm, D, Dm, E, Em, F, Fm, G, Gm

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
│   ├── best_chord_model_final.pth # Trained model
│   ├── class_mapping_final.json   # Class mappings
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CameraRecorder.jsx
│   │   │   ├── VideoPlayer.jsx
│   │   │   └── ChordOverlay.jsx
│   │   ├── App.jsx
│   │   ├── index.js
│   │   └── index.css
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   └── README.md
└── README.md
```

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


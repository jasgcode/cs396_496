# Guitar Chord Recognition Backend

FastAPI backend for processing video and recognizing guitar chords.

## Setup

1. **Kill any existing process on port 8000** (if needed):
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or manually:
lsof -ti:8000  # Get the PID
kill -9 <PID>  # Replace <PID> with the number
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the model files in the backend directory:
   - `best_chord_model_2.pth` (new model, 13 classes - currently in use)
   - `class_mapping_2.json` (new mapping, 13 classes - currently in use)
   - `best_chord_model_final.pth` (old model, 14 classes - kept for reference)
   - `class_mapping_final.json` (old mapping, 14 classes - kept for reference)

4. Run the server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Upload video file and get chord predictions

### POST /predict

Upload a video file and receive chord predictions at 2 fps intervals.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `video` (file)

**Response:**
```json
[
  {
    "timestamp": 0.5,
    "chord": "Am",
    "confidence": 0.95
  },
  {
    "timestamp": 1.0,
    "chord": "C",
    "confidence": 0.87
  }
]
```

## Processing Pipeline

1. Video is processed at 2 fps (every 0.5 seconds)
2. For each frame:
   - Detect guitar strings and rotate frame to align horizontally
   - Detect left hand (fretting hand) using MediaPipe
   - Crop to hand bounding box with 50% padding
   - Remove black borders
   - Resize to 224x224
   - Normalize with ImageNet mean/std
   - Predict chord using MobileNetV2 model


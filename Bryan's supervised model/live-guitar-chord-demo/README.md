# Live Guitar Chord Recognition

Real-time guitar chord recognition application using MobileNetV2 and OpenCV.

## Features

- **Real-time processing**: Processes every frame from webcam feed
- **Prediction smoothing**: Uses sliding window (5 predictions) with majority voting
- **Live display**: Shows current chord with confidence and top 3 predictions
- **Fast startup**: Standalone Python application, no web server needed
- **Same preprocessing**: Uses identical preprocessing pipeline as training

## Requirements

- Python 3.8+
- Webcam
- Model files from `../guitar-chord-demo/backend/`:
  - `best_chord_model_2.pth`
  - `class_mapping_2.json`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model files are in the correct location:
   - The script expects model files in `../guitar-chord-demo/backend/`
   - Or update `MODEL_DIR` in `live_chord_recognition.py`

## Usage

Run the application:
```bash
python live_chord_recognition.py
```

### Controls

- **'q'**: Quit application
- **'c'**: Switch between cameras (if multiple available)

## How It Works

1. **Frame Processing**: Captures frames from webcam and processes every frame
2. **Preprocessing**: 
   - Detects guitar strings and rotates frame
   - Detects left hand (fretting hand) using MediaPipe
   - Crops to hand region with 50% padding
   - Removes black borders
3. **Prediction**: Uses MobileNetV2 to predict chord from preprocessed frame
4. **Smoothing**: 
   - Maintains sliding window of last 5 predictions
   - Requires 3 votes (majority) to change chord
   - Enforces 0.5s minimum display time
   - Updates display max once per second
5. **Display**: 
   - Shows live video feed
   - Overlays current chord (top-left) with confidence
   - Side panel shows top 3 chords with probabilities

## Configuration

Edit constants at the top of `live_chord_recognition.py`:

- `WINDOW_SIZE`: Number of predictions in sliding window (default: 5)
- `MIN_VOTES`: Minimum votes needed to change chord (default: 3)
- `MIN_DISPLAY_TIME`: Minimum time before allowing chord change (default: 0.5s)
- `MAX_UPDATE_RATE`: Maximum display update rate (default: 1.0s)
- `CAMERA_INDEX`: Camera to use (default: 0)

## Performance

- **Processing**: Every frame is processed
- **Display Updates**: Maximum 1 update per second (throttled)
- **Smoothing**: Uses same logic as video demo (5 predictions, 3 votes, 0.5s persistence)

## Troubleshooting

- **Camera not found**: Try changing `CAMERA_INDEX` or use 'c' key to switch cameras
- **Model not found**: Check that `MODEL_DIR` points to correct location
- **Low FPS**: Processing every frame is computationally intensive. Consider reducing resolution or processing every Nth frame.


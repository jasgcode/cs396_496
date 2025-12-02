# Guitar Chord Recognition Frontend

React web application for real-time guitar chord recognition demo.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Features

- **Camera Recording**: Access webcam and record video
- **Real-time Processing**: Send video to backend for chord recognition
- **Playback with Overlay**: Play recorded video with chord predictions overlaid
- **Smooth Transitions**: Chord overlay updates smoothly during playback

## Usage

1. Click "Start Recording" to begin
2. Grant camera permissions when prompted
3. Click "Start Recording" again to begin recording
4. Click "Stop Recording" when done
5. Wait for processing (chords are analyzed at 2 fps)
6. Watch playback with chord overlay
7. Click "Record Again" to start over

## Components

- **App.jsx**: Main app component with state management
- **CameraRecorder.jsx**: Handles webcam access and video recording
- **VideoPlayer.jsx**: Plays back recorded video with controls
- **ChordOverlay.jsx**: Displays current detected chord

## API Integration

The frontend communicates with the backend at `http://localhost:8000/predict` to process videos and get chord predictions.


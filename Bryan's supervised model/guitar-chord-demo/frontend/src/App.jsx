import React, { useState } from 'react';
import CameraRecorder from './components/CameraRecorder';
import VideoPlayer from './components/VideoPlayer';
import './App.css';

const API_URL = 'http://localhost:8000';

async function predictChords(videoBlob) {
  const formData = new FormData();
  formData.append('video', videoBlob, 'recording.webm');
  
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`Backend error: ${response.statusText}`);
  }
  
  return await response.json();
}

function App() {
  const [recording, setRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleRecordingComplete = async (videoBlob) => {
    setRecordedVideo(videoBlob);
    setProcessing(true);
    setError(null);
    
    try {
      const results = await predictChords(videoBlob);
      setPredictions(results);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'Failed to process video. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  const handleRecordAgain = () => {
    setRecordedVideo(null);
    setPredictions(null);
    setError(null);
    setRecording(false);
    
    // Clean up video blob
    if (recordedVideo) {
      URL.revokeObjectURL(URL.createObjectURL(recordedVideo));
    }
  };

  const handleStartRecording = () => {
    setRecording(true);
    setError(null);
  };

  return (
    <div className="App">
      {!recording && !recordedVideo && (
        <div className="start-screen">
          <h1 className="app-title">Guitar Chord Recognition</h1>
          <p className="app-subtitle">Record yourself playing chords and see real-time recognition</p>
          <button 
            className="start-button"
            onClick={handleStartRecording}
          >
            Start Recording
          </button>
        </div>
      )}

      {recording && !recordedVideo && (
        <CameraRecorder 
          onRecordingComplete={handleRecordingComplete}
          onCancel={() => setRecording(false)}
        />
      )}

      {processing && (
        <div className="processing-screen">
          <div className="spinner"></div>
          <h2>Analyzing chords...</h2>
          <p>This may take a few moments</p>
        </div>
      )}

      {error && (
        <div className="error-screen">
          <h2>Error</h2>
          <p>{error}</p>
          <button className="retry-button" onClick={handleRecordAgain}>
            Try Again
          </button>
        </div>
      )}

      {recordedVideo && predictions && !processing && (
        <VideoPlayer 
          videoBlob={recordedVideo}
          predictions={predictions}
          onRecordAgain={handleRecordAgain}
        />
      )}
    </div>
  );
}

export default App;


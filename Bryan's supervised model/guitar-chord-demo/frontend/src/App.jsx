import React, { useState } from 'react';
import CameraRecorder from './components/CameraRecorder';
import VideoPlayer from './components/VideoPlayer';
import './App.css';

const API_URL = 'http://localhost:8000';

async function predictChords(videoBlob) {
  const formData = new FormData();
  formData.append('video', videoBlob, 'recording.webm');
  
  try {
    // Create abort controller for timeout (5 minutes)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes
    
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      let errorText = '';
      try {
        errorText = await response.text();
      } catch (e) {
        errorText = response.statusText;
      }
      throw new Error(`Backend error (${response.status}): ${errorText || response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. The video may be too long or the server is processing. Please try a shorter recording.');
    } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      throw new Error('Cannot connect to backend server. Make sure the backend is running on http://localhost:8000');
    }
    throw error;
  }
}

const CHORD_SEQUENCES = {
  'G->D->Am': ['G', 'D', 'Am'],
  'F->C->D->C': ['F', 'C', 'D', 'C']
};

function App() {
  const [recording, setRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSequence, setSelectedSequence] = useState('');
  const [selectedChords, setSelectedChords] = useState([]);

  const handleRecordingComplete = async (videoBlob) => {
    setRecordedVideo(videoBlob);
    setProcessing(true);
    setError(null);
    
    console.log('Video blob size:', (videoBlob.size / 1024 / 1024).toFixed(2), 'MB');
    
    // First, check if backend is reachable
    try {
      console.log('Checking backend connectivity...');
      const healthCheck = await fetch(`${API_URL}/health`);
      if (!healthCheck.ok) {
        throw new Error(`Backend health check failed: ${healthCheck.status}`);
      }
      console.log('Backend is reachable');
    } catch (err) {
      console.error('Backend connectivity error:', err);
      setError(`Cannot connect to backend server. Make sure the backend is running on ${API_URL}. Error: ${err.message}`);
      setProcessing(false);
      return;
    }
    
    console.log('Sending video to backend...');
    
    try {
      const results = await predictChords(videoBlob);
      console.log('Received predictions:', results.length, 'predictions');
      setPredictions(results);
    } catch (err) {
      console.error('Prediction error:', err);
      console.error('Error details:', {
        name: err.name,
        message: err.message,
        stack: err.stack
      });
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
    setSelectedSequence('');
    setSelectedChords([]);
    
    // Clean up video blob
    if (recordedVideo) {
      URL.revokeObjectURL(URL.createObjectURL(recordedVideo));
    }
  };

  const handleStartRecording = () => {
    if (selectedSequence) {
      setSelectedChords(CHORD_SEQUENCES[selectedSequence]);
      setRecording(true);
      setError(null);
    }
  };

  const handleSequenceSelect = (sequence) => {
    setSelectedSequence(sequence);
    setSelectedChords(CHORD_SEQUENCES[sequence]);
  };

  return (
    <div className="App">
      {!recording && !recordedVideo && (
        <div className="start-screen">
          <h1 className="app-title">Guitar Chord Recognition</h1>
          <p className="app-subtitle">Record yourself playing chords and see real-time recognition</p>
          
          <div className="chord-selection">
            <label className="chord-select-label">
              Select chord sequence you will play (for accuracy calculation):
            </label>
            <div className="sequence-buttons">
              {Object.keys(CHORD_SEQUENCES).map(sequence => (
                <button
                  key={sequence}
                  className={`sequence-button ${selectedSequence === sequence ? 'selected' : ''}`}
                  onClick={() => handleSequenceSelect(sequence)}
                >
                  {sequence}
                </button>
              ))}
            </div>
            {selectedSequence && (
              <p className="selected-sequence-info">
                Selected: {CHORD_SEQUENCES[selectedSequence].join(' → ')}
              </p>
            )}
          </div>

          <button 
            className="start-button"
            onClick={handleStartRecording}
            disabled={!selectedSequence}
          >
            Start Recording
          </button>
          {!selectedSequence && (
            <p className="warning-text">Please select a chord sequence before recording</p>
          )}
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
          groundTruthChords={selectedChords}
        />
      )}
    </div>
  );
}

export default App;


import React, { useRef, useEffect, useState } from 'react';
import './CameraRecorder.css';

function CameraRecorder({ onRecordingComplete, onCancel }) {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream = null;
    let intervalId = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user' // Front-facing camera
          },
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
        setError('Could not access camera. Please check permissions.');
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []);

  const startRecording = () => {
    if (!videoRef.current || !videoRef.current.srcObject) {
      setError('Camera not available');
      return;
    }

    const stream = videoRef.current.srcObject;
    const options = {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 2500000
    };

    try {
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        onRecordingComplete(blob);
      };

      mediaRecorder.start();
      setRecording(true);
      setDuration(0);

      // Update duration every second
      const intervalId = setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);

      // Store interval ID to clear it later
      mediaRecorderRef.current.intervalId = intervalId;
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Could not start recording. Please try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      if (mediaRecorderRef.current.intervalId) {
        clearInterval(mediaRecorderRef.current.intervalId);
      }
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="camera-recorder">
      {error && (
        <div className="camera-error">
          <p>{error}</p>
          <button onClick={onCancel}>Go Back</button>
        </div>
      )}
      
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="camera-preview"
      />

      <div className="recording-controls">
        {!recording ? (
          <>
            <button className="start-recording-button" onClick={startRecording}>
              Start Recording
            </button>
            <button className="cancel-button" onClick={onCancel}>
              Cancel
            </button>
          </>
        ) : (
          <>
            <div className="recording-indicator">
              <span className="recording-dot"></span>
              <span className="recording-text">Recording: {formatDuration(duration)}</span>
            </div>
            <button className="stop-recording-button" onClick={stopRecording}>
              Stop Recording
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default CameraRecorder;


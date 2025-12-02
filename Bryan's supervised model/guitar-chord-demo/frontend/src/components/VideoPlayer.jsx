import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import ChordOverlay from './ChordOverlay';
import ProbabilitiesPanel from './ProbabilitiesPanel';
import './VideoPlayer.css';

function VideoPlayer({ videoBlob, predictions, onRecordAgain, groundTruthChords }) {
  const videoRef = useRef(null);
  const [currentChord, setCurrentChord] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [panelHeight, setPanelHeight] = useState(300); // Default panel height

  const updateChord = useCallback((time) => {
    if (!predictions || predictions.length === 0) {
      setCurrentChord(null);
      return;
    }

    // Find the prediction closest to current time
    let closest = predictions[0];
    let minDiff = Math.abs(predictions[0].timestamp - time);

    for (let i = 1; i < predictions.length; i++) {
      const diff = Math.abs(predictions[i].timestamp - time);
      if (diff < minDiff) {
        minDiff = diff;
        closest = predictions[i];
      }
    }

    // Only update if within 0.5 seconds
    if (minDiff <= 0.5) {
      setCurrentChord(closest);
    } else {
      setCurrentChord(null);
    }
  }, [predictions]);

  useEffect(() => {
    if (videoRef.current && videoBlob) {
      const url = URL.createObjectURL(videoBlob);
      videoRef.current.src = url;

      const video = videoRef.current;

      const handleLoadedMetadata = () => {
        setDuration(video.duration);
      };

      const handleTimeUpdate = () => {
        setCurrentTime(video.currentTime);
        updateChord(video.currentTime);
      };

      const handlePlay = () => setIsPlaying(true);
      const handlePause = () => setIsPlaying(false);

      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);

      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('timeupdate', handleTimeUpdate);
        video.removeEventListener('play', handlePlay);
        video.removeEventListener('pause', handlePause);
        URL.revokeObjectURL(url);
      };
    }
  }, [videoBlob, updateChord]);

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
    }
  };

  const handleSeek = (e) => {
    if (videoRef.current) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = x / rect.width;
      const newTime = percent * duration;
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
      updateChord(newTime);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Calculate accuracy based on ground truth
  const accuracy = useMemo(() => {
    if (!groundTruthChords || groundTruthChords.length === 0 || !predictions) {
      return null;
    }

    const groundTruthSet = new Set(groundTruthChords);
    let correct = 0;
    let total = 0;

    predictions.forEach(pred => {
      // Skip "No hand detected" predictions
      if (pred.chord && pred.chord !== 'No hand detected') {
        total++;
        if (groundTruthSet.has(pred.chord)) {
          correct++;
        }
      }
    });

    return {
      correct,
      total,
      percentage: total > 0 ? (correct / total) * 100 : 0
    };
  }, [groundTruthChords, predictions]);

  return (
    <div className="video-player-container">
      <div className="video-section" style={{ height: `calc(100vh - ${panelHeight}px)` }}>
        <div className="video-wrapper">
          <video
            ref={videoRef}
            className="playback-video"
            playsInline
          />
          <ChordOverlay chord={currentChord} />
        </div>

        <div className="video-controls">
          <button className="play-pause-button" onClick={togglePlayPause}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          
          <div className="scrubber-container" onClick={handleSeek}>
            <div className="scrubber-track">
              <div 
                className="scrubber-progress" 
                style={{ width: `${(currentTime / duration) * 100}%` }}
              />
            </div>
          </div>

          <div className="time-display">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>

          <button className="record-again-button" onClick={onRecordAgain}>
            Record Again
          </button>
        </div>
      </div>

      <div className="probabilities-section" style={{ height: `${panelHeight}px` }}>
        <div className="panel-resizer" 
          onMouseDown={(e) => {
            e.preventDefault();
            const startY = e.clientY;
            const startHeight = panelHeight;

            const handleMouseMove = (e) => {
              const deltaY = startY - e.clientY;
              const newHeight = Math.max(200, Math.min(600, startHeight + deltaY));
              setPanelHeight(newHeight);
            };

            const handleMouseUp = () => {
              document.removeEventListener('mousemove', handleMouseMove);
              document.removeEventListener('mouseup', handleMouseUp);
            };

            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
          }}
        />
        <ProbabilitiesPanel currentPrediction={currentChord} accuracy={accuracy} />
      </div>
    </div>
  );
}

export default VideoPlayer;


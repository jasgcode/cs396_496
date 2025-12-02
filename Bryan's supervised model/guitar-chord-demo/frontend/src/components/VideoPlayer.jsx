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
  
  // Smoothing state
  const recentPredictionsRef = useRef([]); // Last 5 predictions
  const currentDisplayedChordRef = useRef(null);
  const chordDisplayStartTimeRef = useRef(0);
  const MIN_DISPLAY_TIME = 0.5; // Minimum 0.5 seconds before allowing change
  const WINDOW_SIZE = 5; // Last 5 predictions
  const MIN_VOTES = 3; // Need at least 3 votes to change

  const updateChord = useCallback((time) => {
    if (!predictions || predictions.length === 0) {
      setCurrentChord(null);
      recentPredictionsRef.current = [];
      return;
    }

    // Reset if time jumped backwards significantly (video looped or seeked)
    if (chordDisplayStartTimeRef.current > 0 && time < chordDisplayStartTimeRef.current - 1.0) {
      recentPredictionsRef.current = [];
      currentDisplayedChordRef.current = null;
      chordDisplayStartTimeRef.current = time;
    }

    // Find all predictions within the time window (last 2.5 seconds at 2 fps = 5 predictions)
    const timeWindow = 2.5; // seconds
    const relevantPredictions = predictions.filter(p => 
      p.timestamp >= 0 && // Only future/present predictions
      p.timestamp <= time && // Only predictions up to current time
      (time - p.timestamp) <= timeWindow // Within time window
    );

    if (relevantPredictions.length === 0) {
      setCurrentChord(null);
      recentPredictionsRef.current = [];
      return;
    }

    // Get the last WINDOW_SIZE predictions (most recent)
    const recentPredictions = relevantPredictions
      .slice(-WINDOW_SIZE)
      .filter(p => p.chord && p.chord !== 'No hand detected'); // Ignore "No hand detected" in voting

    // Update the recent predictions ref
    recentPredictionsRef.current = recentPredictions;

    if (recentPredictions.length === 0) {
      setCurrentChord(null);
      return;
    }

    // Count votes for each chord
    const voteCount = {};
    recentPredictions.forEach(p => {
      voteCount[p.chord] = (voteCount[p.chord] || 0) + 1;
    });

    // Find chord with most votes
    let majorityChord = null;
    let maxVotes = 0;
    for (const [chord, votes] of Object.entries(voteCount)) {
      if (votes > maxVotes) {
        maxVotes = votes;
        majorityChord = chord;
      }
    }

    // Check if we have enough votes (at least MIN_VOTES)
    if (maxVotes < MIN_VOTES) {
      // Not enough consensus, keep current chord if it exists
      if (currentDisplayedChordRef.current) {
        return; // Don't change
      }
      // No current chord, use the most recent prediction
      majorityChord = recentPredictions[recentPredictions.length - 1].chord;
    }

    // Temporal persistence: Check minimum display time
    const now = time;
    const timeSinceLastChange = now - chordDisplayStartTimeRef.current;
    
    if (majorityChord === currentDisplayedChordRef.current) {
      // Same chord, no change needed
      return;
    }

    // Different chord - check if enough time has passed
    if (currentDisplayedChordRef.current !== null && timeSinceLastChange < MIN_DISPLAY_TIME) {
      // Not enough time has passed, keep current chord
      return;
    }

    // Update the displayed chord
    currentDisplayedChordRef.current = majorityChord;
    chordDisplayStartTimeRef.current = now;

    // Find the most recent prediction for this chord to get probabilities
    const mostRecentPrediction = recentPredictions
      .filter(p => p.chord === majorityChord)
      .slice(-1)[0] || recentPredictions[recentPredictions.length - 1];

    setCurrentChord(mostRecentPrediction);
  }, [predictions]);

  useEffect(() => {
    if (videoRef.current && videoBlob) {
      const url = URL.createObjectURL(videoBlob);
      videoRef.current.src = url;

      const video = videoRef.current;

      const handleLoadedMetadata = () => {
        setDuration(video.duration);
        // Reset smoothing state when video loads
        recentPredictionsRef.current = [];
        currentDisplayedChordRef.current = null;
        chordDisplayStartTimeRef.current = 0;
      };

      const handleTimeUpdate = () => {
        const newTime = video.currentTime;
        setCurrentTime(newTime);
        
        // Reset smoothing state if video looped back to start
        if (newTime < chordDisplayStartTimeRef.current - 1.0) {
          // Video jumped backwards (likely looped or seeked)
          recentPredictionsRef.current = [];
          currentDisplayedChordRef.current = null;
          chordDisplayStartTimeRef.current = newTime;
        }
        
        updateChord(newTime);
      };

      const handlePlay = () => {
        setIsPlaying(true);
        // Reset smoothing state when starting playback
        const currentTime = video.currentTime;
        recentPredictionsRef.current = [];
        currentDisplayedChordRef.current = null;
        chordDisplayStartTimeRef.current = currentTime;
      };
      
      const handlePause = () => setIsPlaying(false);
      
      const handleEnded = () => {
        // Reset when video ends (before it loops)
        recentPredictionsRef.current = [];
        currentDisplayedChordRef.current = null;
        chordDisplayStartTimeRef.current = 0;
        setCurrentChord(null);
      };

      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);
      video.addEventListener('ended', handleEnded);

      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('timeupdate', handleTimeUpdate);
        video.removeEventListener('play', handlePlay);
        video.removeEventListener('pause', handlePause);
        video.removeEventListener('ended', handleEnded);
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
      
      // Reset smoothing state on seek
      recentPredictionsRef.current = [];
      currentDisplayedChordRef.current = null;
      chordDisplayStartTimeRef.current = newTime;
      
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


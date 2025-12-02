import React from 'react';
import './ChordOverlay.css';

function ChordOverlay({ chord }) {
  if (!chord) {
    return null;
  }

  const isNoHand = chord.chord === 'No hand detected';

  return (
    <div className={`chord-overlay ${isNoHand ? 'no-hand' : ''}`}>
      <div className="chord-name">{chord.chord}</div>
      {!isNoHand && (
        <div className="chord-confidence">
          {Math.round(chord.confidence * 100)}% confident
        </div>
      )}
    </div>
  );
}

export default ChordOverlay;


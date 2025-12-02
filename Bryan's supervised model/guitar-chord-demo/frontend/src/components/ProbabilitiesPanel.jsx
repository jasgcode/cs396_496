import React from 'react';
import './ProbabilitiesPanel.css';

const CHORDS = ['A', 'Am', 'B', 'Bm', 'C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm'];

function ProbabilitiesPanel({ currentPrediction, accuracy }) {
  if (!currentPrediction || !currentPrediction.probabilities) {
    return (
      <div className="probabilities-panel">
        <div className="probabilities-header">
          <h3>Probabilities (per 2 seconds)</h3>
        </div>
        <div className="probabilities-content">
          <p className="no-data">No prediction data available</p>
        </div>
      </div>
    );
  }

  const probs = currentPrediction.probabilities;
  
  // Sort chords by probability (descending)
  const sortedChords = CHORDS.map(chord => ({
    chord,
    prob: probs[chord] || 0
  })).sort((a, b) => b.prob - a.prob);

  return (
    <div className="probabilities-panel">
      <div className="probabilities-header">
        <h3>Probabilities @ {typeof currentPrediction.timestamp === 'number' ? currentPrediction.timestamp.toFixed(1) : currentPrediction.timestamp}s</h3>
        {accuracy && (
          <div className="accuracy-display">
            <span className="accuracy-label">Accuracy:</span>
            <span className="accuracy-value">{accuracy.correct}/{accuracy.total} ({accuracy.percentage.toFixed(1)}%)</span>
          </div>
        )}
      </div>
      <div className="probabilities-content">
        <div className="probabilities-list">
          {sortedChords.map(({ chord, prob }) => {
            const isPredicted = currentPrediction.chord === chord;
            const percentage = (prob * 100).toFixed(1);
            const barWidth = (prob * 100).toFixed(1);
            
            return (
              <div key={chord} className={`probability-item ${isPredicted ? 'predicted' : ''}`}>
                <div className="probability-label">
                  <span className="chord-name">{chord}</span>
                  {isPredicted && <span className="predicted-badge">✓</span>}
                </div>
                <div className="probability-bar-container">
                  <div 
                    className="probability-bar" 
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
                <div className="probability-value">{percentage}%</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default ProbabilitiesPanel;


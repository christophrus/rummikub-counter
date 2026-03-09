import TileCard from './TileCard';
import './ResultDisplay.css';

function ResultDisplay({ result, uploadedImage, onReset }) {
  const { tiles, total_score, tile_count, processing_time_ms } = result;

  // Steine nach Farbe gruppieren
  const colorGroups = {};
  tiles.forEach((tile) => {
    const color = tile.is_joker ? 'joker' : (tile.color || 'unbekannt');
    if (!colorGroups[color]) colorGroups[color] = [];
    colorGroups[color].push(tile);
  });

  const recognizedTiles = tiles.filter((t) => t.number !== null || t.is_joker);
  const unrecognizedTiles = tiles.filter((t) => t.number === null && !t.is_joker);

  return (
    <div className="result-section">
      {/* Score-Anzeige */}
      <div className="score-card">
        <div className="score-main">
          <span className="score-label">Gesamtpunktzahl</span>
          <span className="score-value">{total_score}</span>
        </div>
        <div className="score-details">
          <div className="score-detail">
            <span className="detail-value">{tile_count}</span>
            <span className="detail-label">Steine erkannt</span>
          </div>
          <div className="score-detail">
            <span className="detail-value">{recognizedTiles.length}</span>
            <span className="detail-label">Zahlen gelesen</span>
          </div>
          <div className="score-detail">
            <span className="detail-value">{(processing_time_ms / 1000).toFixed(1)}s</span>
            <span className="detail-label">Verarbeitungszeit</span>
          </div>
        </div>
      </div>

      {/* Erkannte Steine */}
      {recognizedTiles.length > 0 && (
        <div className="tiles-section">
          <h2>Erkannte Steine</h2>
          <div className="tiles-grid">
            {recognizedTiles.map((tile, index) => (
              <TileCard key={index} tile={tile} />
            ))}
          </div>
        </div>
      )}

      {/* Nicht erkannte Steine */}
      {unrecognizedTiles.length > 0 && (
        <div className="tiles-section">
          <h2 className="section-warning">
            ⚠️ Nicht erkannte Steine ({unrecognizedTiles.length})
          </h2>
          <p className="section-hint">
            Diese Steine wurden gefunden, aber die Zahl konnte nicht gelesen werden.
          </p>
          <div className="tiles-grid">
            {unrecognizedTiles.map((tile, index) => (
              <TileCard key={`unknown-${index}`} tile={tile} />
            ))}
          </div>
        </div>
      )}

      {/* Farb-Zusammenfassung */}
      {Object.keys(colorGroups).length > 0 && (
        <div className="color-summary">
          <h2>Nach Farbe</h2>
          <div className="color-bars">
            {Object.entries(colorGroups).map(([color, groupTiles]) => {
              const groupScore = groupTiles.reduce((sum, t) => sum + (t.number || (t.is_joker ? 30 : 0)), 0);
              return (
                <div key={color} className={`color-bar color-${color}`}>
                  <span className="color-name">{color}</span>
                  <span className="color-count">{groupTiles.length} Steine</span>
                  <span className="color-score">{groupScore} Punkte</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Reset-Button */}
      <div className="reset-section">
        <button className="reset-button" onClick={onReset}>
          🔄 Neues Bild analysieren
        </button>
      </div>
    </div>
  );
}

export default ResultDisplay;

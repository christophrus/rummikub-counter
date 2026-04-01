import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { analyzeImageDebug } from '../services/api';
import TileCard from './TileCard';
import './ResultDisplay.css';

function ResultDisplay({ result, uploadedImage, uploadedFile, onReset }) {
  const [debugImage, setDebugImage] = useState(null);
  const [debugLoading, setDebugLoading] = useState(false);
  const { t } = useTranslation();
  const { tiles, total_score, tile_count, processing_time_ms } = result;

  const handleDebug = async () => {
    if (!uploadedFile) return;
    if (debugImage) {
      setDebugImage(null);
      return;
    }
    setDebugLoading(true);
    try {
      const data = await analyzeImageDebug(uploadedFile);
      setDebugImage(data.debug_image);
    } catch {
      setDebugImage(null);
    } finally {
      setDebugLoading(false);
    }
  };

  const recognizedTiles = tiles.filter((t) => t.number !== null || t.is_joker);
  const unrecognizedTiles = tiles.filter((t) => t.number === null && !t.is_joker);

  // Group recognized tiles by class (number or joker)
  const groupedTiles = recognizedTiles.reduce((groups, tile) => {
    const key = tile.is_joker ? 'joker' : String(tile.number);
    if (!groups[key]) {
      groups[key] = { ...tile, count: 1 };
    } else {
      groups[key].count += 1;
    }
    return groups;
  }, {});
  const groupedTileList = Object.values(groupedTiles).sort((a, b) => {
    if (a.is_joker) return 1;
    if (b.is_joker) return -1;
    return a.number - b.number;
  });

  return (
    <div className="result-section">
      {/* Score-Anzeige */}
      <div className="score-card">
        <div className="score-main">
          <span className="score-label">{t('result.totalScore')}</span>
          <span className="score-value">{total_score}</span>
        </div>
        <div className="score-details">
          <div className="score-detail">
            <span className="detail-value">{tile_count}</span>
            <span className="detail-label">{t('result.tilesDetected')}</span>
          </div>
          <div className="score-detail">
            <span className="detail-value">{recognizedTiles.length}</span>
            <span className="detail-label">{t('result.numbersRead')}</span>
          </div>
          <div className="score-detail">
            <span className="detail-value">{(processing_time_ms / 1000).toFixed(1)}s</span>
            <span className="detail-label">{t('result.processingTime')}</span>
          </div>
        </div>
      </div>

      {/* Erkannte Steine */}
      {recognizedTiles.length > 0 && (
        <div className="tiles-section">
          <h2>{t('result.recognizedTiles')}</h2>
          <div className="tiles-grid">
            {groupedTileList.map((tile, index) => (
              <TileCard key={index} tile={tile} />
            ))}
          </div>
        </div>
      )}

      {/* Nicht erkannte Steine */}
      {unrecognizedTiles.length > 0 && (
        <div className="tiles-section">
          <h2 className="section-warning">
            ⚠️ {t('result.unrecognizedTiles')} ({unrecognizedTiles.length})
          </h2>
          <p className="section-hint">{t('result.unrecognizedHint')}</p>
          <div className="tiles-grid">
            {unrecognizedTiles.map((tile, index) => (
              <TileCard key={`unknown-${index}`} tile={tile} />
            ))}
          </div>
        </div>
      )}

      {/* Debug-Bild */}
      {uploadedFile && (
        <div className="debug-section">
          <button
            className={`debug-button ${debugImage ? 'active' : ''}`}
            onClick={handleDebug}
            disabled={debugLoading}
          >
            {debugLoading
              ? t('result.debugLoading')
              : debugImage
                ? t('result.debugHide')
                : t('result.debugShow')}
          </button>
          {debugImage && (
            <div className="debug-image-container">
              <img src={debugImage} alt="Debug" className="debug-image" />
            </div>
          )}
        </div>
      )}

      {/* Reset-Button */}
      <div className="reset-section">
        <button className="reset-button" onClick={onReset}>
          {t('result.resetButton')}
        </button>
      </div>
    </div>
  );
}

export default ResultDisplay;

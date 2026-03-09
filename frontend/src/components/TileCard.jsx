import './TileCard.css';

const COLOR_EMOJIS = {
  rot: '🔴',
  blau: '🔵',
  schwarz: '⚫',
  orange: '🟠',
};

function TileCard({ tile }) {
  const { number, color, confidence, is_joker } = tile;

  if (is_joker) {
    return (
      <div className="tile-card tile-joker">
        <span className="tile-number">🃏</span>
        <span className="tile-label">Joker</span>
        <span className="tile-points">30 Pkt</span>
      </div>
    );
  }

  const colorClass = color ? `tile-${color}` : '';
  const emoji = COLOR_EMOJIS[color] || '❓';
  const confidencePercent = Math.round((confidence || 0) * 100);

  return (
    <div className={`tile-card ${colorClass}`}>
      <span className="tile-color-dot">{emoji}</span>
      <span className="tile-number">
        {number !== null ? number : '?'}
      </span>
      {number !== null && (
        <span className="tile-confidence">{confidencePercent}%</span>
      )}
    </div>
  );
}

export default TileCard;

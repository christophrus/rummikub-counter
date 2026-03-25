import './TileCard.css';

function TileCard({ tile }) {
  const { number, confidence, is_joker, count } = tile;

  if (is_joker) {
    return (
      <div className={`tile-wrapper${count > 1 ? ' has-count' : ''}`}>
        {count > 1 && <span className="tile-count">{count}×</span>}
        <div className="tile-card tile-joker">
          <span className="tile-number">🃏</span>
          <span className="tile-label">Joker</span>
          <span className="tile-points">{count > 1 ? `${count * 20} Pkt` : '20 Pkt'}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`tile-wrapper${count > 1 ? ' has-count' : ''}`}>
      {count > 1 && <span className="tile-count">{count}×</span>}
      <div className="tile-card">
        <span className="tile-number">
          {number !== null ? number : '?'}
        </span>
      </div>
    </div>
  );
}

export default TileCard;

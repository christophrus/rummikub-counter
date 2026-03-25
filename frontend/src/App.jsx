import { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleAnalysisStart = () => {
    setIsLoading(true);
    setError(null);
    setResult(null);
  };

  const handleAnalysisComplete = (data) => {
    setResult(data);
    setIsLoading(false);
  };

  const handleAnalysisError = (err) => {
    setError(err.message || 'Ein Fehler ist aufgetreten.');
    setIsLoading(false);
  };

  const handleImageSelected = (imageUrl) => {
    setUploadedImage(imageUrl);
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setUploadedImage(null);
    setIsLoading(false);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">🎲</div>
        <h1>Rummikub Stein-Erkennung</h1>
        <p className="subtitle">
          Lade ein Foto deiner Rummikub-Steine hoch und lass die KI die Punkte zählen
        </p>
      </header>

      <main className="app-main">
        <ImageUpload
          onAnalysisStart={handleAnalysisStart}
          onAnalysisComplete={handleAnalysisComplete}
          onAnalysisError={handleAnalysisError}
          onImageSelected={handleImageSelected}
          isLoading={isLoading}
        />

        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="error-close">✕</button>
          </div>
        )}

        {isLoading && (
          <div className="loading-section">
            <div className="spinner" />
            <p>KI analysiert dein Bild...</p>
            <p className="loading-detail">
              🧠 Deep Learning Modell (CNN + LSTM) erkennt die Steine
            </p>
          </div>
        )}

        {result && (
          <ResultDisplay
            result={result}
            uploadedImage={uploadedImage}
            onReset={handleReset}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by EasyOCR (Deep Learning) • OpenCV • FastAPI • React</p>
      </footer>
    </div>
  );
}

export default App;

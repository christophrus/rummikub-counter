import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import LanguageSwitcher from './components/LanguageSwitcher';
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const { t } = useTranslation();

  // Firefox auf Android kann die Seite aus dem Speicher werfen,
  // wenn die Kamera ein sehr großes Foto macht. Wenn die Seite
  // aus dem bfcache wiederhergestellt wird, ist der State verloren
  // und die UI zeigt einen veralteten Ladezustand. Wir erkennen
  // bfcache-Restores und setzen den Lade-Indikator zurück.
  useEffect(() => {
    const handlePageShow = (event) => {
      if (event.persisted) {
        // Seite wurde aus bfcache wiederhergestellt
        // (z.B. nachdem Firefox die Seite wegen Speicherdruck
        //  durch ein Kamera-Foto aus dem RAM geworfen hat)
        setIsLoading(false);
      }
    };

    window.addEventListener('pageshow', handlePageShow);
    return () => window.removeEventListener('pageshow', handlePageShow);
  }, []);

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
    setUploadedFile(null);
    setIsLoading(false);
  };

  return (
    <div className="app">
      <header className="app-header">
        <LanguageSwitcher />
        <div className="logo">🎲</div>
        <h1>{t('app.title')}</h1>
        <p className="subtitle">{t('app.subtitle')}</p>
      </header>

      <main className="app-main">
        <ImageUpload
          onAnalysisStart={handleAnalysisStart}
          onAnalysisComplete={handleAnalysisComplete}
          onAnalysisError={handleAnalysisError}
          onImageSelected={handleImageSelected}
          onFileSelected={setUploadedFile}
          isLoading={isLoading}
        />

        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <span>{error || t('error.generic')}</span>
            <button onClick={() => setError(null)} className="error-close">✕</button>
          </div>
        )}

        {isLoading && (
          <div className="loading-section">
            <div className="spinner" />
            <p>{t('loading.text')}</p>
            <p className="loading-detail">{t('loading.detail')}</p>
          </div>
        )}

        {result && (
          <ResultDisplay
            result={result}
            uploadedImage={uploadedImage}
            uploadedFile={uploadedFile}
            onReset={handleReset}
          />
        )}
      </main>

      <section className="seo-section">
        <h2>{t('seo.heading')}</h2>
        <p>{t('seo.text')}</p>
        <h3>{t('seo.howTitle')}</h3>
        <ol className="seo-steps">
          <li>{t('seo.step1')}</li>
          <li>{t('seo.step2')}</li>
          <li>{t('seo.step3')}</li>
          <li>{t('seo.step4')}</li>
        </ol>
      </section>

      <footer className="app-footer">
        <p>{t('app.footer')}</p>
        <p>
          <a href="https://github.com/ChristophRus/rummikub-counter" target="_blank" rel="noopener noreferrer" className="footer-link">
            GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;

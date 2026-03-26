import { useState } from 'react';
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
  const { t } = useTranslation();

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
      </footer>
    </div>
  );
}

export default App;

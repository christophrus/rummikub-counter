import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { analyzeImage } from '../services/api';
import './ImageUpload.css';

function ImageUpload({ onAnalysisStart, onAnalysisComplete, onAnalysisError, onImageSelected, onFileSelected, isLoading }) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const { t } = useTranslation();

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      onAnalysisError({ message: t('upload.invalidFile') });
      return;
    }

    // Vorschau erstellen
    const imageUrl = URL.createObjectURL(file);
    setPreview(imageUrl);
    onImageSelected(imageUrl);
    onFileSelected(file);

    // Analyse starten
    onAnalysisStart();

    try {
      const result = await analyzeImage(file);
      onAnalysisComplete(result);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        t('upload.analysisError');
      onAnalysisError({ message });
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
  };

  return (
    <div className="upload-section">
      <div
        className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${isLoading ? 'disabled' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt={t('upload.preview')} className="preview-image" />
            {!isLoading && <p className="preview-hint">{t('upload.previewHint')}</p>}
          </div>
        ) : (
          <div className="drop-content">
            <div className="drop-icon">📷</div>
            <p className="drop-text" dangerouslySetInnerHTML={{ __html: t('upload.dropText') }} />
            <p className="drop-hint">{t('upload.dropHint')}</p>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        hidden
        disabled={isLoading}
      />
    </div>
  );
}

export default ImageUpload;

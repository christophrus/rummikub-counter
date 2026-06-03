import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { analyzeImage } from '../services/api';
import './ImageUpload.css';

// Maximale Bildgröße: längere Seite auf diesen Wert begrenzen
// Reduziert Speicherdruck auf mobilen Geräten massiv und verhindert
// Page-Reloads in Firefox, wenn Kamera-Fotos zu groß sind.
const MAX_IMAGE_PX = 1920;

/**
 * Verkleinert ein Bild client-seitig via Canvas auf max MAX_IMAGE_PX.
 * Gibt ein Blob (image/jpeg, Qualität 0.85) zurück – deutlich kleiner
 * als das Kamera-Original, was Speicher und Bandbreite spart.
 */
function resizeImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      const { naturalWidth: w, naturalHeight: h } = img;

      // Nur verkleinern, wenn nötig
      if (Math.max(w, h) <= MAX_IMAGE_PX) {
        resolve(file);
        return;
      }

      const scale = MAX_IMAGE_PX / Math.max(w, h);
      const newW = Math.round(w * scale);
      const newH = Math.round(h * scale);

      const canvas = document.createElement('canvas');
      canvas.width = newW;
      canvas.height = newH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, newW, newH);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(new File([blob], file.name || 'camera.jpg', { type: 'image/jpeg' }));
          } else {
            // Fallback: Original verwenden
            resolve(file);
          }
        },
        'image/jpeg',
        0.85
      );
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(file); // Fallback: Original
    };

    img.src = url;
  });
}

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

    // Bild client-seitig verkleinern, um Speicherdruck zu reduzieren
    // (besonders wichtig für mobile Firefox mit Kamera-Fotos)
    const resizedFile = await resizeImage(file);

    // Vorschau erstellen
    const imageUrl = URL.createObjectURL(resizedFile);
    setPreview(imageUrl);
    onImageSelected(imageUrl);
    onFileSelected(resizedFile);

    // Analyse starten
    onAnalysisStart();

    try {
      const result = await analyzeImage(resizedFile);
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
        capture="environment"
        onChange={handleChange}
        hidden
        disabled={isLoading}
      />
    </div>
  );
}

export default ImageUpload;

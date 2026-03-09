import { useRef, useState } from 'react';
import { analyzeImage } from '../services/api';
import './ImageUpload.css';

function ImageUpload({ onAnalysisStart, onAnalysisComplete, onAnalysisError, onImageSelected, isLoading }) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      onAnalysisError({ message: 'Bitte wähle eine Bilddatei aus.' });
      return;
    }

    // Vorschau erstellen
    const imageUrl = URL.createObjectURL(file);
    setPreview(imageUrl);
    onImageSelected(imageUrl);

    // Analyse starten
    onAnalysisStart();

    try {
      const result = await analyzeImage(file);
      onAnalysisComplete(result);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Fehler bei der Analyse.';
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
            <img src={preview} alt="Vorschau" className="preview-image" />
            {!isLoading && <p className="preview-hint">Klicke oder ziehe ein neues Bild hierher</p>}
          </div>
        ) : (
          <div className="drop-content">
            <div className="drop-icon">📷</div>
            <p className="drop-text">
              <strong>Bild hierher ziehen</strong> oder klicken zum Auswählen
            </p>
            <p className="drop-hint">JPG, PNG, WebP • Max. 10 MB</p>
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

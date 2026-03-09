import axios from 'axios';

const API_BASE = '/api';

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE}/analyze`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000, // 2 Minuten (OCR kann dauern)
  });

  return response.data;
}

export async function analyzeImageDebug(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE}/analyze-debug`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });

  return response.data;
}

export async function healthCheck() {
  const response = await axios.get('/health');
  return response.data;
}

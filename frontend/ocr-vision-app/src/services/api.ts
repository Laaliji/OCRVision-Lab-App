import axios from 'axios';
import { OCRResult } from '../types/types';

// Définir l'URL de base pour l'API
const API_URL = 'http://localhost:8080';

// Créer une instance axios avec des configurations communes
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const OCRService = {
  // Méthode pour traiter une image
  async processImage(imageFile: File): Promise<OCRResult> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      
      const response = await apiClient.post('/process_image', formData);
      return response.data;
    } catch (error) {
      console.error('Error processing image:', error);
      throw error;
    }
  },
  
  // Pour obtenir les informations sur l'application
  async getInfo(): Promise<{ version: string }> {
    try {
      const response = await apiClient.get('/info');
      return response.data;
    } catch (error) {
      console.error('Error fetching app info:', error);
      throw error;
    }
  }
}; 
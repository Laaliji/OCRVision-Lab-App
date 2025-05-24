import axios from 'axios';
import { OCRResult, TransformationResult } from '../types/types';

// Définir l'URL de base pour l'API
const API_URL = 'http://localhost:8080';

// Créer une instance axios avec des configurations communes
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export interface StepResult {
  success: boolean;
  step: number;
  message: string;
  original_image?: string;
  degraded_image?: string;
  preprocessed_image?: string;
  segmented_image?: string;
  degradation_info?: any;
  preprocessing_info?: any;
  segmentation_info?: any;
  recognition_results?: any;
  error?: string;
}

export const OCRService = {
  // Méthode pour traiter une étape spécifique
  async processStep(imageFile: File, step: number): Promise<StepResult> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('step', step.toString());
      
      const response = await apiClient.post('/process_step', formData);
      return response.data;
    } catch (error) {
      console.error('Error processing step:', error);
      throw error;
    }
  },

  // Méthode pour traiter une image (méthode complète)
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
  
  // Méthode pour traiter une image avec des transformations spécifiques
  async processImageWithTransformations(imageFile: File, transformations: string[]): Promise<OCRResult> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('transformations', JSON.stringify(transformations));
      
      const response = await apiClient.post('/process_with_transformations', formData);
      return response.data;
    } catch (error) {
      console.error('Error processing image with transformations:', error);
      throw error;
    }
  },
  
  // Méthode pour appliquer une technique de transformation spécifique avec des paramètres personnalisés
  async applyTransformation(
    imageFile: File, 
    techniqueId: string, 
    parameters: Record<string, any>
  ): Promise<TransformationResult> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('technique_id', techniqueId);
      formData.append('parameters', JSON.stringify(parameters));
      
      const response = await apiClient.post('/apply_transformation', formData);
      return response.data;
    } catch (error) {
      console.error('Error applying transformation:', error);
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
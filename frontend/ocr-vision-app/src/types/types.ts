export interface CharacterPrediction {
  character: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
}

export interface ModelResults {
  degraded_text: string;
  preprocessed_text: string;
  degraded_predictions: CharacterPrediction[];
  preprocessed_predictions: CharacterPrediction[];
  degraded_confidence: number;
  preprocessed_confidence: number;
  comparison: string;
}

export interface TesseractResults {
  degraded_text: string;
  preprocessed_text: string;
  comparison: string;
}

export interface OCRResult {
  success: boolean;
  error?: string;
  original_image: string;
  degraded_image: string;
  preprocessed_image: string;
  segmented_image?: string;
  word_boxes?: BoundingBox[];
  character_boxes?: BoundingBox[];
  transformation_info?: string[];
  
  // Model-based results
  model_results: ModelResults;
  
  // Traditional Tesseract results
  tesseract_results: TesseractResults;
  
  // Legacy fields for backward compatibility
  degraded_text: string;
  preprocessed_text: string;
  comparison?: string;
}

export interface ProcessStep {
  id: number;
  title: string;
  description: string;
  isActive: boolean;
  isCompleted: boolean;
}

export interface UploadResponse {
  success: boolean;
  error?: string;
  filePath?: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  text?: string;
  confidence?: number;
}

export interface ImageTransformation {
  name: string;
  description: string;
  id: string;
  applied: boolean;
}

export interface ModelPrediction {
  character: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface TransformationResult {
  transformedImage: string;
  description: string;
  parameters: Record<string, string | number>;
}

export interface ProcessingTechnique {
  id: string;
  name: string;
  description: string;
  type: 'degradation' | 'preprocessing';
  parameters: ProcessingParameter[];
}

export interface ProcessingParameter {
  id: string;
  name: string;
  description: string;
  type: 'range' | 'select' | 'checkbox';
  min?: number;
  max?: number;
  step?: number;
  default: number | string | boolean;
  options?: {value: string; label: string}[];
}

// Add html2canvas type definition
declare global {
  interface Window {
    html2canvas: (element: HTMLElement, options?: any) => Promise<HTMLCanvasElement>;
  }
}

export interface WordRecognitionResult {
  success: boolean;
  error?: string;
  original_image: string;
  preprocessed_image: string;
  visualization: string;
  recognition_mode: 'character' | 'text';
  
  // Word recognition results
  word_recognition: {
    original_word: string;
    preprocessed_word: string;
    best_word: string;
    original_confidence: number;
    preprocessed_confidence: number;
    comparison: string;
    character_details: CharacterPrediction[];
  };
  
  // Results from other methods for comparison
  comparison: {
    character_by_character: string;
    word_segmentation: string;
    tesseract: string;
  };
} 
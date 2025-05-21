export interface OCRResult {
  success: boolean;
  error?: string;
  original_image: string;
  degraded_image: string;
  preprocessed_image: string;
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
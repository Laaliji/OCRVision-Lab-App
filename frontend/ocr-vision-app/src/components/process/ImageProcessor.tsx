import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSpinner, faCheckCircle, faInfoCircle, faWrench } from '@fortawesome/free-solid-svg-icons';
import Button from '../common/Button';
import { ImageCharacteristics } from './ImageValidator';
import { useLanguage } from '../../contexts/LanguageContext';

interface ImageProcessorProps {
  imageFile: File | null;
  characteristics: ImageCharacteristics | null;
  onProcessingComplete: (processedFile: File | null) => void;
}

const ImageProcessor: React.FC<ImageProcessorProps> = ({ 
  imageFile, 
  characteristics, 
  onProcessingComplete 
}) => {
  const { t } = useLanguage();
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);
  const [processedFile, setProcessedFile] = useState<File | null>(null);
  const [processedCharacteristics, setProcessedCharacteristics] = useState<ImageCharacteristics | null>(null);
  
  // Set original image URL when file changes
  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setOriginalImageUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [imageFile]);
  
  // Auto-process when image doesn't meet requirements
  useEffect(() => {
    if (imageFile && characteristics && !meetsRequirements(characteristics)) {
      processImage();
    }
  }, [imageFile, characteristics]);
  
  const meetsRequirements = (chars: ImageCharacteristics): boolean => {
    return chars.isGrayscale && chars.hasRequiredSize;
  };
  
  const processImage = async () => {
    if (!imageFile || !characteristics) return;
    
    setIsProcessing(true);
    
    try {
      // Create an image element to process
      const img = new Image();
      const imageUrl = URL.createObjectURL(imageFile);
      
      img.onload = async () => {
        // Create canvas to manipulate the image
        const canvas = document.createElement('canvas');
        
        // Calculate new dimensions that meet the requirements
        const targetWidth = Math.min(Math.max(characteristics.width, 20), 64);
        const targetHeight = Math.min(Math.max(characteristics.height, 20), 64);
        
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          console.error("Could not get canvas context");
          setIsProcessing(false);
          return;
        }
        
        // Draw image at new dimensions
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        
        // Convert to grayscale if needed
        if (!characteristics.isGrayscale) {
          const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
          const data = imageData.data;
          
          for (let i = 0; i < data.length; i += 4) {
            // Standard grayscale conversion formula
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            data[i] = data[i + 1] = data[i + 2] = gray;
          }
          
          ctx.putImageData(imageData, 0, 0);
        }
        
        // Convert canvas to blob
        canvas.toBlob(async (blob) => {
          if (!blob) {
            console.error("Failed to create blob from canvas");
            setIsProcessing(false);
            return;
          }
          
          // Create a new file from the blob
          const processedFile = new File([blob], 'processed_' + imageFile.name, {
            type: 'image/png',
            lastModified: new Date().getTime()
          });
          
          // Set the processed file
          setProcessedFile(processedFile);
          
          // Create URL for display
          const processedUrl = URL.createObjectURL(blob);
          setProcessedImageUrl(processedUrl);
          
          // Update processed characteristics
          setProcessedCharacteristics({
            width: targetWidth,
            height: targetHeight,
            aspectRatio: targetWidth / targetHeight,
            bitDepth: 8,
            isGrayscale: true,
            hasRequiredSize: true
          });
          
          // Notify parent component
          onProcessingComplete(processedFile);
          
          // Clean up URL
          URL.revokeObjectURL(imageUrl);
          
          setIsProcessing(false);
        }, 'image/png');
      };
      
      img.onerror = () => {
        console.error("Error loading image for processing");
        setIsProcessing(false);
      };
      
      // Start loading the image
      img.src = imageUrl;
      
    } catch (error) {
      console.error("Error processing image:", error);
      setIsProcessing(false);
    }
  };
  
  const handleUseOriginal = () => {
    onProcessingComplete(imageFile);
  };
  
  const handleUseProcessed = () => {
    onProcessingComplete(processedFile);
  };
  
  if (!imageFile || !characteristics) {
    return null;
  }
  
  // If image already meets requirements and no processing done
  if (meetsRequirements(characteristics) && !processedImageUrl) {
    return (
      <div className="mt-4 bg-green-50 border-l-4 border-green-500 p-3 rounded">
        <div className="flex items-center text-green-700">
          <FontAwesomeIcon icon={faCheckCircle} className="mr-2" />
          <p>{t('validation.meetsRequirements')}</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="mt-6">
      {isProcessing ? (
        <div className="flex items-center justify-center p-8">
          <FontAwesomeIcon icon={faSpinner} className="text-indigo-500 mr-3 animate-spin text-xl" />
          <p className="text-gray-700">{t('validation.analyzing')}</p>
        </div>
      ) : (
        <>
          {processedImageUrl && (
            <>
              <h3 className="text-lg font-medium text-gray-800 mb-3 flex items-center">
                <FontAwesomeIcon icon={faWrench} className="mr-2 text-indigo-500" />
                {t('processing.title')}
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium text-gray-700 mb-2">{t('processing.original')}</h4>
                  {originalImageUrl && (
                    <div className="relative">
                      <img 
                        src={originalImageUrl} 
                        alt="Original" 
                        className="w-full h-auto rounded" 
                      />
                      <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                        {t('processing.notCompatible')}
                      </div>
                    </div>
                  )}
                  <div className="mt-2 text-sm text-gray-500">
                    <ul>
                      <li>
                        <span className="font-medium">{t('processing.dimensions')} </span>
                        {characteristics.width}x{characteristics.height}
                      </li>
                      <li>
                        <span className="font-medium">{t('processing.grayscale')} </span>
                        {characteristics.isGrayscale ? "Yes" : "No"}
                      </li>
                    </ul>
                  </div>
                </div>
                
                <div className="border rounded-lg p-4 border-green-300 bg-green-50">
                  <h4 className="font-medium text-gray-700 mb-2">{t('processing.processed')}</h4>
                  {processedImageUrl && (
                    <div className="relative">
                      <img 
                        src={processedImageUrl} 
                        alt="Processed" 
                        className="w-full h-auto rounded" 
                      />
                      <div className="absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                        {t('processing.compatible')}
                      </div>
                    </div>
                  )}
                  {processedCharacteristics && (
                    <div className="mt-2 text-sm text-gray-500">
                      <ul>
                        <li>
                          <span className="font-medium">{t('processing.dimensions')} </span>
                          {processedCharacteristics.width}x{processedCharacteristics.height}
                        </li>
                        <li>
                          <span className="font-medium">{t('processing.grayscale')} </span>
                          Yes
                        </li>
                      </ul>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="mt-4 flex items-center justify-center space-x-4">
                <Button
                  variant="outline"
                  onClick={handleUseOriginal}
                  disabled={!meetsRequirements(characteristics)}
                >
                  {t('button.useOriginal')}
                </Button>
                <Button
                  variant="primary"
                  onClick={handleUseProcessed}
                >
                  {t('button.useProcessed')}
                </Button>
              </div>
              
              <div className="mt-4 bg-blue-50 border-l-4 border-blue-500 p-3 text-sm text-blue-700">
                <div className="flex">
                  <FontAwesomeIcon icon={faInfoCircle} className="mt-0.5 mr-2" />
                  <p>
                    <strong>{t('processing.note')}</strong> {t('processing.noteDesc')}
                  </p>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
};

export default ImageProcessor; 
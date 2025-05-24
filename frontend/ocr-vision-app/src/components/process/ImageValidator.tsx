import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheckCircle, faExclamationTriangle, faInfoCircle, faSpinner } from '@fortawesome/free-solid-svg-icons';
import { useLanguage } from '../../contexts/LanguageContext';

interface ImageValidatorProps {
  imageFile: File | null;
  onValidationComplete: (isValid: boolean, message: string, characteristics: ImageCharacteristics | null) => void;
}

export interface ImageCharacteristics {
  width: number;
  height: number;
  aspectRatio: number;
  bitDepth: number;
  isGrayscale: boolean;
  hasRequiredSize: boolean;
}

const REQUIRED_CHARACTERISTICS = {
  minWidth: 20,
  maxWidth: 64,
  minHeight: 20,
  maxHeight: 64,
  bitDepth: 8,
  mustBeGrayscale: true
};

const ImageValidator: React.FC<ImageValidatorProps> = ({ imageFile, onValidationComplete }) => {
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean;
    message: string;
    characteristics: ImageCharacteristics | null;
  } | null>(null);
  const { t } = useLanguage();

  useEffect(() => {
    if (imageFile) {
      validateImage(imageFile);
    } else {
      setValidationResult(null);
    }
  }, [imageFile]);

  const validateImage = async (file: File) => {
    setIsValidating(true);
    
    try {
      // Create an image element to analyze dimensions
      const img = new Image();
      const imageUrl = URL.createObjectURL(file);
      
      img.onload = () => {
        // Get image dimensions
        const width = img.width;
        const height = img.height;
        const aspectRatio = width / height;
        
        // Create canvas to analyze pixel data (for grayscale check)
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          onValidationComplete(false, "Error analyzing image data", null);
          setIsValidating(false);
          return;
        }
        
        // Draw image to canvas
        ctx.drawImage(img, 0, 0);
        
        // Check if image is grayscale by sampling pixels
        const isGrayscale = checkIfGrayscale(ctx, width, height);
        
        // Assume 8-bit depth for standard images (can't reliably detect from canvas)
        const bitDepth = 8;
        
        // Check if size matches requirements
        const hasRequiredSize = 
          width >= REQUIRED_CHARACTERISTICS.minWidth && 
          width <= REQUIRED_CHARACTERISTICS.maxWidth &&
          height >= REQUIRED_CHARACTERISTICS.minHeight && 
          height <= REQUIRED_CHARACTERISTICS.maxHeight;
        
        // Compile characteristics
        const characteristics: ImageCharacteristics = {
          width,
          height,
          aspectRatio,
          bitDepth,
          isGrayscale,
          hasRequiredSize
        };
        
        // Determine if image is valid for model
        const isValid = (
          (!REQUIRED_CHARACTERISTICS.mustBeGrayscale || isGrayscale) &&
          hasRequiredSize
        );
        
        let message = "";
        if (isValid) {
          message = "Image meets dataset requirements";
        } else {
          const issues = [];
          if (REQUIRED_CHARACTERISTICS.mustBeGrayscale && !isGrayscale) {
            issues.push(t('validation.mustBeGrayscale'));
          }
          if (!hasRequiredSize) {
            issues.push(t('validation.sizeRequirement', {
              min: REQUIRED_CHARACTERISTICS.minWidth,
              max: REQUIRED_CHARACTERISTICS.maxWidth
            }));
          }
          message = `${t('validation.doesNotMeet')} ${issues.join(', ')}`;
        }
        
        setValidationResult({
          isValid,
          message,
          characteristics
        });
        
        // Clean up URL
        URL.revokeObjectURL(imageUrl);
        
        // Report validation result
        onValidationComplete(isValid, message, characteristics);
        setIsValidating(false);
      };
      
      img.onerror = () => {
        onValidationComplete(false, "Error loading image for validation", null);
        setIsValidating(false);
      };
      
      // Start loading the image
      img.src = imageUrl;
      
    } catch (error) {
      console.error("Error validating image:", error);
      onValidationComplete(false, "Error validating image", null);
      setIsValidating(false);
    }
  };
  
  const checkIfGrayscale = (ctx: CanvasRenderingContext2D, width: number, height: number): boolean => {
    // Sample pixels to check if image is grayscale
    // We don't need to check every pixel, just a representative sample
    const sampleSize = Math.min(100, width * height);
    const sampleStep = Math.max(1, Math.floor((width * height) / sampleSize));
    
    for (let i = 0; i < width * height; i += sampleStep) {
      const x = i % width;
      const y = Math.floor(i / width);
      const pixelData = ctx.getImageData(x, y, 1, 1).data;
      
      // For a grayscale image, R, G, and B values should be equal
      if (!(pixelData[0] === pixelData[1] && pixelData[1] === pixelData[2])) {
        return false;
      }
    }
    
    return true;
  };
  
  if (!imageFile || !validationResult) {
    return null;
  }
  
  const { isValid, message, characteristics } = validationResult;
  
  return (
    <div className="mt-4">
      {isValidating ? (
        <div className="flex items-center text-gray-600">
          <FontAwesomeIcon icon={faSpinner} className="mr-2 animate-spin" />
          <p>Validating image characteristics...</p>
        </div>
      ) : (
        <>
          <div className={`flex items-center ${isValid ? 'text-green-600' : 'text-red-600'}`}>
            <FontAwesomeIcon 
              icon={isValid ? faCheckCircle : faExclamationTriangle} 
              className="mr-2" 
            />
            <p className="font-medium">{message}</p>
          </div>
          
          {characteristics && (
            <div className="mt-2 bg-gray-50 p-3 rounded border text-sm">
              <h4 className="font-medium text-gray-700 mb-1 flex items-center">
                <FontAwesomeIcon icon={faInfoCircle} className="mr-1 text-gray-500" />
                {t('validation.characteristics')}
              </h4>
              <ul className="grid grid-cols-2 gap-1">
                <li>
                  <span className="text-gray-500">{t('validation.dimensions')}</span>{" "}
                  <span className={!characteristics.hasRequiredSize ? 'text-red-600 font-medium' : ''}>
                    {characteristics.width} x {characteristics.height}
                  </span>
                </li>
                <li>
                  <span className="text-gray-500">{t('validation.aspectRatio')}</span>{" "}
                  {characteristics.aspectRatio.toFixed(2)}
                </li>
                <li>
                  <span className="text-gray-500">{t('validation.bitDepth')}</span>{" "}
                  {characteristics.bitDepth}
                </li>
                <li>
                  <span className="text-gray-500">{t('validation.grayscale')}</span>{" "}
                  <span className={REQUIRED_CHARACTERISTICS.mustBeGrayscale && !characteristics.isGrayscale ? 'text-red-600 font-medium' : ''}>
                    {characteristics.isGrayscale ? t('validation.yes') : t('validation.no')}
                  </span>
                </li>
              </ul>
            </div>
          )}
          
          {!isValid && (
            <div className="mt-3 bg-yellow-50 border-l-4 border-yellow-400 p-3 text-sm text-yellow-700">
              <strong>{t('validation.note.title')}</strong> {t('validation.note.text')}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ImageValidator; 
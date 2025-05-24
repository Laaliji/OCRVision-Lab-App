import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faArrowLeft, faChartLine, faCheckCircle, faExclamationTriangle, faSync, faInfo, faCheck, faImage, faInfoCircle, faPlay, faPause, faStepForward, faUndo, faArrowDown } from '@fortawesome/free-solid-svg-icons';
import { ClipLoader } from 'react-spinners';
import ProcessStepper from './ProcessStepper';
import ImageUploader from './ImageUploader';
import Button from '../common/Button';
import { OCRService } from '../../services/api';
import { ProcessStep, OCRResult, ImageTransformation } from '../../types/types';
import ImageValidator, { ImageCharacteristics } from './ImageValidator';
import ImageProcessor from './ImageProcessor';
import { useLanguage } from '../../contexts/LanguageContext';
import ResultActions from './ResultActions';

const Process: React.FC = () => {
  const { t } = useLanguage();
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [progress, setProgress] = useState<number>(0);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<File | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [appliedTransformations, setAppliedTransformations] = useState<string[]>([
    'gaussian_noise', 'salt_pepper', 'brightness', 'blur'
  ]);
  
  // New state for image validation and processing
  const [imageCharacteristics, setImageCharacteristics] = useState<ImageCharacteristics | null>(null);
  const [isImageValid, setIsImageValid] = useState<boolean>(false);
  const [validationMessage, setValidationMessage] = useState<string>("");
  const [activeImage, setActiveImage] = useState<File | null>(null);
  
  // Required transformations
  const requiredTransformations = [
    { name: 'Gaussian Noise', description: 'Random noise is added to the image', id: 'gaussian_noise' },
    { name: 'Salt & Pepper', description: 'Salt and pepper noise adds white and black pixels', id: 'salt_pepper' },
    { name: 'Brightness', description: 'Brightness is adjusted to simulate different lighting', id: 'brightness' },
    { name: 'Blur', description: 'Gaussian blur simulates out of focus or low resolution', id: 'blur' }
  ];
  
  // New state for segmentation
  const [segmentationStep, setSegmentationStep] = useState<number>(0);
  
  // Set original image URL when a file is selected
  useEffect(() => {
    if (selectedImage) {
      const imageUrl = URL.createObjectURL(selectedImage);
      setOriginalImageUrl(imageUrl);
      return () => URL.revokeObjectURL(imageUrl);
    }
  }, [selectedImage]);
  
  // Handle image validation completion
  const handleValidationComplete = (
    isValid: boolean, 
    message: string, 
    characteristics: ImageCharacteristics | null
  ) => {
    setIsImageValid(isValid);
    setValidationMessage(message);
    setImageCharacteristics(characteristics);
    
    // If image is valid, set it as active
    if (isValid && selectedImage) {
      setActiveImage(selectedImage);
    }
  };
  
  // Handle image processing completion
  const handleProcessingComplete = (processedFile: File | null) => {
    setProcessedImage(processedFile);
    if (processedFile) {
      setActiveImage(processedFile);
    }
  };
  
  // Definition of process steps
  const steps: ProcessStep[] = [
    { 
      id: 1, 
      title: t('steps.upload'), 
      description: t('steps.upload.desc'), 
      isActive: currentStep === 1, 
      isCompleted: currentStep > 1 
    },
    { 
      id: 2, 
      title: t('steps.validate'), 
      description: t('steps.validate.desc'), 
      isActive: currentStep === 2, 
      isCompleted: currentStep > 2 
    },
    { 
      id: 3, 
      title: t('steps.transform'), 
      description: t('steps.transform.desc'), 
      isActive: currentStep === 3, 
      isCompleted: currentStep > 3 
    },
    { 
      id: 4, 
      title: t('steps.segment'), 
      description: t('steps.segment.desc'), 
      isActive: currentStep === 4, 
      isCompleted: currentStep > 4 
    },
    { 
      id: 5, 
      title: t('steps.recognize'), 
      description: t('steps.recognize.desc'), 
      isActive: currentStep === 5, 
      isCompleted: currentStep > 5 
    },
    { 
      id: 6, 
      title: t('steps.results'), 
      description: t('steps.results.desc'), 
      isActive: currentStep === 6, 
      isCompleted: currentStep > 6 
    }
  ];
  
  const handleImageSelected = (file: File) => {
    setSelectedImage(file);
    setProcessedImage(null);
    setActiveImage(null);
    setError(null);
  };
  
  const handleNextStep = async () => {
    if (currentStep < steps.length) {
      // If we're at step 1 and have an image, start processing
      if (currentStep === 1) {
        if (!selectedImage) {
          setError(t('error.selectImage'));
          return;
        }
        
        // Move to validation step
        setCurrentStep(2);
        setProgress(Math.floor((2 / steps.length) * 100));
      }
      // Validation and preprocessing
      else if (currentStep === 2) {
        if (!activeImage) {
          setError(t('error.meetsRequirements'));
          return;
        }
        
        // Move to transformations step
        setCurrentStep(3);
        setProgress(Math.floor((3 / steps.length) * 100));
      }
      // Apply transformations
      else if (currentStep === 3) {
        // Move to segmentation step
        setCurrentStep(4);
        setProgress(Math.floor((4 / steps.length) * 100));
      }
      // Word and character segmentation
      else if (currentStep === 4) {
        // Move to recognition step
        setCurrentStep(5);
        setProgress(Math.floor((5 / steps.length) * 100));
      }
      // Character recognition
      else if (currentStep === 5) {
        // Start processing with transformations
        setIsProcessing(true);
        try {
          if (activeImage) {
            // Call the full processing endpoint
            const processingResult = await OCRService.processImageWithTransformations(
              activeImage, 
              appliedTransformations
            );
            
            if (processingResult.success) {
              setResult(processingResult);
            } else {
              throw new Error(processingResult.error || "Failed to process the image");
            }
            
            // Move to results step
            setCurrentStep(6);
            setProgress(100);
          }
        } catch (error) {
          console.error("Recognition error:", error);
          setError(error instanceof Error ? error.message : t('error.processing'));
        } finally {
          setIsProcessing(false);
        }
      }
    }
  };
  
  const handlePreviousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      setProgress(Math.floor(((currentStep - 1) / steps.length) * 100));
    }
  };
  
  const handleStepClick = (stepNumber: number) => {
    // Only allow going back to previous steps or current step
    if (stepNumber <= currentStep) {
      setCurrentStep(stepNumber);
      setProgress(Math.floor((stepNumber / steps.length) * 100));
    }
  };
  
  const resetProcess = () => {
    setCurrentStep(1);
    setProgress(0);
    setSelectedImage(null);
    setProcessedImage(null);
    setActiveImage(null);
    setImageCharacteristics(null);
    setIsImageValid(false);
    setResult(null);
    setShowResults(false);
    setError(null);
  };
  
  // Show error notification if present
  useEffect(() => {
    if (error) {
      // Auto-clear error after 5 seconds
      const timer = setTimeout(() => {
        setError(null);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [error]);
  
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="text-center mb-8 p-6">
            <h2 className="text-2xl font-bold mb-4">{t('upload.title')}</h2>
            <div className="max-w-xl mx-auto mb-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
              <div className="flex">
                <FontAwesomeIcon icon={faInfo} className="text-blue-500 mr-3 mt-0.5" />
                <div>
                  <p className="text-sm text-blue-700 text-left">
                    <strong>{t('upload.requirements')}</strong>
                  </p>
                  <ul className="list-disc pl-5 text-sm text-blue-700 mt-1 text-left">
                    <li>{t('upload.requirement1')}</li>
                    <li>{t('upload.requirement2')}</li>
                    <li>{t('upload.requirement3')}</li>
                    <li>{t('upload.requirement4')}</li>
                  </ul>
                </div>
              </div>
            </div>
            <ImageUploader onImageSelected={handleImageSelected} />
            {selectedImage && (
              <div className="mt-4">
                <p className="text-gray-600">
                  {t('upload.selected')} {selectedImage.name} ({Math.round(selectedImage.size / 1024)} KB)
                </p>
                <ImageValidator 
                  imageFile={selectedImage}
                  onValidationComplete={handleValidationComplete}
                />
              </div>
            )}
            <div className="flex justify-end mt-8">
              <Button
                variant="primary"
                onClick={handleNextStep}
                disabled={!selectedImage}
                icon={faArrowRight}
                iconPosition="right"
              >
                {t('button.next')}
              </Button>
            </div>
          </div>
        );
      
      case 2:
        return (
          <div className="text-center mb-8 p-6">
            <h2 className="text-2xl font-bold mb-6">{t('validation.title')}</h2>
            
            {/* Information tooltip */}
            <div className="relative mb-6 inline-block group">
              <button className="px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-sm font-medium flex items-center">
                <FontAwesomeIcon icon={faInfoCircle} className="mr-2" />
                {t('validation.about')}
              </button>
              <div className="absolute z-10 hidden group-hover:block left-0 right-0 md:left-1/2 md:-translate-x-1/2 w-80 md:w-96 p-4 mt-2 bg-white rounded-lg shadow-lg border border-gray-200">
                <div className="text-left">
                  <p className="text-sm text-gray-600 mb-2">{t('validation.aboutDesc')}</p>
                  <div className="grid grid-cols-2 gap-2 mt-3">
                    <div className="text-xs bg-gray-50 p-2 rounded">
                      <span className="text-indigo-600 font-medium block">1.</span> 
                      {t('validation.req1')}
                    </div>
                    <div className="text-xs bg-gray-50 p-2 rounded">
                      <span className="text-indigo-600 font-medium block">2.</span> 
                      {t('validation.req2')}
                    </div>
                    <div className="text-xs bg-gray-50 p-2 rounded">
                      <span className="text-indigo-600 font-medium block">3.</span> 
                      {t('validation.req3')}
                    </div>
                    <div className="text-xs bg-gray-50 p-2 rounded">
                      <span className="text-indigo-600 font-medium block">4.</span> 
                      {t('validation.req4')}
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-3 italic">
                    {t('validation.processDesc')}
                  </p>
                </div>
              </div>
            </div>
            
            {imageCharacteristics ? (
              <ImageProcessor 
                imageFile={selectedImage}
                characteristics={imageCharacteristics}
                onProcessingComplete={handleProcessingComplete}
              />
            ) : selectedImage ? (
              <div className="flex justify-center items-center h-40">
                <ClipLoader color="#6366F1" size={40} />
                <p className="ml-4 text-gray-600">{t('validation.analyzing')}</p>
              </div>
            ) : (
              <div className="text-gray-500 p-8">
                {t('validation.uploadFirst')}
              </div>
            )}
            
            <div className="flex justify-between mt-8">
              <Button
                variant="secondary"
                onClick={handlePreviousStep}
                disabled={isProcessing}
                icon={faArrowLeft}
                iconPosition="left"
              >
                {t('button.previous')}
              </Button>
              <Button
                variant="primary"
                onClick={handleNextStep}
                disabled={isProcessing || !activeImage}
                icon={faArrowRight}
                iconPosition="right"
              >
                {t('button.next')}
              </Button>
            </div>
          </div>
        );
      
      case 3:
        return (
          <div className="mb-8 p-6">
            <h2 className="text-2xl font-bold mb-4 text-center">{t('transform.title')}</h2>
            <p className="text-center text-gray-600 mb-8">
              {t('transform.desc')}
            </p>
            
            <div className="flex flex-wrap gap-4 justify-center">
              <div className="w-36 h-36 bg-white rounded-xl shadow-sm border border-indigo-100 p-3 flex flex-col items-center justify-center relative group cursor-help">
                <div className="absolute inset-0 bg-indigo-500 opacity-0 group-hover:opacity-90 rounded-xl transition-opacity duration-300 flex items-center justify-center p-3">
                  <p className="text-white text-xs text-center">{t('transform.gaussian.desc')}</p>
                </div>
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center mb-3">
                  <FontAwesomeIcon icon={faCheck} className="text-indigo-500" />
                </div>
                <h4 className="font-medium text-center">{t('transform.gaussian.title')}</h4>
              </div>
              
              <div className="w-36 h-36 bg-white rounded-xl shadow-sm border border-indigo-100 p-3 flex flex-col items-center justify-center relative group cursor-help">
                <div className="absolute inset-0 bg-indigo-500 opacity-0 group-hover:opacity-90 rounded-xl transition-opacity duration-300 flex items-center justify-center p-3">
                  <p className="text-white text-xs text-center">{t('transform.saltpepper.desc')}</p>
                </div>
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center mb-3">
                  <FontAwesomeIcon icon={faCheck} className="text-indigo-500" />
                </div>
                <h4 className="font-medium text-center">{t('transform.saltpepper.title')}</h4>
              </div>
              
              <div className="w-36 h-36 bg-white rounded-xl shadow-sm border border-indigo-100 p-3 flex flex-col items-center justify-center relative group cursor-help">
                <div className="absolute inset-0 bg-indigo-500 opacity-0 group-hover:opacity-90 rounded-xl transition-opacity duration-300 flex items-center justify-center p-3">
                  <p className="text-white text-xs text-center">{t('transform.brightness.desc')}</p>
                </div>
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center mb-3">
                  <FontAwesomeIcon icon={faCheck} className="text-indigo-500" />
                </div>
                <h4 className="font-medium text-center">{t('transform.brightness.title')}</h4>
              </div>
              
              <div className="w-36 h-36 bg-white rounded-xl shadow-sm border border-indigo-100 p-3 flex flex-col items-center justify-center relative group cursor-help">
                <div className="absolute inset-0 bg-indigo-500 opacity-0 group-hover:opacity-90 rounded-xl transition-opacity duration-300 flex items-center justify-center p-3">
                  <p className="text-white text-xs text-center">{t('transform.blur.desc')}</p>
                </div>
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center mb-3">
                  <FontAwesomeIcon icon={faCheck} className="text-indigo-500" />
                </div>
                <h4 className="font-medium text-center">{t('transform.blur.title')}</h4>
              </div>
            </div>
            
            <div className="mt-6 flex justify-center">
              <div className="bg-green-50 border-l-4 border-green-400 p-3 rounded-r-lg shadow-sm max-w-3xl">
                <div className="flex">
                  <FontAwesomeIcon icon={faInfo} className="text-green-400 mr-3 mt-0.5" />
                  <div>
                    <p className="text-sm text-green-700">
                      <strong>{t('transform.insight.title')}</strong> {t('transform.insight.desc')}
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex justify-between mt-8">
              <Button
                variant="secondary"
                onClick={handlePreviousStep}
                disabled={isProcessing}
                icon={faArrowLeft}
                iconPosition="left"
              >
                {t('button.previous')}
              </Button>
              <Button
                variant="primary"
                onClick={handleNextStep}
                disabled={isProcessing}
                icon={faArrowRight}
                iconPosition="right"
              >
                {t('button.next')}
              </Button>
            </div>
          </div>
        );
      
      case 4:
        return (
          <div className="text-center mb-8 p-6">
            <h2 className="text-2xl font-bold mb-6">{t('segment.title')}</h2>
            
            <p className="text-gray-600 mb-6">
              {t('segment.desc')}
            </p>
            
            {/* Segmentation process visualization */}
            <div className="flex flex-col items-center mb-8">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl">
                {/* Step 0: Original Image */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">0</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step0')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <img 
                        src={originalImageUrl} 
                        alt="Original" 
                        className="max-h-44 max-w-full object-contain" 
                      />
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation0')}
                  </p>
                </div>
                
                {/* Step 1: Grayscale Conversion */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">1</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step1')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3 relative">
                    {activeImage && originalImageUrl ? (
                      <img 
                        src={originalImageUrl} 
                        alt="Grayscale" 
                        className="max-h-44 max-w-full object-contain grayscale" 
                      />
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation1')}
                  </p>
                </div>
                
                {/* Step 2: Gaussian Blur */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">2</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step2')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <img 
                        src={originalImageUrl} 
                        alt="Gaussian Blur" 
                        className="max-h-44 max-w-full object-contain grayscale blur-sm" 
                      />
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation2')}
                  </p>
                </div>
                
                {/* Step 3: Adaptive Thresholding */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">3</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step3')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <div className="max-h-44 max-w-full">
                        <img 
                          src="/images/segmentation/step3_thresholding.jpg" 
                          alt="Thresholding" 
                          className="max-h-44 max-w-full object-contain" 
                          onError={(e) => {
                            // Fallback to simulation using CSS filters if image not available
                            const target = e.target as HTMLImageElement;
                            target.onerror = null;
                            target.src = originalImageUrl;
                            target.className = "max-h-44 max-w-full object-contain grayscale contrast-200 brightness-150";
                          }}
                        />
                      </div>
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation3')}
                  </p>
                </div>
                
                {/* Step 4: Morphological Operations */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">4</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step4')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <div className="max-h-44 max-w-full">
                        <img 
                          src="/images/segmentation/step4_morphology.jpg" 
                          alt="Morphology" 
                          className="max-h-44 max-w-full object-contain" 
                          onError={(e) => {
                            // Fallback to simulation using CSS filters if image not available
                            const target = e.target as HTMLImageElement;
                            target.onerror = null;
                            target.src = originalImageUrl;
                            target.className = "max-h-44 max-w-full object-contain grayscale contrast-150 brightness-125";
                          }}
                        />
                      </div>
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation4')}
                  </p>
                </div>
                
                {/* Step 5: Find Contours */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">5</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step5')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <div className="max-h-44 max-w-full">
                        <img 
                          src="/images/segmentation/step5_contours.jpg" 
                          alt="Contours" 
                          className="max-h-44 max-w-full object-contain" 
                          onError={(e) => {
                            // Fallback to simulation using CSS filters if image not available
                            const target = e.target as HTMLImageElement;
                            target.onerror = null;
                            target.src = originalImageUrl;
                            target.className = "max-h-44 max-w-full object-contain grayscale invert contrast-150";
                          }}
                        />
                      </div>
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation5')}
                  </p>
                </div>
                
                {/* Step 6: Filter Contours */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">6</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step6')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <div className="max-h-44 max-w-full">
                        <img 
                          src="/images/segmentation/step6_filtered.jpg" 
                          alt="Filtered Contours" 
                          className="max-h-44 max-w-full object-contain" 
                          onError={(e) => {
                            // Fallback to simulation using CSS filters if image not available
                            const target = e.target as HTMLImageElement;
                            target.onerror = null;
                            target.src = originalImageUrl;
                            target.className = "max-h-44 max-w-full object-contain grayscale invert contrast-150 brightness-125";
                          }}
                        />
                      </div>
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation6')}
                  </p>
                </div>
                
                {/* Step 7: Extract Characters */}
                <div 
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-bold text-indigo-700">7</span>
                  </div>
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.step7')}</h3>
                  
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                    {activeImage && originalImageUrl ? (
                      <div className="max-h-44 max-w-full">
                        <img 
                          src="/images/segmentation/step7_extraction.jpg" 
                          alt="Character Extraction" 
                          className="max-h-44 max-w-full object-contain" 
                          onError={(e) => {
                            // Fallback to simulation - show colored boxes around potential characters
                            const target = e.target as HTMLImageElement;
                            target.onerror = null;
                            target.src = originalImageUrl;
                            target.style.border = "2px solid #6366F1";
                            target.style.boxSizing = "border-box";
                          }}
                        />
                      </div>
                    ) : (
                      <div className="text-gray-400 text-sm">{t('segment.noImage')}</div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-600">
                    {t('segment.stepExplanation7')}
                  </p>
                </div>
                
                {/* Flow diagram - optional */}
                <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200 lg:col-span-3">
                  <h3 className="text-indigo-700 font-medium text-center mb-3">{t('segment.process')}</h3>
                  
                  <div className="flex flex-wrap justify-center gap-2 items-center">
                    {[
                      t('segment.step1'),
                      t('segment.step2'),
                      t('segment.step3'),
                      t('segment.step4'),
                      t('segment.step5'),
                      t('segment.step6'),
                      t('segment.step7')
                    ].map((step, index) => (
                      <React.Fragment key={index}>
                        <div className="px-3 py-2 rounded-lg text-xs shadow-sm border border-gray-100 bg-white">
                          <span className="font-medium text-indigo-600 mr-1">{index + 1}.</span> {step}
                        </div>
                        {index < 6 && <FontAwesomeIcon icon={faArrowRight} className="text-gray-400 mx-1" />}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex justify-between mt-8">
              <Button
                variant="secondary"
                onClick={handlePreviousStep}
                disabled={isProcessing}
                icon={faArrowLeft}
                iconPosition="left"
              >
                {t('button.previous')}
              </Button>
              <Button
                variant="primary"
                onClick={handleNextStep}
                disabled={isProcessing}
                icon={faArrowRight}
                iconPosition="right"
              >
                {t('button.next')}
              </Button>
            </div>
          </div>
        );
      
      case 5:
        return (
          <div className="text-center mb-8 p-6">
            <h2 className="text-2xl font-bold mb-6">{t('recognition.title')}</h2>
            
            {isProcessing ? (
              <div className="flex justify-center items-center h-64">
                <ClipLoader color="#6366F1" size={60} />
                <p className="ml-4 text-gray-600">{t('recognition.processing')}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                <div className="w-full max-w-6xl mb-8">
                  {/* Recognition visualization */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    {/* Step 1: Apply degradation techniques */}
                    <div 
                      className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                    >
                      <div className="flex items-center mb-3">
                        <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mr-3">
                          <span className="text-sm font-bold text-indigo-700">1</span>
                        </div>
                        <h3 className="text-indigo-700 font-medium">{t('recognition.step1')}</h3>
                      </div>
                      
                      <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                        {activeImage && originalImageUrl ? (
                          <div className="flex space-x-2">
                            <img 
                              src={originalImageUrl} 
                              alt="Original" 
                              className="max-h-44 max-w-[45%] object-contain" 
                            />
                            <FontAwesomeIcon icon={faArrowRight} className="text-gray-400 self-center" />
                            <img 
                              src={originalImageUrl} 
                              alt="Degraded" 
                              className="max-h-44 max-w-[45%] object-contain grayscale brightness-110 contrast-125" 
                            />
                          </div>
                        ) : (
                          <div className="text-gray-400 text-sm">{t('recognition.noImage')}</div>
                        )}
                      </div>
                      
                      <p className="text-xs text-gray-600">
                        {t('recognition.stepExplanation1')}
                      </p>
                    </div>
                    
                    {/* Step 2: Preprocess images */}
                    <div 
                      className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                    >
                      <div className="flex items-center mb-3">
                        <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mr-3">
                          <span className="text-sm font-bold text-indigo-700">2</span>
                        </div>
                        <h3 className="text-indigo-700 font-medium">{t('recognition.step2')}</h3>
                      </div>
                      
                      <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                        {activeImage && originalImageUrl ? (
                          <div className="max-h-44 max-w-full">
                            <img 
                              src="/images/recognition/step2_preprocessing.jpg" 
                              alt="Preprocessing" 
                              className="max-h-44 max-w-full object-contain" 
                              onError={(e) => {
                                // Fallback to simulation using CSS filters if image not available
                                const target = e.target as HTMLImageElement;
                                target.onerror = null;
                                target.src = originalImageUrl;
                                target.className = "max-h-44 max-w-full object-contain grayscale contrast-150 brightness-110";
                              }}
                            />
                          </div>
                        ) : (
                          <div className="text-gray-400 text-sm">{t('recognition.noImage')}</div>
                        )}
                      </div>
                      
                      <p className="text-xs text-gray-600">
                        {t('recognition.stepExplanation2')}
                      </p>
                    </div>
                    
                    {/* Step 3: Segment characters */}
                    <div 
                      className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                    >
                      <div className="flex items-center mb-3">
                        <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mr-3">
                          <span className="text-sm font-bold text-indigo-700">3</span>
                        </div>
                        <h3 className="text-indigo-700 font-medium">{t('recognition.step3')}</h3>
                      </div>
                      
                      <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                        {activeImage && originalImageUrl ? (
                          <div className="max-h-44 max-w-full">
                            <img 
                              src="/images/recognition/step3_segmentation.jpg" 
                              alt="Segmentation" 
                              className="max-h-44 max-w-full object-contain" 
                              onError={(e) => {
                                // Fallback to simulation - show colored boxes around potential characters
                                const target = e.target as HTMLImageElement;
                                target.onerror = null;
                                target.src = originalImageUrl;
                                target.style.border = "2px solid #6366F1";
                                target.style.boxSizing = "border-box";
                              }}
                            />
                          </div>
                        ) : (
                          <div className="text-gray-400 text-sm">{t('recognition.noImage')}</div>
                        )}
                      </div>
                      
                      <p className="text-xs text-gray-600">
                        {t('recognition.stepExplanation3')}
                      </p>
                    </div>
                    
                    {/* Step 4: Prepare character segments */}
                    <div 
                      className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                    >
                      <div className="flex items-center mb-3">
                        <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mr-3">
                          <span className="text-sm font-bold text-indigo-700">4</span>
                        </div>
                        <h3 className="text-indigo-700 font-medium">{t('recognition.step4')}</h3>
                      </div>
                      
                      <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                        {activeImage && originalImageUrl ? (
                          <div className="grid grid-cols-4 gap-2 p-2">
                            {Array.from({ length: 8 }).map((_, i) => (
                              <div 
                                key={i} 
                                className="w-16 h-16 bg-gray-200 flex items-center justify-center rounded"
                              >
                                <span className="text-xs text-gray-600">32x32</span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-gray-400 text-sm">{t('recognition.noImage')}</div>
                        )}
                      </div>
                      
                      <div className="flex flex-wrap gap-2 mt-2">
                        <span className="inline-block bg-indigo-50 text-indigo-700 text-xs px-2 py-1 rounded">
                          {t('recognition.step4a')}
                        </span>
                        <span className="inline-block bg-indigo-50 text-indigo-700 text-xs px-2 py-1 rounded">
                          {t('recognition.step4b')}
                        </span>
                        <span className="inline-block bg-indigo-50 text-indigo-700 text-xs px-2 py-1 rounded">
                          {t('recognition.step4c')}
                        </span>
                        <span className="inline-block bg-indigo-50 text-indigo-700 text-xs px-2 py-1 rounded">
                          {t('recognition.step4d')}
                        </span>
                      </div>
                    </div>
                    
                    {/* Step 5: CNN Model Prediction */}
                    <div 
                      className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                    >
                      <div className="flex items-center mb-3">
                        <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mr-3">
                          <span className="text-sm font-bold text-indigo-700">5</span>
                        </div>
                        <h3 className="text-indigo-700 font-medium">{t('recognition.step5')}</h3>
                      </div>
                      
                      <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                        <div className="flex flex-col items-center">
                          <div className="flex space-x-2 mb-2">
                            <div className="w-12 h-12 bg-gray-200 rounded"></div>
                            <FontAwesomeIcon icon={faArrowRight} className="text-gray-400 self-center" />
                            <div className="bg-indigo-100 p-2 rounded text-sm text-center">
                              <div className="font-bold">CNN</div>
                              <div className="text-xs">MobileNetV2</div>
                            </div>
                            <FontAwesomeIcon icon={faArrowRight} className="text-gray-400 self-center" />
                            <div className="bg-green-100 p-2 rounded text-sm">
                              <div>A: 98%</div>
                            </div>
                          </div>
                          <div className="text-xs text-gray-500 italic">Prediction process for each character</div>
                        </div>
                      </div>
                      
                      <p className="text-xs text-gray-600">
                        {t('recognition.stepExplanation5')}
                      </p>
                    </div>
                    
                    {/* Step 6: Combine Results */}
                    <div 
                      className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                    >
                      <div className="flex items-center mb-3">
                        <div className="bg-indigo-50 rounded-full w-8 h-8 flex items-center justify-center mr-3">
                          <span className="text-sm font-bold text-indigo-700">6</span>
                        </div>
                        <h3 className="text-indigo-700 font-medium">{t('recognition.step6')}</h3>
                      </div>
                      
                      <div className="h-48 flex items-center justify-center bg-gray-50 rounded-lg mb-3">
                        <div className="flex flex-col items-center space-y-4">
                          <div className="bg-green-50 border border-green-200 rounded p-2 w-40 text-center">
                            <div className="text-xs text-gray-600">Degraded Result:</div>
                            <div className="font-bold text-green-700">HELLO</div>
                            <div className="text-xs text-gray-500">Confidence: 96%</div>
                          </div>
                          
                          <div className="bg-gray-50 border border-gray-200 rounded p-2 w-40 text-center">
                            <div className="text-xs text-gray-600">Original Result:</div>
                            <div className="font-bold text-gray-700">HFLLO</div>
                            <div className="text-xs text-gray-500">Confidence: 78%</div>
                          </div>
                        </div>
                      </div>
                      
                      <p className="text-xs text-gray-600">
                        {t('recognition.stepExplanation6')}
                      </p>
                    </div>
                  </div>
                  
                  {/* Process flow */}
                  <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200 mb-6">
                    <h3 className="text-indigo-700 font-medium text-center mb-3">{t('recognition.process')}</h3>
                    
                    <div className="flex flex-wrap justify-center gap-2 items-center">
                      {[
                        t('recognition.step1'),
                        t('recognition.step2'),
                        t('recognition.step3'),
                        t('recognition.step4'),
                        t('recognition.step5'),
                        t('recognition.step6')
                      ].map((step, index) => (
                        <React.Fragment key={index}>
                          <div className="px-3 py-2 rounded-lg text-xs shadow-sm border border-gray-100 bg-white">
                            <span className="font-medium text-indigo-600 mr-1">{index + 1}.</span> {step}
                          </div>
                          {index < 5 && <FontAwesomeIcon icon={faArrowRight} className="text-gray-400 mx-1" />}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                  
                  <Button
                    variant="primary"
                    onClick={handleNextStep}
                    disabled={isProcessing}
                    icon={faSync}
                    iconPosition="left"
                    className="px-8 py-3"
                  >
                    {t('button.processImage')}
                  </Button>
                </div>
              </div>
            )}
            
            <div className="flex justify-between mt-8">
              <Button
                variant="secondary"
                onClick={handlePreviousStep}
                disabled={isProcessing}
                icon={faArrowLeft}
                iconPosition="left"
              >
                {t('button.previous')}
              </Button>
            </div>
          </div>
        );
      
      case 6:
        return (
          <div className="mb-8 p-6">
            <h2 className="text-2xl font-bold mb-6 text-center">{t('results.title')}</h2>
            {result && (
              <div className="flex flex-col space-y-6">
                {/* Images Section - Horizontal Layout */}
                <div className="flex flex-wrap md:flex-nowrap gap-4">
                  <div className="w-full md:w-1/3 bg-white p-4 rounded-lg shadow-sm">
                    <h3 className="text-lg font-semibold mb-3">{t('results.originalImage')}</h3>
                    <div className="mb-2">
                      <img src={result.original_image} alt="Original" className="w-full h-auto rounded-lg" />
                    </div>
                  </div>
                  
                  <div className="w-full md:w-1/3 bg-white p-4 rounded-lg shadow-sm">
                    <h3 className="text-lg font-semibold mb-3">{t('results.degradedImage')}</h3>
                    <div className="mb-2">
                      <img src={result.degraded_image} alt="Degraded" className="w-full h-auto rounded-lg" />
                    </div>
                  </div>
                  
                  <div className="w-full md:w-1/3 bg-white p-4 rounded-lg shadow-sm">
                    <h3 className="text-lg font-semibold mb-3">{t('results.segmentedChars')}</h3>
                    <div className="mb-2">
                      <img src={result.segmented_image} alt="Segmented" className="w-full h-auto rounded-lg" />
                    </div>
                  </div>
                </div>
                
                {/* Horizontal layout for text results and analysis */}
                <div className="flex flex-wrap md:flex-nowrap gap-4">
                  <div className="w-full md:w-1/2 bg-white p-4 rounded-lg shadow-sm">
                    <h3 className="text-lg font-semibold mb-3">{t('results.extractedText')}</h3>
                    <div className="bg-gray-50 p-3 rounded border min-h-[120px] max-h-[200px] overflow-auto">
                      {result.model_results?.degraded_text || t('results.noText')}
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      {t('results.confidence')} {result.model_results?.degraded_confidence.toFixed(2) || 0}%
                    </p>
                  </div>
                  
                  <div className="w-full md:w-1/2 bg-white p-4 rounded-lg shadow-sm">
                    <h3 className="text-lg font-semibold mb-3">{t('results.analysis')}</h3>
                    <div className="bg-gray-50 p-3 rounded border min-h-[120px] max-h-[200px] overflow-auto">
                      <p className="text-sm whitespace-pre-wrap">
                        {result.model_results?.comparison || "No comparison available"}
                      </p>
                    </div>
                    
                    {/* Transformations applied */}
                    <div className="mt-3">
                      <p className="text-sm font-medium text-gray-700">{t('results.appliedTransformations')}</p>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {result.transformation_info?.map((info: string, index: number) => (
                          <span key={index} className="inline-block bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded">
                            {info}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Add ResultActions component */}
                <div className="mt-4">
                  <ResultActions result={result} />
                </div>
                
                {/* Note as a floating notification/tooltip */}
                <div className="bg-blue-50 border-l-4 border-blue-400 p-3 text-sm rounded-r-lg shadow-sm max-w-3xl mx-auto">
                  <p className="text-blue-700">
                    <strong>{t('results.note')}</strong> {t('results.noteDesc')}
                  </p>
                </div>
              </div>
            )}
            <div className="flex justify-between mt-8">
              <Button
                variant="secondary"
                onClick={handlePreviousStep}
                icon={faArrowLeft}
                iconPosition="left"
              >
                {t('button.previous')}
              </Button>
              <Button
                variant="success"
                onClick={resetProcess}
                icon={faCheckCircle}
                iconPosition="left"
              >
                {t('button.startAgain')}
              </Button>
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="container mx-auto px-4 py-8 max-w-6xl"
    >
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="mb-10 text-center"
      >
        <h1 className="text-3xl font-bold text-gray-800 mb-4">{t('app.title')}</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          {t('app.subtitle')}
        </p>
      </motion.div>

      {/* Error Notification */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 rounded"
        >
          <div className="flex items-center">
            <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-500 mr-3" />
            <p className="text-red-700">{error}</p>
          </div>
        </motion.div>
      )}

      {!showResults ? (
        <>
          {/* Stepper */}
          <ProcessStepper 
            steps={steps} 
            currentStep={currentStep} 
            progress={progress}
            onStepClick={handleStepClick}
          />
          
          {/* Step Content */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
          >
            {renderStepContent()}
          </motion.div>
        </>
      ) : (
        /* Results Section */
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden p-6"
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mb-8"
          >
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              Résultats d'analyse OCR
            </h2>
            <p className="text-gray-600">
              Voici un récapitulatif complet des résultats de l'analyse OCR effectuée sur votre image.
            </p>
          </motion.div>
          
          {result && (
            <>
              {/* Images Comparison */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.4 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10"
              >
                <motion.div 
                  whileHover={{ y: -5 }}
                  className="bg-gray-50 rounded-xl border border-gray-200 p-4"
                >
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Original Image</h3>
                  <img 
                    src={result.original_image} 
                    alt="Original" 
                    className="w-full h-auto object-contain rounded-lg"
                  />
                </motion.div>
                <motion.div 
                  whileHover={{ y: -5 }}
                  className="bg-green-50 rounded-xl border border-green-200 p-4"
                >
                  <h3 className="text-sm font-medium text-green-700 mb-2">Degraded Image (Best Results)</h3>
                  <img 
                    src={result.degraded_image} 
                    alt="Degraded" 
                    className="w-full h-auto object-contain rounded-lg"
                  />
                </motion.div>
                <motion.div 
                  whileHover={{ y: -5 }}
                  className="bg-gray-50 rounded-xl border border-gray-200 p-4"
                >
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Segmented Characters</h3>
                  <img 
                    src={result.segmented_image} 
                    alt="Segmented" 
                    className="w-full h-auto object-contain rounded-lg"
                  />
                </motion.div>
              </motion.div>
              
              {/* Analysis Results */}
              {result.model_results?.comparison && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                  className="bg-gray-50 rounded-xl p-6 mb-10"
                >
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h3>
                  <motion.div 
                    whileHover={{ scale: 1.02 }}
                    className="flex items-center justify-center p-4 bg-white rounded-lg shadow-sm border border-gray-100"
                  >
                    <div className="text-lg text-indigo-700 font-medium flex items-center">
                      <FontAwesomeIcon icon={faCheckCircle} className="mr-3 text-green-500" />
                      {result.model_results.comparison}
                    </div>
                  </motion.div>
                </motion.div>
              )}
              
              {/* Text Results */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.8 }}
                className="mb-10"
              >
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">
                    Extracted Text
                  </h3>
                  <div className="relative">
                    <pre className="whitespace-pre-wrap text-gray-700 text-sm font-mono bg-gray-50 p-4 rounded-lg overflow-auto h-64 border border-gray-200">
                      {result.model_results?.degraded_text || "No text detected"}
                    </pre>
                  </div>
                  <div className="text-sm text-gray-600 mt-2">
                    Confidence: {result.model_results?.degraded_confidence.toFixed(2) || 0}%
                  </div>
                </div>
              </motion.div>
            </>
          )}
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1 }}
            className="flex justify-between items-center"
          >
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                variant="outline"
                icon={faArrowLeft}
                iconPosition="left"
                onClick={resetProcess}
              >
                Recommencer
              </Button>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default Process; 
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faArrowLeft, faChartLine, faCheckCircle, faExclamationTriangle, faSync, faInfo, faCheck, faImage } from '@fortawesome/free-solid-svg-icons';
import { ClipLoader } from 'react-spinners';
import ProcessStepper from './ProcessStepper';
import ImageUploader from './ImageUploader';
import Button from '../common/Button';
import { OCRService } from '../../services/api';
import { ProcessStep, OCRResult, ImageTransformation } from '../../types/types';
import ImageValidator, { ImageCharacteristics } from './ImageValidator';
import ImageProcessor from './ImageProcessor';
import { useLanguage } from '../../contexts/LanguageContext';

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
  
  const renderTransformationInfo = () => {
    return (
      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-4">{t('transform.techniques.title')}</h3>
        <p className="text-sm text-gray-600 mb-4">
          {t('transform.techniques.description')}
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 border rounded-lg bg-indigo-50 border-indigo-200">
            <div className="flex items-center">
              <div className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center mr-3">
                <FontAwesomeIcon icon={faCheck} className="text-xs" />
              </div>
              <div className="flex-grow">
                <h4 className="font-medium">{t('transform.gaussian.title')}</h4>
                <p className="text-xs text-gray-500">{t('transform.gaussian.desc')}</p>
              </div>
            </div>
          </div>
          
          <div className="p-4 border rounded-lg bg-indigo-50 border-indigo-200">
            <div className="flex items-center">
              <div className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center mr-3">
                <FontAwesomeIcon icon={faCheck} className="text-xs" />
              </div>
              <div className="flex-grow">
                <h4 className="font-medium">{t('transform.saltpepper.title')}</h4>
                <p className="text-xs text-gray-500">{t('transform.saltpepper.desc')}</p>
              </div>
            </div>
          </div>
          
          <div className="p-4 border rounded-lg bg-indigo-50 border-indigo-200">
            <div className="flex items-center">
              <div className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center mr-3">
                <FontAwesomeIcon icon={faCheck} className="text-xs" />
              </div>
              <div className="flex-grow">
                <h4 className="font-medium">{t('transform.brightness.title')}</h4>
                <p className="text-xs text-gray-500">{t('transform.brightness.desc')}</p>
              </div>
            </div>
          </div>
          
          <div className="p-4 border rounded-lg bg-indigo-50 border-indigo-200">
            <div className="flex items-center">
              <div className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center mr-3">
                <FontAwesomeIcon icon={faCheck} className="text-xs" />
              </div>
              <div className="flex-grow">
                <h4 className="font-medium">{t('transform.blur.title')}</h4>
                <p className="text-xs text-gray-500">{t('transform.blur.desc')}</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 bg-green-50 border-l-4 border-green-400 p-4 rounded">
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
    );
  };

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
            <div className="max-w-xl mx-auto mb-6 bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded text-left">
              <div className="flex">
                <FontAwesomeIcon icon={faImage} className="text-indigo-500 mr-3 mt-0.5" />
                <div>
                  <p className="text-sm text-indigo-700">
                    <strong>{t('validation.about')}</strong> {t('validation.aboutDesc')}
                  </p>
                  <ul className="list-disc pl-5 text-sm text-indigo-700 mt-1">
                    <li>{t('validation.req1')}</li>
                    <li>{t('validation.req2')}</li>
                    <li>{t('validation.req3')}</li>
                    <li>{t('validation.req4')}</li>
                  </ul>
                  <p className="text-sm text-indigo-700 mt-1">
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
            
            {renderTransformationInfo()}
            
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
            <h2 className="text-2xl font-bold mb-8">{t('segment.title')}</h2>
            <p className="text-gray-600 mb-8">
              {t('segment.desc')}
            </p>
            <div className="flex flex-col items-center justify-center">
              <div className="max-w-xl bg-gray-50 p-4 rounded-lg border border-gray-200 mb-8">
                <h3 className="font-medium text-gray-700 mb-2 text-left">{t('segment.process')}</h3>
                <ol className="list-decimal pl-5 text-sm text-gray-600 space-y-2 text-left">
                  <li>{t('segment.step1')}</li>
                  <li>{t('segment.step2')}</li>
                  <li>{t('segment.step3')}</li>
                  <li>{t('segment.step4')}</li>
                  <li>{t('segment.step5')}</li>
                  <li>{t('segment.step6')}</li>
                  <li>{t('segment.step7')}</li>
                </ol>
              </div>
              {activeImage && originalImageUrl && (
                <div className="max-w-md mx-auto">
                  <img 
                    src={originalImageUrl} 
                    alt="Segmentation Preview" 
                    className="w-full h-auto rounded-lg shadow-md opacity-70" 
                  />
                  <div className="mt-2 text-sm text-gray-500 italic">
                    {t('segment.preview')}
                  </div>
                </div>
              )}
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
            <h2 className="text-2xl font-bold mb-8">{t('recognition.title')}</h2>
            {isProcessing ? (
              <div className="flex justify-center items-center h-64">
                <ClipLoader color="#6366F1" size={60} />
                <p className="ml-4 text-gray-600">{t('recognition.processing')}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                <div className="max-w-xl bg-gray-50 p-4 rounded-lg border border-gray-200 mb-8">
                  <h3 className="font-medium text-gray-700 mb-2 text-left">{t('recognition.process')}</h3>
                  <ol className="list-decimal pl-5 text-sm text-gray-600 space-y-2 text-left">
                    <li>{t('recognition.step1')}</li>
                    <li>{t('recognition.step2')}</li>
                    <li>{t('recognition.step3')}</li>
                    <li>{t('recognition.step4')}
                      <ul className="list-disc pl-5 mt-1 mb-1">
                        <li>{t('recognition.step4a')}</li>
                        <li>{t('recognition.step4b')}</li>
                        <li>{t('recognition.step4c')}</li>
                        <li>{t('recognition.step4d')}</li>
                      </ul>
                    </li>
                    <li>{t('recognition.step5')}</li>
                    <li>{t('recognition.step6')}</li>
                  </ol>
                </div>
                <Button
                  variant="primary"
                  onClick={handleNextStep}
                  disabled={isProcessing}
                  icon={faSync}
                  iconPosition="left"
                >
                  {t('button.processImage')}
                </Button>
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
            <h2 className="text-2xl font-bold mb-8 text-center">{t('results.title')}</h2>
            {result && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-lg shadow-md">
                  <h3 className="text-lg font-semibold mb-4">{t('results.originalImage')}</h3>
                  <div className="mb-4">
                    <img src={result.original_image} alt="Original" className="w-full h-auto rounded-lg" />
                  </div>
                </div>
                
                <div className="bg-white p-6 rounded-lg shadow-md">
                  <h3 className="text-lg font-semibold mb-4">{t('results.degradedImage')}</h3>
                  <div className="mb-4">
                    <img src={result.degraded_image} alt="Degraded" className="w-full h-auto rounded-lg" />
                  </div>
                  <div className="text-sm text-gray-600">
                    <p>{t('results.appliedTransformations')}</p>
                    <ul className="list-disc pl-5 mt-1">
                      {result.transformation_info?.map((info: string, index: number) => (
                        <li key={index}>{info}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                
                <div className="bg-white p-6 rounded-lg shadow-md">
                  <h3 className="text-lg font-semibold mb-4">{t('results.segmentedChars')}</h3>
                  <div className="mb-4">
                    <img src={result.segmented_image} alt="Segmented" className="w-full h-auto rounded-lg" />
                  </div>
                </div>
                
                <div className="bg-white p-6 rounded-lg shadow-md col-span-1 md:col-span-2">
                  <h3 className="text-lg font-semibold mb-4">{t('results.modelResults')}</h3>
                  
                  <div className="mb-6">
                    <h4 className="font-medium mb-2">{t('results.extractedText')}</h4>
                    <div className="bg-gray-50 p-3 rounded border">
                      {result.model_results?.degraded_text || t('results.noText')}
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      {t('results.confidence')} {result.model_results?.degraded_confidence.toFixed(2) || 0}%
                    </p>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">{t('results.analysis')}</h4>
                    <div className="bg-gray-50 p-3 rounded border">
                      <p className="text-sm whitespace-pre-wrap">
                        {result.model_results?.comparison || "No comparison available"}
                      </p>
                    </div>
                    
                    {/* Note about confidence */}
                    <div className="mt-4 bg-blue-50 border-l-4 border-blue-400 p-3 text-sm">
                      <p className="text-blue-700">
                        <strong>{t('results.note')}</strong> {t('results.noteDesc')}
                      </p>
                    </div>
                  </div>
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
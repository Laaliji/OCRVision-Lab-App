import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faArrowLeft, faTimes, faPlay, faPause, faStepForward, faStepBackward, faImage } from '@fortawesome/free-solid-svg-icons';
import { useLanguage } from '../../contexts/LanguageContext';

interface GuidedTourProps {
  isOpen: boolean;
  onClose: () => void;
  steps: {
    title: string;
    description: string;
    image?: string;
  }[];
  processType: 'segmentation' | 'recognition';
}

const GuidedTour: React.FC<GuidedTourProps> = ({ isOpen, onClose, steps, processType }) => {
  const { t } = useLanguage();
  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);
  const [imageLoadFailed, setImageLoadFailed] = useState<Record<number, boolean>>({});
  
  // Reset step when opened
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
      stopAutoPlay();
      setImageLoadFailed({});
    }
  }, [isOpen]);
  
  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);
  
  const nextStep = () => {
    setCurrentStep(prev => {
      if (prev >= steps.length - 1) {
        stopAutoPlay();
        return steps.length - 1;
      }
      return prev + 1;
    });
  };
  
  const prevStep = () => {
    setCurrentStep(prev => (prev > 0 ? prev - 1 : 0));
    stopAutoPlay();
  };
  
  const startAutoPlay = () => {
    if (isAutoPlaying) return;
    
    setIsAutoPlaying(true);
    const id = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          clearInterval(id);
          setIsAutoPlaying(false);
          return steps.length - 1;
        }
        return prev + 1;
      });
    }, 3000); // Change step every 3 seconds
    
    setIntervalId(id);
  };
  
  const stopAutoPlay = () => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
    setIsAutoPlaying(false);
  };
  
  const handleImageError = (index: number) => {
    setImageLoadFailed(prev => ({
      ...prev,
      [index]: true
    }));
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-800">
            {processType === 'segmentation' ? t('guidedTour.segmentationTitle') : t('guidedTour.recognitionTitle')}
          </h2>
          <button 
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <FontAwesomeIcon icon={faTimes} />
          </button>
        </div>
        
        {/* Progress indicator */}
        <div className="px-4 py-2 bg-gray-50">
          <div className="flex justify-between items-center text-xs text-gray-500">
            <span>{t('guidedTour.step')} {currentStep + 1} / {steps.length}</span>
            <span>{Math.round(((currentStep + 1) / steps.length) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 h-1 mt-1 rounded-full overflow-hidden">
            <div 
              className="bg-indigo-500 h-full transition-all duration-300 ease-out"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            ></div>
          </div>
        </div>
        
        {/* Content */}
        <div className="flex-grow overflow-auto p-6">
          <div className="mb-4">
            <h3 className="text-lg font-medium text-indigo-700 mb-2">{steps[currentStep].title}</h3>
            <p className="text-gray-600">{steps[currentStep].description}</p>
          </div>
          
          {steps[currentStep].image && !imageLoadFailed[currentStep] ? (
            <div className="mt-4 flex justify-center">
              <img 
                src={steps[currentStep].image} 
                alt={steps[currentStep].title}
                className="max-h-72 rounded-lg shadow-md" 
                onError={() => handleImageError(currentStep)}
              />
            </div>
          ) : (
            <div className="mt-4 flex justify-center">
              <div className="bg-gray-100 rounded-lg shadow-md w-full max-w-md h-56 flex flex-col items-center justify-center text-gray-400">
                <FontAwesomeIcon icon={faImage} size="3x" className="mb-3" />
                <p className="text-sm">{t('guidedTour.imageNotAvailable')}</p>
              </div>
            </div>
          )}
        </div>
        
        {/* Controls */}
        <div className="p-4 border-t flex items-center justify-between">
          <div>
            <button
              onClick={prevStep}
              disabled={currentStep === 0}
              className="px-3 py-1 bg-gray-200 rounded-md text-gray-700 disabled:opacity-50 mr-2"
            >
              <FontAwesomeIcon icon={faArrowLeft} className="mr-1" /> {t('guidedTour.prev')}
            </button>
            
            <button
              onClick={nextStep}
              disabled={currentStep >= steps.length - 1}
              className="px-3 py-1 bg-gray-200 rounded-md text-gray-700 disabled:opacity-50"
            >
              {t('guidedTour.next')} <FontAwesomeIcon icon={faArrowRight} className="ml-1" />
            </button>
          </div>
          
          <div>
            {isAutoPlaying ? (
              <button
                onClick={stopAutoPlay}
                className="px-3 py-1 bg-red-500 text-white rounded-md"
              >
                <FontAwesomeIcon icon={faPause} className="mr-1" /> {t('guidedTour.pause')}
              </button>
            ) : (
              <button
                onClick={startAutoPlay}
                disabled={currentStep >= steps.length - 1}
                className="px-3 py-1 bg-green-500 text-white rounded-md disabled:opacity-50"
              >
                <FontAwesomeIcon icon={faPlay} className="mr-1" /> {t('guidedTour.play')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GuidedTour; 
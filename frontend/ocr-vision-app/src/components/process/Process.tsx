import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faArrowLeft, faChartLine, faCheckCircle, faExclamationTriangle } from '@fortawesome/free-solid-svg-icons';
import { ClipLoader } from 'react-spinners';
import ProcessStepper from './ProcessStepper';
import ImageUploader from './ImageUploader';
import Button from '../common/Button';
import { OCRService } from '../../services/api';
import { ProcessStep, OCRResult } from '../../types/types';

const Process: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [progress, setProgress] = useState<number>(0);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Définition des étapes du processus
  const steps: ProcessStep[] = [
    { id: 1, title: 'Upload', description: 'Téléchargement de l\'image', isActive: currentStep === 1, isCompleted: currentStep > 1 },
    { id: 2, title: 'Analyse', description: 'Analyse initiale de l\'image', isActive: currentStep === 2, isCompleted: currentStep > 2 },
    { id: 3, title: 'Traitement', description: 'Prétraitement avancé', isActive: currentStep === 3, isCompleted: currentStep > 3 },
    { id: 4, title: 'Extraction', description: 'Extraction du texte', isActive: currentStep === 4, isCompleted: currentStep > 4 },
  ];
  
  const handleImageSelected = (file: File) => {
    setSelectedImage(file);
    setError(null);
  };
  
  const handleNextStep = async () => {
    if (currentStep < steps.length) {
      // Si on est à l'étape 1 et qu'on a une image, on démarre le traitement
      if (currentStep === 1 && selectedImage) {
        setIsProcessing(true);
        setError(null);
        
        try {
          // Simulation de progression pour l'étape 2
          setProgress(25);
          setCurrentStep(2);
          
          // Après un court délai, passer à l'étape 3
          setTimeout(() => {
            setProgress(50);
            setCurrentStep(3);
            
            // Après un autre délai, passer à l'étape 4 et appeler l'API
            setTimeout(async () => {
              setProgress(75);
              setCurrentStep(4);
              
              // Appel à l'API pour traiter l'image
              try {
                const ocrResult = await OCRService.processImage(selectedImage);
                if (ocrResult.success === false) {
                  throw new Error(ocrResult.error || "Une erreur est survenue lors du traitement");
                }
                setResult(ocrResult);
                setProgress(100);
              } catch (error) {
                console.error('Error processing image:', error);
                setError(error instanceof Error ? error.message : "Une erreur est survenue lors du traitement");
              } finally {
                setIsProcessing(false);
              }
            }, 3000);
          }, 3000);
        } catch (error) {
          console.error('Error during processing:', error);
          setError(error instanceof Error ? error.message : "Une erreur est survenue");
          setIsProcessing(false);
        }
      } else {
        // Progression normale pour les autres étapes
        setCurrentStep(currentStep + 1);
        setProgress(((currentStep + 1) / steps.length) * 100);
      }
    }
  };
  
  const resetProcess = () => {
    setCurrentStep(1);
    setProgress(0);
    setSelectedImage(null);
    setResult(null);
    setShowResults(false);
    setError(null);
  };
  
  const viewFullResults = () => {
    setShowResults(true);
  };
  
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.6 }
    },
    exit: { 
      opacity: 0, 
      y: -20,
      transition: { duration: 0.3 }
    }
  };
  
  const contentVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        staggerChildren: 0.2,
        delayChildren: 0.1,
        duration: 0.5 
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.4 }
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
        <h1 className="text-3xl font-bold text-gray-800 mb-4">Extraction OCR</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Téléchargez votre image, suivez le processus et obtenez un texte extrait de haute qualité grâce à notre traitement d'image avancé.
        </p>
      </motion.div>

      {!showResults ? (
        <>
          {/* Stepper */}
          <ProcessStepper 
            steps={steps} 
            currentStep={currentStep} 
            progress={progress} 
          />
          
          {/* Step Content */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
          >
            {/* Step 1: Upload */}
            {currentStep === 1 && (
              <motion.div 
                variants={contentVariants}
                initial="hidden"
                animate="visible"
                className="p-6"
              >
                <motion.div variants={itemVariants}>
                  <h2 className="text-2xl font-bold mb-4 text-gray-800">
                    Téléchargez votre image
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Sélectionnez une image contenant du texte à extraire. Pour de meilleurs résultats, utilisez une image claire avec un bon contraste.
                  </p>
                </motion.div>
                
                <motion.div 
                  variants={itemVariants}
                  className="grid grid-cols-1 md:grid-cols-2 gap-8"
                >
                  <div>
                    <ImageUploader onImageSelected={handleImageSelected} />
                    
                    {error && (
                      <motion.div 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm"
                      >
                        <p className="flex items-center">
                          <span className="mr-2"><FontAwesomeIcon icon={faExclamationTriangle} /></span>
                          {error}
                        </p>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
                
                <motion.div 
                  variants={itemVariants} 
                  className="mt-8 flex justify-end"
                >
                  <motion.div
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    <Button
                      variant="primary"
                      icon={faArrowRight}
                      iconPosition="right"
                      disabled={!selectedImage}
                      onClick={handleNextStep}
                    >
                      Continuer
                    </Button>
                  </motion.div>
                </motion.div>
              </motion.div>
            )}
            
            {/* Step 2: Analysis */}
            {currentStep === 2 && (
              <motion.div 
                variants={contentVariants}
                initial="hidden"
                animate="visible"
                className="p-6"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <motion.div variants={itemVariants}>
                    <h2 className="text-2xl font-bold mb-4 text-gray-800">
                      Analyse de l'image
                    </h2>
                    <p className="text-gray-600 mb-6">
                      Nous analysons les caractéristiques de votre image pour déterminer le meilleur traitement à appliquer.
                    </p>
                    
                    <div className="flex items-center justify-center py-8">
                      <ClipLoader color="#6366f1" size={60} />
                    </div>
                    
                    <motion.div 
                      variants={itemVariants}
                      className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4"
                    >
                      <h3 className="text-blue-800 font-medium mb-2 flex items-center">
                        <i className="fas fa-info-circle mr-2"></i> Que se passe-t-il?
                      </h3>
                      <p className="text-blue-700 text-sm">
                        Nous évaluons la résolution, le contraste, le bruit et d'autres facteurs pour déterminer la meilleure approche de prétraitement.
                      </p>
                    </motion.div>
                  </motion.div>
                  
                  <motion.div 
                    variants={itemVariants}
                    className="flex items-center justify-center"
                  >
                    {selectedImage && (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                        className="bg-gray-50 rounded-xl border border-gray-200 p-4"
                      >
                        <h3 className="text-sm font-medium text-gray-700 mb-2">Image originale</h3>
                        <img 
                          src={URL.createObjectURL(selectedImage)} 
                          alt="Original" 
                          className="w-full h-auto object-contain rounded-lg max-h-48"
                        />
                      </motion.div>
                    )}
                  </motion.div>
                </div>
              </motion.div>
            )}
            
            {/* Step 3: Preprocessing */}
            {currentStep === 3 && (
              <motion.div 
                variants={contentVariants}
                initial="hidden"
                animate="visible"
                className="p-6"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <motion.div variants={itemVariants}>
                    <h2 className="text-2xl font-bold mb-4 text-gray-800">
                      Prétraitement avancé
                    </h2>
                    <p className="text-gray-600 mb-6">
                      Nous appliquons des techniques avancées pour améliorer la visibilité du texte.
                    </p>
                    
                    <div className="flex items-center justify-center py-8">
                      <ClipLoader color="#6366f1" size={60} />
                    </div>
                    
                    <motion.div 
                      variants={itemVariants}
                      className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 mb-4"
                    >
                      <h3 className="text-indigo-800 font-medium mb-2 flex items-center">
                        <i className="fas fa-magic mr-2"></i> Techniques appliquées
                      </h3>
                      <ul className="space-y-2 text-sm text-indigo-700">
                        <motion.li
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.5 }}
                        >• Correction de couleur et optimisation du contraste</motion.li>
                        <motion.li
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.7 }}
                        >• Réduction intelligente du bruit</motion.li>
                        <motion.li
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.9 }}
                        >• Binarisation adaptative</motion.li>
                        <motion.li
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 1.1 }}
                        >• Morphologie mathématique</motion.li>
                        <motion.li
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 1.3 }}
                        >• Détection de contours</motion.li>
                      </ul>
                    </motion.div>
                  </motion.div>
                  
                  <motion.div 
                    variants={itemVariants}
                    className="flex flex-col space-y-4"
                  >
                    {selectedImage && (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                        className="bg-gray-50 rounded-xl border border-gray-200 p-4"
                      >
                        <h3 className="text-sm font-medium text-gray-700 mb-2">Traitement en cours...</h3>
                        <div className="w-full h-40 flex items-center justify-center bg-gray-100 rounded-lg">
                          <div className="flex flex-col items-center">
                            <ClipLoader color="#6366f1" size={40} />
                            <p className="text-gray-400 text-sm mt-4">Prétraitement avancé</p>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                </div>
              </motion.div>
            )}
            
            {/* Step 4: OCR Results */}
            {currentStep === 4 && (
              <motion.div 
                variants={contentVariants}
                initial="hidden"
                animate="visible"
                className="p-6"
              >
                <motion.div variants={itemVariants}>
                  <h2 className="text-2xl font-bold mb-4 text-gray-800">
                    Résultats d'extraction OCR
                  </h2>
                  <p className="text-gray-600 mb-8">
                    Comparez les résultats d'extraction de texte avant et après prétraitement.
                  </p>
                </motion.div>
                
                {error ? (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 mb-8"
                  >
                    <h3 className="font-medium flex items-center mb-2">
                      <FontAwesomeIcon icon={faExclamationTriangle} className="mr-2" />
                      Erreur lors du traitement
                    </h3>
                    <p>{error}</p>
                    <div className="mt-4">
                      <Button
                        variant="outline"
                        icon={faArrowLeft}
                        iconPosition="left"
                        onClick={resetProcess}
                      >
                        Réessayer
                      </Button>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div 
                    variants={itemVariants}
                    className="grid grid-cols-1 md:grid-cols-2 gap-8"
                  >
                    <div className="flex flex-col h-full">
                      <h3 className="text-lg font-medium text-gray-800 mb-3">Image d'origine</h3>
                      <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 flex-grow mb-4">
                        {isProcessing ? (
                          <div className="w-full h-40 flex items-center justify-center">
                            <div className="flex flex-col items-center">
                              <ClipLoader color="#6366f1" size={30} />
                              <p className="text-gray-400 text-sm mt-4">Extraction en cours...</p>
                            </div>
                          </div>
                        ) : result ? (
                          <pre className="whitespace-pre-wrap text-gray-700 text-sm font-mono bg-gray-50 p-2 rounded-md overflow-auto max-h-64">
                            {result.degraded_text || "Aucun texte détecté"}
                          </pre>
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p className="text-gray-400">En attente de traitement...</p>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex flex-col h-full">
                      <h3 className="text-lg font-medium text-indigo-800 mb-3">Image prétraitée</h3>
                      <div className="bg-indigo-50 rounded-xl border border-indigo-200 p-4 flex-grow mb-4">
                        {isProcessing ? (
                          <div className="w-full h-40 flex items-center justify-center">
                            <div className="flex flex-col items-center">
                              <ClipLoader color="#6366f1" size={30} />
                              <p className="text-indigo-400 text-sm mt-4">Extraction en cours...</p>
                            </div>
                          </div>
                        ) : result ? (
                          <pre className="whitespace-pre-wrap text-indigo-700 text-sm font-mono bg-indigo-50 p-2 rounded-md overflow-auto max-h-64">
                            {result.preprocessed_text || "Aucun texte détecté"}
                          </pre>
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p className="text-indigo-400">En attente de traitement...</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}
                
                {!error && result && (
                  <motion.div 
                    variants={itemVariants}
                    className="mt-12 flex justify-between items-center"
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
                    
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Button
                        variant="primary"
                        icon={faChartLine}
                        iconPosition="right"
                        onClick={viewFullResults}
                      >
                        Voir les résultats complets
                      </Button>
                    </motion.div>
                  </motion.div>
                )}
              </motion.div>
            )}
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
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Image originale</h3>
                  <img 
                    src={result.original_image} 
                    alt="Original" 
                    className="w-full h-auto object-contain rounded-lg"
                  />
                </motion.div>
                <motion.div 
                  whileHover={{ y: -5 }}
                  className="bg-gray-50 rounded-xl border border-gray-200 p-4"
                >
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Image dégradée</h3>
                  <img 
                    src={result.degraded_image} 
                    alt="Degraded" 
                    className="w-full h-auto object-contain rounded-lg"
                  />
                </motion.div>
                <motion.div 
                  whileHover={{ y: -5 }}
                  className="bg-indigo-50 rounded-xl border border-indigo-200 p-4"
                >
                  <h3 className="text-sm font-medium text-indigo-700 mb-2">Image prétraitée</h3>
                  <img 
                    src={result.preprocessed_image} 
                    alt="Preprocessed" 
                    className="w-full h-auto object-contain rounded-lg"
                  />
                </motion.div>
              </motion.div>
              
              {/* Analysis Results */}
              {result.comparison && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                  className="bg-gray-50 rounded-xl p-6 mb-10"
                >
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">Analyse comparative</h3>
                  <motion.div 
                    whileHover={{ scale: 1.02 }}
                    className="flex items-center justify-center p-4 bg-white rounded-lg shadow-sm border border-gray-100"
                  >
                    <div className="text-lg text-indigo-700 font-medium flex items-center">
                      <FontAwesomeIcon icon={faCheckCircle} className="mr-3 text-green-500" />
                      {result.comparison}
                    </div>
                  </motion.div>
                </motion.div>
              )}
              
              {/* Text Results */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.8 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10"
              >
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">
                    Texte extrait (Image dégradée)
                  </h3>
                  <div className="relative">
                    <pre className="whitespace-pre-wrap text-gray-700 text-sm font-mono bg-gray-50 p-4 rounded-lg overflow-auto h-64 border border-gray-200">
                      {result.degraded_text || "Aucun texte détecté"}
                    </pre>
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-indigo-800 mb-3">
                    Texte extrait (Image prétraitée)
                  </h3>
                  <div className="relative">
                    <pre className="whitespace-pre-wrap text-indigo-700 text-sm font-mono bg-indigo-50 p-4 rounded-lg overflow-auto h-64 border border-indigo-200">
                      {result.preprocessed_text || "Aucun texte détecté"}
                    </pre>
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
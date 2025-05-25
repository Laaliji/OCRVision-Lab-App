import React, { useState, useRef, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload, faSpinner, faTrash, faMagic, faRedo, faCopy, faDownload, faCheck } from '@fortawesome/free-solid-svg-icons';
import { OCRService } from '../../services/api';
import { WordRecognitionResult, CharacterPrediction } from '../../types/types';
import { useLanguage } from '../../contexts/LanguageContext';
import { useLocation } from 'react-router-dom';

const TextRecognition: React.FC = () => {
  const { t } = useLanguage();
  const location = useLocation();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<WordRecognitionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Reset copy/download status after 2 seconds
  useEffect(() => {
    if (copied) {
      const timeout = setTimeout(() => setCopied(false), 2000);
      return () => clearTimeout(timeout);
    }
  }, [copied]);
  
  useEffect(() => {
    if (downloaded) {
      const timeout = setTimeout(() => setDownloaded(false), 2000);
      return () => clearTimeout(timeout);
    }
  }, [downloaded]);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      
      // Check file type
      const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff'];
      if (!validTypes.includes(selectedFile.type)) {
        setError('Type de fichier non supporté. Veuillez utiliser JPG, PNG, ou BMP.');
        return;
      }
      
      // Check file size (max 5MB)
      const maxSize = 5 * 1024 * 1024;
      if (selectedFile.size > maxSize) {
        setError('Fichier trop volumineux. La taille maximale est de 5MB.');
        return;
      }
      
      setFile(selectedFile);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };
  
  const handleRemoveFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError(t('error.selectImage'));
      return;
    }
    
    setIsProcessing(true);
    setError(null);
    
    try {
      // Always use 'text' mode for better readability and spacing
      const result = await OCRService.processWordRecognition(file, 'text');
      setResult(result);
    } catch (err) {
      console.error('Error processing image:', err);
      setError(t('error.processing'));
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleReset = () => {
    setResult(null);
    handleRemoveFile();
  };
  
  const formatConfidence = (confidence: number): string => {
    return (confidence * 100).toFixed(2) + '%';
  };
  
  const handleCopyToClipboard = () => {
    if (!result) return;
    
    try {
      navigator.clipboard.writeText(result.word_recognition.best_word);
      setCopied(true);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };
  
  const handleDownloadText = () => {
    if (!result) return;
    
    try {
      const text = result.word_recognition.best_word;
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'recognized-text.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setDownloaded(true);
    } catch (err) {
      console.error('Failed to download text:', err);
    }
  };
  
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">{t('textRecognition.title')}</h1>
        <p className="text-gray-600 mt-2">
          {t('textRecognition.subtitle')}
        </p>
      </div>
      
      {!result ? (
        <div className="bg-white rounded-xl shadow-sm p-6">
          <form onSubmit={handleSubmit}>
            {/* File Upload */}
            <div className="mb-6">
              <h2 className="text-lg font-medium text-gray-800 mb-3">{t('textRecognition.uploadTitle')}</h2>
              
              {!preview ? (
                <div
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-300 transition"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    onChange={handleFileChange}
                    accept=".jpg,.jpeg,.png,.bmp,.tiff"
                  />
                  <FontAwesomeIcon icon={faUpload} className="text-gray-400 text-3xl mb-3" />
                  <p className="text-gray-600 mb-1">{t('textRecognition.uploadInstruction')}</p>
                  <p className="text-gray-400 text-sm">{t('textRecognition.uploadFormats')}</p>
                </div>
              ) : (
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="mb-3">
                    <img
                      src={preview}
                      alt="Preview"
                      className="max-h-64 mx-auto object-contain rounded"
                    />
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="text-sm text-gray-600 truncate">
                      {file?.name} ({file?.size ? (file.size / 1024).toFixed(1) : '0'} KB)
                    </div>
                    <button
                      type="button"
                      className="text-red-500 hover:text-red-700"
                      onClick={handleRemoveFile}
                    >
                      <FontAwesomeIcon icon={faTrash} />
                    </button>
                  </div>
                </div>
              )}
              
              {error && (
                <div className="mt-3 text-red-500 text-sm">{error}</div>
              )}
            </div>
            
            {/* Submit Button */}
            <div className="flex justify-center">
              <button
                type="submit"
                disabled={!file || isProcessing}
                className={`bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center w-full md:w-auto ${
                  !file || isProcessing ? 'opacity-50 cursor-not-allowed' : 'hover:bg-indigo-700'
                }`}
              >
                {isProcessing ? (
                  <>
                    <FontAwesomeIcon icon={faSpinner} spin className="mr-2" />
                    {t('textRecognition.processing')}
                  </>
                ) : (
                  <>
                    <FontAwesomeIcon icon={faMagic} className="mr-2" />
                    {t('textRecognition.process')}
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm p-6">
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">{t('textRecognition.resultsTitle')}</h2>
            <p className="text-gray-600">
              {t('textRecognition.resultsSubtitle')}
            </p>
          </div>
          
          {/* Recognition Result */}
          <div className="bg-indigo-50 rounded-xl p-6 mb-8">
            <h3 className="text-xl font-semibold text-indigo-800 mb-4">{t('textRecognition.recognizedText')}</h3>
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <h2 className="text-3xl font-bold text-center text-indigo-700">
                {result.word_recognition.best_word || t('textRecognition.noTextDetected')}
              </h2>
              <div className="flex justify-center mt-4">
                <div className="px-3 py-1 bg-indigo-100 rounded-full">
                  <span className="text-sm text-indigo-800">
                    {t('textRecognition.confidence')} {formatConfidence(result.word_recognition.preprocessed_confidence)}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Add copy/download actions */}
            <div className="mt-4 flex flex-wrap justify-center gap-3">
              <button
                onClick={handleCopyToClipboard}
                className="flex items-center px-4 py-2 bg-white border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition"
              >
                <FontAwesomeIcon 
                  icon={copied ? faCheck : faCopy} 
                  className={`mr-2 ${copied ? 'text-green-600' : 'text-gray-500'}`} 
                />
                {copied ? t('results.copied') : t('results.copy')}
              </button>
              
              <button
                onClick={handleDownloadText}
                className="flex items-center px-4 py-2 bg-white border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition"
              >
                <FontAwesomeIcon 
                  icon={downloaded ? faCheck : faDownload} 
                  className={`mr-2 ${downloaded ? 'text-green-600' : 'text-gray-500'}`} 
                />
                {downloaded ? t('results.downloaded') : t('results.download')}
              </button>
            </div>
          </div>
          
          {/* Visualization */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3">{t('textRecognition.originalImage')}</h3>
              <img
                src={result.original_image}
                alt="Original"
                className="w-full h-auto object-contain rounded-lg border border-gray-200 max-h-64"
              />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-indigo-800 mb-3">{t('textRecognition.characterVisualization')}</h3>
              <img
                src={result.visualization}
                alt="Visualization"
                className="w-full h-auto object-contain rounded-lg border border-indigo-200 max-h-64"
              />
              <p className="text-gray-600 text-sm mt-2">
                {t('textRecognition.visualizationDesc')}
              </p>
            </div>
          </div>
          
          {/* Character Recognition Process */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">{t('textRecognition.characterRecognitionProcess')}</h3>
            <div className="bg-indigo-50 p-4 rounded-lg">
              <ol className="list-decimal ml-6 space-y-2 text-gray-700">
                <li>
                  <span className="font-medium">{t('textRecognition.process.step1.title')}</span> {t('textRecognition.process.step1.desc')}
                </li>
                <li>
                  <span className="font-medium">{t('textRecognition.process.step2.title')}</span> {t('textRecognition.process.step2.desc')}
                </li>
                <li>
                  <span className="font-medium">{t('textRecognition.process.step3.title')}</span>
                  <ul className="list-disc ml-6 mt-1 text-gray-600">
                    <li>{t('textRecognition.process.step3.item1')}</li>
                    <li>{t('textRecognition.process.step3.item2')}</li>
                    <li>{t('textRecognition.process.step3.item3')}</li>
                    <li>{t('textRecognition.process.step3.item4')}</li>
                  </ul>
                </li>
                <li>
                  <span className="font-medium">{t('textRecognition.process.step4.title')}</span> {t('textRecognition.process.step4.desc')}
                </li>
                <li>
                  <span className="font-medium">{t('textRecognition.process.step5.title')}</span> {t('textRecognition.process.step5.desc')}
                </li>
                <li>
                  <span className="font-medium">{t('textRecognition.process.step6.title')}</span> {t('textRecognition.process.step6.desc')}
                </li>
              </ol>
            </div>
          </div>
          
          {/* Method Comparison */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">{t('textRecognition.methodComparison')}</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white rounded-lg overflow-hidden border border-gray-200">
                <thead className="bg-gray-50 text-gray-700">
                  <tr>
                    <th className="py-2 px-4 text-left">{t('textRecognition.method')}</th>
                    <th className="py-2 px-4 text-left">{t('textRecognition.recognizedTextLabel')}</th>
                    <th className="py-2 px-4 text-left">{t('results.confidence')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  <tr>
                    <td className="py-2 px-4">{t('textRecognition.directOriginal')}</td>
                    <td className="py-2 px-4 font-mono">
                      {result.word_recognition.original_word || t('textRecognition.noTextDetected')}
                    </td>
                    <td className="py-2 px-4">
                      {formatConfidence(result.word_recognition.original_confidence)}
                    </td>
                  </tr>
                  
                  <tr className="bg-indigo-50">
                    <td className="py-2 px-4 font-medium text-indigo-800">
                      {t('textRecognition.directPreprocessed')}
                    </td>
                    <td className="py-2 px-4 font-mono font-medium text-indigo-800">
                      {result.word_recognition.preprocessed_word || t('textRecognition.noTextDetected')}
                    </td>
                    <td className="py-2 px-4 font-medium text-indigo-800">
                      {formatConfidence(result.word_recognition.preprocessed_confidence)}
                    </td>
                  </tr>
                  
                  <tr>
                    <td className="py-2 px-4">{t('textRecognition.characterByCharacter')}</td>
                    <td className="py-2 px-4 font-mono">
                      {result.comparison.character_by_character || t('textRecognition.noTextDetected')}
                    </td>
                    <td className="py-2 px-4">-</td>
                  </tr>
                  
                  <tr>
                    <td className="py-2 px-4">{t('textRecognition.wordSegmentation')}</td>
                    <td className="py-2 px-4 font-mono">
                      {result.comparison.word_segmentation || t('textRecognition.noTextDetected')}
                    </td>
                    <td className="py-2 px-4">-</td>
                  </tr>
                  
                  <tr>
                    <td className="py-2 px-4">{t('textRecognition.tesseractOCR')}</td>
                    <td className="py-2 px-4 font-mono">
                      {result.comparison.tesseract || t('textRecognition.noTextDetected')}
                    </td>
                    <td className="py-2 px-4">-</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Reset Button */}
          <div className="flex justify-center">
            <button
              type="button"
              onClick={handleReset}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium flex items-center hover:bg-indigo-700"
            >
              <FontAwesomeIcon icon={faRedo} className="mr-2" />
              {t('textRecognition.tryAnother')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TextRecognition; 
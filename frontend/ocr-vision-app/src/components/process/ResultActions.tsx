import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCopy, faDownload, faCheck, faExclamationTriangle } from '@fortawesome/free-solid-svg-icons';
import { useLanguage } from '../../contexts/LanguageContext';
import { OCRResult } from '../../types/types';

interface ResultActionsProps {
  result: OCRResult | null;
}

const ResultActions: React.FC<ResultActionsProps> = ({ result }) => {
  const { t } = useLanguage();
  const [copySuccess, setCopySuccess] = useState<boolean | null>(null);
  const [downloadSuccess, setDownloadSuccess] = useState<boolean | null>(null);
  
  if (!result || !result.model_results) {
    return null;
  }
  
  const extractedText = result.model_results.degraded_text || '';
  const confidence = result.model_results.degraded_confidence || 0;
  
  // Function to copy text to clipboard
  const copyToClipboard = async () => {
    try {
      const textToCopy = `
OCR Results:
-----------
Extracted Text: ${extractedText}
Confidence: ${confidence.toFixed(2)}%
Date: ${new Date().toLocaleString()}
      `;
      
      await navigator.clipboard.writeText(textToCopy.trim());
      setCopySuccess(true);
      
      // Reset success message after 2 seconds
      setTimeout(() => {
        setCopySuccess(null);
      }, 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      setCopySuccess(false);
      
      // Reset error message after 3 seconds
      setTimeout(() => {
        setCopySuccess(null);
      }, 3000);
    }
  };
  
  // Function to download results as a text file
  const downloadResults = () => {
    try {
      const textContent = `
OCR Vision App - Results
=======================
Date: ${new Date().toLocaleString()}

Extracted Text:
--------------
${extractedText}

Confidence: ${confidence.toFixed(2)}%

Additional Information:
---------------------
Image was processed using controlled degradation techniques for optimal OCR performance.
      `;
      
      const blob = new Blob([textContent.trim()], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = `ocr-results-${new Date().toISOString().split('T')[0]}.txt`;
      document.body.appendChild(a);
      a.click();
      
      // Clean up
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setDownloadSuccess(true);
      
      // Reset success message after 2 seconds
      setTimeout(() => {
        setDownloadSuccess(null);
      }, 2000);
    } catch (err) {
      console.error('Failed to download results: ', err);
      setDownloadSuccess(false);
      
      // Reset error message after 3 seconds
      setTimeout(() => {
        setDownloadSuccess(null);
      }, 3000);
    }
  };
  
  return (
    <div className="flex flex-col space-y-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
      <h3 className="font-medium text-gray-700 mb-2">{t('results.actions')}</h3>
      
      <div className="flex flex-wrap gap-4">
        <button
          onClick={copyToClipboard}
          className="flex items-center px-4 py-2 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200 transition-colors"
        >
          <FontAwesomeIcon icon={copySuccess === true ? faCheck : (copySuccess === false ? faExclamationTriangle : faCopy)} className="mr-2" />
          {copySuccess === true 
            ? t('results.copied') 
            : (copySuccess === false ? t('results.copyFailed') : t('results.copy'))}
        </button>
        
        <button
          onClick={downloadResults}
          className="flex items-center px-4 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors"
        >
          <FontAwesomeIcon icon={downloadSuccess === true ? faCheck : (downloadSuccess === false ? faExclamationTriangle : faDownload)} className="mr-2" />
          {downloadSuccess === true 
            ? t('results.downloaded') 
            : (downloadSuccess === false ? t('results.downloadFailed') : t('results.download'))}
        </button>
      </div>
      
      {/* Preview box */}
      <div className="mt-2">
        <div className="text-xs text-gray-500 mb-1">{t('results.preview')}:</div>
        <div className="bg-white p-3 border rounded-md text-sm font-mono whitespace-pre-wrap max-h-32 overflow-y-auto">
          {extractedText || t('results.noText')}
        </div>
      </div>
    </div>
  );
};

export default ResultActions; 
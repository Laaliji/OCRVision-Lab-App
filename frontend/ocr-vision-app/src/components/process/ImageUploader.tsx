import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faTrashAlt, faImage } from '@fortawesome/free-solid-svg-icons';
import { motion } from 'framer-motion';
import Button from '../common/Button';

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected }) => {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };
  
  const handleButtonClick = () => {
    // Déclencher directement le clic sur l'input file
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  const processFile = (file: File) => {
    setError(null);
    
    // Vérifier le type de fichier
    if (!file.type.match('image.*')) {
      setError('Veuillez sélectionner une image valide');
      return;
    }
    
    // Vérifier la taille du fichier (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('L\'image est trop volumineuse (max 5MB)');
      return;
    }
    
    setImage(file);
    onImageSelected(file);
    
    // Créer un aperçu
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };
  
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };
  
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };
  
  const handleRemoveImage = () => {
    setImage(null);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  return (
    <div className="w-full">
      {!preview ? (
        <>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="w-full"
          >
            <div 
              className={`w-full h-64 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 transition-colors
                ${isDragging ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 bg-gray-50 hover:bg-gray-100'}
                ${error ? 'border-red-500 bg-red-50' : ''}
              `}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={handleButtonClick}
              style={{ cursor: 'pointer' }}
            >
              <motion.div 
                className="w-20 h-20 mb-4 flex items-center justify-center bg-indigo-100 rounded-full text-indigo-500"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <FontAwesomeIcon icon={faCloudUploadAlt} size="2x" />
              </motion.div>
              
              <h3 className="text-lg font-medium text-gray-700 mb-2">
                Déposez votre image ici
              </h3>
              <p className="text-sm text-gray-500 mb-4 text-center">
                ou cliquez pour parcourir vos fichiers<br />
                (PNG, JPG, JPEG, TIFF, BMP)
              </p>
              
              {error && (
                <p className="text-red-500 text-sm mb-2">{error}</p>
              )}
              
              <input 
                type="file" 
                ref={fileInputRef}
                className="hidden" 
                accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp"
                onChange={handleImageChange}
              />
              
              <motion.button
                type="button"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-indigo-600 text-white px-4 py-2 rounded-full text-sm font-medium flex items-center"
                onClick={(e) => {
                  e.stopPropagation(); // Empêcher la propagation pour éviter un double clic
                  handleButtonClick();
                }}
              >
                <FontAwesomeIcon icon={faImage} className="mr-2" />
                Parcourir
              </motion.button>
            </div>
          </motion.div>
        </>
      ) : (
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full bg-gray-50 rounded-xl border border-gray-200 p-4 shadow-sm"
        >
          <div className="relative mb-4">
            <img 
              src={preview} 
              alt="Aperçu" 
              className="w-full h-auto max-h-64 object-contain rounded-lg"
            />
            <div className="absolute top-2 right-2">
              <motion.button 
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleRemoveImage}
                className="bg-white p-2 rounded-full shadow-md text-red-500 hover:text-red-700 transition"
                type="button"
              >
                <FontAwesomeIcon icon={faTrashAlt} />
              </motion.button>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-800 text-sm truncate">
                {image?.name}
              </h3>
              <p className="text-gray-500 text-xs">
                {image && formatFileSize(image.size)}
              </p>
            </div>
            <motion.div
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="bg-green-100 text-green-700 text-xs py-1 px-2 rounded-full"
            >
              Prêt pour le traitement
            </motion.div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ImageUploader; 
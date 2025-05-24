import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEye } from '@fortawesome/free-solid-svg-icons';
import { useLanguage } from '../../contexts/LanguageContext';

const Footer: React.FC = () => {
  const { t } = useLanguage();
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-gray-800 text-white py-8 mt-20">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center space-x-2">
              <FontAwesomeIcon icon={faEye} className="text-indigo-400 text-xl" />
              <span className="text-xl font-bold">OCR<span className="text-indigo-400">Vision</span></span>
            </div>
            <p className="text-gray-400 text-sm mt-2">
              {t('footer.copyright').replace('{year}', currentYear.toString())}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">{t('footer.developed')}</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 
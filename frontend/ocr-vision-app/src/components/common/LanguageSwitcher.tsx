import React from 'react';
import { useLanguage } from '../../contexts/LanguageContext';

const LanguageSwitcher: React.FC = () => {
  const { language, setLanguage } = useLanguage();

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'fr' : 'en');
  };

  return (
    <button
      onClick={toggleLanguage}
      className="flex items-center justify-center px-3 py-1 text-sm font-medium text-gray-700 bg-gray-100 border border-gray-300 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
      aria-label={language === 'en' ? 'Switch to French' : 'Switch to English'}
    >
      <span className="mr-1 font-bold">
        {language === 'en' ? 'FR' : 'EN'}
      </span>
      <span className="text-xs text-gray-500">
        {language === 'en' ? '🇫🇷' : '🇬🇧'}
      </span>
    </button>
  );
};

export default LanguageSwitcher; 
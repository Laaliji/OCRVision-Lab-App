import React from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEye } from '@fortawesome/free-solid-svg-icons';
import LanguageSwitcher from '../common/LanguageSwitcher';
import { useLanguage } from '../../contexts/LanguageContext';

const Header: React.FC = () => {
  const { t } = useLanguage();
  
  const handleHashLinkClick = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    
    // If we're already on the home page
    if (window.location.pathname === '/' || window.location.pathname === '') {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      // If we're on another page, navigate to home with hash
      window.location.href = `/#${id}`;
    }
  };

  return (
    <header className="bg-white py-4 shadow-sm">
      <div className="container mx-auto px-4 flex justify-between items-center">
        <Link to="/" className="flex items-center space-x-2">
          <FontAwesomeIcon icon={faEye} className="text-indigo-600 text-2xl" />
          <span className="text-2xl font-bold text-gray-800">OCR<span className="text-indigo-600">Vision</span></span>
        </Link>
        <div className="flex items-center">
          <nav className="hidden md:flex space-x-8 mr-6">
            <Link to="/" className="text-gray-600 hover:text-indigo-600 transition">
              {t('header.home')}
            </Link>
            <a 
              href="/#features" 
              className="text-gray-600 hover:text-indigo-600 transition"
              onClick={(e) => handleHashLinkClick(e, 'features')}
            >
              {t('header.features')}
            </a>
            <a 
              href="/#how-it-works" 
              className="text-gray-600 hover:text-indigo-600 transition"
              onClick={(e) => handleHashLinkClick(e, 'how-it-works')}
            >
              {t('header.howItWorks')}
            </a>
            <Link to="/process" className="text-gray-600 hover:text-indigo-600 transition">
              {t('header.process')}
            </Link>
          </nav>
          <LanguageSwitcher />
        </div>
      </div>
    </header>
  );
};

export default Header; 
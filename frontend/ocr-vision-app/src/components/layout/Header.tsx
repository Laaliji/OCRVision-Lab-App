import React from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEye } from '@fortawesome/free-solid-svg-icons';

const Header: React.FC = () => {
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
        <nav className="hidden md:flex space-x-8">
          <Link to="/" className="text-gray-600 hover:text-indigo-600 transition">Accueil</Link>
          <a 
            href="/#features" 
            className="text-gray-600 hover:text-indigo-600 transition"
            onClick={(e) => handleHashLinkClick(e, 'features')}
          >
            Fonctionnalités
          </a>
          <a 
            href="/#how-it-works" 
            className="text-gray-600 hover:text-indigo-600 transition"
            onClick={(e) => handleHashLinkClick(e, 'how-it-works')}
          >
            Comment ça marche
          </a>
          <Link to="/process" className="text-gray-600 hover:text-indigo-600 transition">Traitement</Link>
        </nav>
      </div>
    </header>
  );
};

export default Header; 
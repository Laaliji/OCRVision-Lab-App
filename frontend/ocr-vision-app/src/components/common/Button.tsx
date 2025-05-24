import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { IconDefinition } from '@fortawesome/fontawesome-svg-core';
import { faSpinner } from '@fortawesome/free-solid-svg-icons';

interface ButtonProps {
  children: React.ReactNode;
  type?: 'button' | 'submit' | 'reset';
  variant?: 'primary' | 'secondary' | 'outline' | 'success';
  size?: 'sm' | 'md' | 'lg';
  icon?: IconDefinition;
  iconPosition?: 'left' | 'right';
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  className?: string;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void | Promise<void>;
}

const Button: React.FC<ButtonProps> = ({
  children,
  type = 'button',
  variant = 'primary',
  size = 'md',
  icon,
  iconPosition = 'left',
  disabled = false,
  loading = false,
  fullWidth = false,
  className = '',
  onClick,
}) => {
  // Classes de base pour tous les boutons
  const baseClasses = 'font-medium rounded-full transition inline-flex items-center justify-center';
  
  // Classes spécifiques à la variante
  const variantClasses = {
    primary: 'btn-primary text-white',
    secondary: 'bg-gray-100 text-gray-800 hover:bg-gray-200',
    outline: 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50',
    success: 'bg-green-500 text-white hover:bg-green-600',
  };
  
  // Classes spécifiques à la taille
  const sizeClasses = {
    sm: 'py-2 px-4 text-sm',
    md: 'py-3 px-6 text-base',
    lg: 'py-4 px-8 text-lg',
  };
  
  // Classes pour le bouton désactivé ou en chargement
  const disabledClasses = (disabled || loading) ? 'opacity-70 cursor-not-allowed' : '';
  
  // Classes pour la largeur du bouton
  const widthClasses = fullWidth ? 'w-full' : '';
  
  // Construction des classes finales
  const buttonClasses = `
    ${baseClasses}
    ${variantClasses[variant]}
    ${sizeClasses[size]}
    ${disabledClasses}
    ${widthClasses}
    ${className}
  `;
  
  // Déterminer quelle icône afficher
  const displayIcon = loading ? faSpinner : icon;
  const iconClasses = loading ? 'animate-spin' : '';
  
  return (
    <button
      type={type}
      className={buttonClasses}
      disabled={disabled || loading}
      onClick={onClick}
    >
      {displayIcon && iconPosition === 'left' && (
        <FontAwesomeIcon icon={displayIcon} className={`mr-2 ${iconClasses}`} />
      )}
      {children}
      {displayIcon && iconPosition === 'right' && (
        <FontAwesomeIcon icon={displayIcon} className={`ml-2 ${iconClasses}`} />
      )}
    </button>
  );
};

export default Button; 
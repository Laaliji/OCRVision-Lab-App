import React, { useState, useEffect } from 'react';

interface TypingEffectProps {
  texts: string[];
  typingSpeed?: number;
  eraseSpeed?: number;
  typingDelay?: number;
  eraseDelay?: number;
  className?: string;
  repeatsPerText?: number;
}

const TypingEffect: React.FC<TypingEffectProps> = ({
  texts,
  typingSpeed = 100,
  eraseSpeed = 80,
  typingDelay = 1500,
  eraseDelay = 2000,
  className = '',
  repeatsPerText = 2,
}) => {
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [currentText, setCurrentText] = useState('');
  const [isTyping, setIsTyping] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [currentRepeat, setCurrentRepeat] = useState(0);
  
  // Reset state when texts array changes (i.e., language changes)
  useEffect(() => {
    setCurrentTextIndex(0);
    setCurrentText('');
    setIsTyping(true);
    setIsPaused(false);
    setCurrentRepeat(0);
  }, [texts]); // Add texts as a dependency to detect language changes

  // Effet pour gérer l'animation de frappe
  useEffect(() => {
    if (!texts || texts.length === 0) return;

    let timeout: NodeJS.Timeout;
    const textToType = texts[currentTextIndex];

    // Si on est en train de taper et pas en pause
    if (isTyping && !isPaused) {
      if (currentText.length < textToType.length) {
        // Continue de taper le texte caractère par caractère
        timeout = setTimeout(() => {
          setCurrentText(textToType.substring(0, currentText.length + 1));
        }, typingSpeed);
      } else {
        // Fini de taper, attendre avant d'effacer
        setIsPaused(true);
        timeout = setTimeout(() => {
          setIsPaused(false);
          setIsTyping(false);
        }, eraseDelay);
      }
    } else if (!isTyping && !isPaused) {
      // En mode effacement et pas en pause
      if (currentText.length > 0) {
        // Continue d'effacer caractère par caractère
        timeout = setTimeout(() => {
          setCurrentText(currentText.substring(0, currentText.length - 1));
        }, eraseSpeed);
      } else {
        // Complètement effacé, attendre avant de taper le prochain texte
        setIsPaused(true);
        timeout = setTimeout(() => {
          setIsPaused(false);
          setIsTyping(true);
          
          // Check if we need to repeat the current text or move to the next one
          if (currentRepeat < repeatsPerText - 1) {
            setCurrentRepeat(currentRepeat + 1);
          } else {
            setCurrentRepeat(0);
            setCurrentTextIndex((currentTextIndex + 1) % texts.length);
          }
        }, typingDelay);
      }
    }

    return () => clearTimeout(timeout);
  }, [currentText, currentTextIndex, currentRepeat, eraseDelay, eraseSpeed, isTyping, isPaused, repeatsPerText, texts, typingDelay, typingSpeed]);

  return (
    <div className={`h-24 flex items-center ${className}`}>
      <span>
        {currentText}<span className="text-indigo-300 animate-pulse">_</span>
      </span>
    </div>
  );
};

export default TypingEffect; 
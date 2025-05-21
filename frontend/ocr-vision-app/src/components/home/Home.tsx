import React from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faImage, faFont, faChartBar, faMagic } from '@fortawesome/free-solid-svg-icons';
import { motion } from 'framer-motion';
import TypingEffect from '../common/TypingEffect';
import Button from '../common/Button';

const Home: React.FC = () => {
  // Variantes d'animation pour les éléments
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        staggerChildren: 0.3
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { 
        type: 'spring',
        stiffness: 100
      }
    }
  };

  return (
    <>
      {/* Hero Section */}
      <section className="hero-gradient text-white py-20 relative overflow-hidden">
        {/* Decorative Elements */}
        <motion.div 
          animate={{ 
            x: [0, 40, -40, 0],
            y: [0, -40, 40, 0],
            scale: [1, 1.2, 0.9, 1]
          }}
          transition={{ 
            repeat: Infinity,
            duration: 15,
            ease: "easeInOut"
          }}
          className="absolute top-0 -left-4 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70"
        />
        <motion.div 
          animate={{ 
            x: [0, -35, 35, 0],
            y: [0, 35, -35, 0],
            scale: [1, 0.8, 1.1, 1]
          }}
          transition={{ 
            repeat: Infinity,
            duration: 18,
            ease: "easeInOut"
          }}
          className="absolute top-0 -right-4 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70"
        />
        <motion.div 
          animate={{ 
            x: [0, 50, -50, 0],
            y: [0, -25, 25, 0],
            scale: [1, 1.3, 0.8, 1]
          }}
          transition={{ 
            repeat: Infinity,
            duration: 20,
            ease: "easeInOut"
          }}
          className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70"
        />
        
        <div className="container mx-auto px-4 relative z-10">
          <div className="flex flex-col md:flex-row items-center">
            <motion.div 
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="md:w-1/2 mb-10 md:mb-0"
            >
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
                <TypingEffect
                  texts={["Extraction intelligente de texte", "Reconnaissance avancée de documents", "Solution OCR de haute précision", "Traitement d'images optimisé"]}
                  typingSpeed={80}
                  eraseSpeed={10}
                  typingDelay={100}
                  eraseDelay={100}
                  repeatsPerText={3}
                />
              </h1>
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.8 }}
                className="text-xl opacity-90 mb-8 max-w-lg"
              >
                OCRVision utilise des techniques avancées de traitement d'image et l'IA pour extraire du texte même à partir d'images à faible résolution.
              </motion.p>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link to="/process">
                  <Button
                    variant="primary"
                    size="lg"
                    icon={faArrowRight}
                    iconPosition="right"
                  >
                    Commencer maintenant
                  </Button>
                </Link>
              </motion.div>
            </motion.div>
            <motion.div 
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="md:w-1/2 flex justify-center"
            >
              <div className="relative">
                <motion.div 
                  animate={{ rotate: [0, 5, 0, -5, 0] }}
                  transition={{ repeat: Infinity, duration: 10, ease: "easeInOut" }}
                  className="w-64 h-64 bg-indigo-100 absolute top-4 -right-4 rounded-lg transform rotate-6"
                />
                <motion.div 
                  animate={{ rotate: [0, -5, 0, 5, 0] }}
                  transition={{ repeat: Infinity, duration: 8, ease: "easeInOut" }}
                  className="w-64 h-64 bg-purple-100 absolute -top-4 -left-4 rounded-lg transform -rotate-6"
                />
                <motion.div 
                  whileHover={{ scale: 1.05 }}
                  className="glass-card rounded-lg p-4 relative z-10 animate-float"
                >
                  <img src="https://images.unsplash.com/photo-1595500381751-d838bbf46553?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80" alt="OCR example" className="rounded shadow-lg" />
                  <motion.div 
                    initial={{ opacity: 0, x: 20, y: 20 }}
                    animate={{ opacity: 1, x: 0, y: 0 }}
                    transition={{ delay: 1, duration: 0.5 }}
                    className="absolute -bottom-4 -right-4 bg-white rounded-lg p-3 shadow-lg"
                  >
                    <i className="fas fa-check-circle text-green-500 mr-2"></i>
                    <span className="text-sm font-medium">Texte extrait avec succès</span>
                  </motion.div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-3xl font-bold text-center mb-12"
          >
            Fonctionnalités principales
          </motion.h2>
          
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {/* Feature 1 */}
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -5 }}
              className="bg-gray-50 rounded-xl p-6 shadow-sm hover:shadow-md transition"
            >
              <div className="feature-icon w-14 h-14 flex items-center justify-center bg-indigo-100 text-indigo-600 text-xl mb-5">
                <FontAwesomeIcon icon={faImage} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Traitement d'images avancé</h3>
              <p className="text-gray-600">
                Conversion en niveaux de gris, seuillage adaptatif, et débruitage pour optimiser la qualité de l'image avant extraction.
              </p>
            </motion.div>
            
            {/* Feature 2 */}
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -5 }}
              className="bg-gray-50 rounded-xl p-6 shadow-sm hover:shadow-md transition"
            >
              <div className="feature-icon w-14 h-14 flex items-center justify-center bg-purple-100 text-purple-600 text-xl mb-5">
                <FontAwesomeIcon icon={faFont} />
              </div>
              <h3 className="text-xl font-semibold mb-3">OCR haute précision</h3>
              <p className="text-gray-600">
                Utilisation de Tesseract OCR pour une reconnaissance de texte fiable, même sur des documents de qualité médiocre.
              </p>
            </motion.div>
            
            {/* Feature 3 */}
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -5 }}
              className="bg-gray-50 rounded-xl p-6 shadow-sm hover:shadow-md transition"
            >
              <div className="feature-icon w-14 h-14 flex items-center justify-center bg-blue-100 text-blue-600 text-xl mb-5">
                <FontAwesomeIcon icon={faChartBar} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Analyse comparative</h3>
              <p className="text-gray-600">
                Visualisez la différence entre l'extraction directe et l'extraction avec prétraitement pour mesurer l'amélioration.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 bg-gray-50">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl font-bold text-center mb-8">Comment ça marche</h2>
            <p className="text-gray-600 text-center max-w-3xl mx-auto mb-12">
              Notre application utilise des techniques avancées de traitement d'images et de vision par ordinateur pour optimiser la reconnaissance de texte dans des images de faible qualité.
            </p>
          </motion.div>
          
          {/* Steps */}
          <div className="flex flex-col space-y-8 mt-12">
            {/* Step 1 */}
            <motion.div 
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="bg-white rounded-xl shadow-sm overflow-hidden border border-gray-100"
            >
              <div className="flex flex-col md:flex-row">
                <div className="bg-indigo-600 p-6 text-white md:w-1/4 flex flex-col justify-center items-center">
                  <motion.div 
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                    className="w-16 h-16 rounded-full bg-white text-indigo-600 flex items-center justify-center text-2xl font-bold mb-4"
                  >
                    1
                  </motion.div>
                  <h3 className="text-xl font-semibold text-center">Acquisition d'image</h3>
                </div>
                <div className="p-6 md:w-3/4">
                  <div className="mb-4">
                    <h4 className="text-lg font-medium mb-2 text-gray-800">Téléchargement et analyse</h4>
                    <p className="text-gray-600">L'utilisateur télécharge une image contenant du texte. Le système analyse les propriétés de l'image : résolution, profondeur de couleur, et rapport signal/bruit.</p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Step 2 */}
            <motion.div 
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white rounded-xl shadow-sm overflow-hidden border border-gray-100"
            >
              <div className="flex flex-col md:flex-row">
                <div className="bg-indigo-600 p-6 text-white md:w-1/4 flex flex-col justify-center items-center">
                  <motion.div 
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                    className="w-16 h-16 rounded-full bg-white text-indigo-600 flex items-center justify-center text-2xl font-bold mb-4"
                  >
                    2
                  </motion.div>
                  <h3 className="text-xl font-semibold text-center">Prétraitement avancé</h3>
                </div>
                <div className="p-6 md:w-3/4">
                  <div className="mb-4">
                    <h4 className="text-lg font-medium mb-2 text-gray-800">Amélioration de l'image</h4>
                    <p className="text-gray-600">Application de techniques avancées comme l'égalisation adaptative d'histogramme, la réduction de bruit et le seuillage adaptatif pour améliorer la visibilité du texte.</p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Step 3 */}
            <motion.div 
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-white rounded-xl shadow-sm overflow-hidden border border-gray-100"
            >
              <div className="flex flex-col md:flex-row">
                <div className="bg-indigo-600 p-6 text-white md:w-1/4 flex flex-col justify-center items-center">
                  <motion.div 
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                    className="w-16 h-16 rounded-full bg-white text-indigo-600 flex items-center justify-center text-2xl font-bold mb-4"
                  >
                    3
                  </motion.div>
                  <h3 className="text-xl font-semibold text-center">Extraction OCR</h3>
                </div>
                <div className="p-6 md:w-3/4">
                  <div className="mb-4">
                    <h4 className="text-lg font-medium mb-2 text-gray-800">Reconnaissance de texte</h4>
                    <p className="text-gray-600">Le moteur OCR analyse l'image prétraitée pour identifier et extraire le texte, avec comparaison des résultats avant et après prétraitement.</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="text-center mt-12"
          >
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link to="/process">
                <Button
                  variant="primary"
                  size="lg"
                  icon={faMagic}
                  iconPosition="right"
                >
                  Essayer maintenant
                </Button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </>
  );
};

export default Home; 
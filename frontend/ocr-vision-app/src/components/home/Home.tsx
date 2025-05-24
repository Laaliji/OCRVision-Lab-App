import React, { useRef } from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faImage, faFont, faChartBar, faMagic, faBolt, faClockRotateLeft } from '@fortawesome/free-solid-svg-icons';
import { motion } from 'framer-motion';
import TypingEffect from '../common/TypingEffect';
import Button from '../common/Button';
import { useLanguage } from '../../contexts/LanguageContext';

const Home: React.FC = () => {
  const { t } = useLanguage();
  const methodsRef = useRef<HTMLElement>(null);
  
  // Scroll to methods section when "Get Started Now" is clicked
  const scrollToMethods = () => {
    methodsRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
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
                  texts={[
                    t('home.hero.title1'),
                    t('home.hero.title2'),
                    t('home.hero.title3'),
                    t('home.hero.title4')
                  ]}
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
                {t('home.hero.subtitle')}
              </motion.p>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="primary"
                  size="lg"
                  icon={faArrowRight}
                  iconPosition="right"
                  onClick={scrollToMethods}
                >
                  {t('home.hero.cta')}
                </Button>
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
                  <div className="absolute -bottom-4 -right-4 bg-white rounded-lg p-3 shadow-md">
                    <div className="flex items-center">
                      <i className="fas fa-check-circle text-green-600 mr-2"></i>
                      <span className="text-sm font-medium text-blue-800">{t('home.hero.successText')}</span>
                    </div>
                  </div>
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
            {t('home.features.title')}
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
              <h3 className="text-xl font-semibold mb-3">{t('home.features.feature1.title')}</h3>
              <p className="text-gray-600">
                {t('home.features.feature1.desc')}
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
              <h3 className="text-xl font-semibold mb-3">{t('home.features.feature2.title')}</h3>
              <p className="text-gray-600">
                {t('home.features.feature2.desc')}
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
              <h3 className="text-xl font-semibold mb-3">{t('home.features.feature3.title')}</h3>
              <p className="text-gray-600">
                {t('home.features.feature3.desc')}
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
            <h2 className="text-3xl font-bold text-center mb-8">{t('home.howItWorks.title')}</h2>
            <p className="text-gray-600 text-center max-w-3xl mx-auto mb-12">
              {t('home.howItWorks.subtitle')}
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
                  <h3 className="text-xl font-semibold text-center">{t('home.howItWorks.step1.title')}</h3>
                </div>
                <div className="p-6 md:w-3/4">
                  <div className="mb-4">
                    <h4 className="text-lg font-medium mb-2 text-gray-800">{t('home.howItWorks.step1.subtitle')}</h4>
                    <p className="text-gray-600">{t('home.howItWorks.step1.desc')}</p>
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
                  <h3 className="text-xl font-semibold text-center">{t('home.howItWorks.step2.title')}</h3>
                </div>
                <div className="p-6 md:w-3/4">
                  <div className="mb-4">
                    <h4 className="text-lg font-medium mb-2 text-gray-800">{t('home.howItWorks.step2.subtitle')}</h4>
                    <p className="text-gray-600">{t('home.howItWorks.step2.desc')}</p>
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
                  <h3 className="text-xl font-semibold text-center">{t('home.howItWorks.step3.title')}</h3>
                </div>
                <div className="p-6 md:w-3/4">
                  <div className="mb-4">
                    <h4 className="text-lg font-medium mb-2 text-gray-800">{t('home.howItWorks.step3.subtitle')}</h4>
                    <p className="text-gray-600">{t('home.howItWorks.step3.desc')}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Methods Section - Add this new section */}
      <section ref={methodsRef} id="recognition-methods" className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold mb-4">{t('home.methods.title')}</h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              {t('home.methods.subtitle')}
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Fast Text Recognition Card */}
            <motion.div
              whileHover={{ y: -5, boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="bg-white rounded-xl overflow-hidden shadow-sm border border-gray-100"
            >
              <div className="p-6">
                <div className="w-16 h-16 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mb-4 mx-auto">
                  <FontAwesomeIcon icon={faBolt} size="2x" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-center">{t('home.methods.fast.title')}</h3>
                <p className="text-gray-600 mb-6 text-center">
                  {t('home.methods.fast.description')}
                </p>
                <ul className="mb-6 space-y-2">
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.fast.feature1')}</span>
                  </li>
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.fast.feature2')}</span>
                  </li>
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.fast.feature3')}</span>
                  </li>
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.fast.feature4')}</span>
                  </li>
                </ul>
                <div className="flex flex-col space-y-3">
                  <Link to="/text-recognition" className="w-full">
                    <Button 
                      variant="primary" 
                      size="md" 
                      icon={faFont} 
                      iconPosition="left"
                      className="w-full"
                    >
                      Text Recognition
                    </Button>
                  </Link>
                </div>
              </div>
            </motion.div>
            
            {/* Complete Process Card */}
            <motion.div
              whileHover={{ y: -5, boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="bg-white rounded-xl overflow-hidden shadow-sm border border-gray-100"
            >
              <div className="p-6">
                <div className="w-16 h-16 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center mb-4 mx-auto">
                  <FontAwesomeIcon icon={faClockRotateLeft} size="2x" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-center">{t('home.methods.complete.title')}</h3>
                <p className="text-gray-600 mb-6 text-center">
                  {t('home.methods.complete.description')}
                </p>
                <ul className="mb-6 space-y-2">
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.complete.feature1')}</span>
                  </li>
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.complete.feature2')}</span>
                  </li>
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.complete.feature3')}</span>
                  </li>
                  <li className="flex items-center">
                    <span className="text-green-500 mr-2">✓</span> 
                    <span>{t('home.methods.complete.feature4')}</span>
                  </li>
                </ul>
                <div className="text-center mt-9">
                  <Link to="/process" className="w-full">
                    <Button 
                      variant="primary" 
                      size="md" 
                      icon={faClockRotateLeft} 
                      iconPosition="left"
                      className="w-full"
                    >
                      {t('home.methods.complete.cta')}
                    </Button>
                  </Link>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>
    </>
  );
};

export default Home; 
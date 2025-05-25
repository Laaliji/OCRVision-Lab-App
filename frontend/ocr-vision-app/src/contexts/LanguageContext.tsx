import React, { createContext, useState, useContext, ReactNode } from 'react';

type Language = 'en' | 'fr';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string, params?: Record<string, any>) => string;
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    // Common
    'app.title': 'OCR Vision App',
    'app.subtitle': 'Upload your image, follow the process step by step and get the text extracted by our CNN model.',
    'button.next': 'Next',
    'button.previous': 'Previous',
    'button.startAgain': 'Start Again',
    'button.processImage': 'Process Image',
    'button.useOriginal': 'Use Original',
    'button.useProcessed': 'Use Processed Image',

    // Header
    'header.home': 'Home',
    'header.features': 'Features',
    'header.howItWorks': 'How It Works',
    'header.process': 'Process',

    // Steps
    'steps.upload': 'Upload Image',
    'steps.upload.desc': 'Provide the image you want to process',
    'steps.validate': 'Validate & Process',
    'steps.validate.desc': 'Ensure image meets requirements',
    'steps.transform': 'Transformations',
    'steps.transform.desc': 'Image degradation techniques',
    'steps.segment': 'Segmentation',
    'steps.segment.desc': 'Character segmentation',
    'steps.recognize': 'Recognition',
    'steps.recognize.desc': 'Character recognition with CNN',
    'steps.results': 'Results',
    'steps.results.desc': 'View extracted text and analysis',

    // Upload Step
    'upload.title': 'Upload Your Image',
    'upload.requirements': 'Important: For best results, please upload:',
    'upload.requirement1': 'Grayscale image containing clear text/digit characters',
    'upload.requirement2': 'Image size between 20x20 and 64x64 pixels',
    'upload.requirement3': 'File format: JPG, PNG, TIFF, or BMP',
    'upload.requirement4': 'Maximum file size: 5MB',
    'upload.selected': 'Selected:',

    // Validation Step
    'validation.title': 'Image Validation & Processing',
    'validation.about': 'About Dataset Requirements:',
    'validation.aboutDesc': 'The OCR model was trained on specific image characteristics. For accurate results, your image should match these requirements.',
    'validation.req1': 'Grayscale images (single channel)',
    'validation.req2': 'Standardized size (typically 32x32 pixels)',
    'validation.req3': 'Centered characters',
    'validation.req4': 'Consistent contrast',
    'validation.processDesc': 'If your image doesn\'t match these requirements, we\'ll process it to help improve recognition accuracy.',
    'validation.analyzing': 'Analyzing image characteristics...',
    'validation.uploadFirst': 'Please upload an image in the previous step',
    'validation.meetsRequirements': 'Image meets all requirements for the model.',
    'validation.note.title': 'Note:',
    'validation.note.text': 'While the image doesn\'t perfectly match the model\'s training data format, we\'ll still process it and apply controlled degradation techniques that often yield better results. We\'ll ensure it meets the basic requirements before proceeding.',

    // Processing Results
    'processing.title': 'Image Processing Results',
    'processing.original': 'Original Image',
    'processing.processed': 'Processed Image',
    'processing.notCompatible': 'Not Compatible',
    'processing.compatible': 'Compatible',
    'processing.dimensions': 'Dimensions:',
    'processing.grayscale': 'Grayscale:',
    'processing.note': 'Important Note:',
    'processing.noteDesc': 'We\'ll ensure your image meets the formatting requirements, but our OCR model actually performs better with slightly degraded images. After processing, we\'ll apply controlled degradation techniques to improve recognition accuracy.',

    // Transformations
    'transform.title': 'Image Transformations',
    'transform.desc': 'The following standard degradation techniques will be applied to test the model\'s robustness.',
    'transform.techniques': 'Standard Image Degradation Techniques',
    'transform.techniquesDesc': 'These transformations will be applied to your image during processing to test model robustness.',
    'transform.gaussianNoise': 'Gaussian Noise',
    'transform.gaussianNoiseDesc': 'Random noise is added to the image',
    'transform.saltPepper': 'Salt & Pepper',
    'transform.saltPepperDesc': 'Salt and pepper noise adds white and black pixels',
    'transform.brightness': 'Brightness',
    'transform.brightnessDesc': 'Brightness is adjusted to simulate different lighting',
    'transform.blur': 'Blur',
    'transform.blurDesc': 'Gaussian blur simulates out of focus or low resolution',
    'transform.insight': 'Key Insight:',
    'transform.insightDesc': 'Our OCR model actually performs better with carefully degraded images! These transformations are not just for testing robustness - they often improve accuracy compared to preprocessing. This is because the model was trained on images with similar characteristics to the degraded versions.',

    // Segmentation
    'segment.title': 'Character Segmentation',
    'segment.desc': 'In this step, the image is processed to identify individual characters:',
    'segment.process': 'Segmentation Process:',
    'segment.step0': 'Original Image',
    'segment.step1': 'Convert image to grayscale',
    'segment.step2': 'Apply Gaussian blur to reduce noise',
    'segment.step3': 'Use adaptive thresholding for binarization',
    'segment.step4': 'Apply morphological operations (closing)',
    'segment.step5': 'Find contours to identify potential characters',
    'segment.step6': 'Filter contours based on size and aspect ratio',
    'segment.step7': 'Extract individual character regions',
    'segment.preview': 'Character segmentation will be performed during final processing',
    'segment.noImage': 'No image uploaded',
    'segment.stepExplanation0': 'This is the original image that will be processed.',
    'segment.stepExplanation1': 'Converting to grayscale simplifies the image by removing color information, making it easier to process.',
    'segment.stepExplanation2': 'Gaussian blur reduces noise and detail level, making character detection more robust.',
    'segment.stepExplanation3': 'Adaptive thresholding creates a binary image, separating text from background based on local lighting conditions.',
    'segment.stepExplanation4': 'Morphological operations (like dilation and erosion) help fill small holes and remove small noise in the binary image.',
    'segment.stepExplanation5': 'Finding contours in the binary image identifies potential character boundaries.',
    'segment.stepExplanation6': 'Filtering contours based on size and aspect ratio helps distinguish actual characters from noise.',
    'segment.stepExplanation7': 'Characters are extracted as individual regions to be processed by the recognition model.',

    // Recognition
    'recognition.title': 'Character Recognition',
    'recognition.processing': 'Running CNN model for character recognition...',
    'recognition.process': 'Recognition Process:',
    'recognition.step0': 'Original Image',
    'recognition.step1': 'Apply standard degradation techniques to test robustness',
    'recognition.step2': 'Preprocess both original and degraded images',
    'recognition.step3': 'Segment characters from both images',
    'recognition.step4': 'For each character segment:',
    'recognition.step4a': 'Resize to 32x32 pixels',
    'recognition.step4b': 'Normalize pixel values',
    'recognition.step4c': 'Feed into MobileNetV2 model',
    'recognition.step4d': 'Get predicted class and confidence',
    'recognition.step5': 'Combine character predictions into text',
    'recognition.step6': 'Compare results between original and preprocessed images',
    'recognition.stepLabel': 'Step',
    'recognition.currentStep': 'Current Step',
    'recognition.expectedResult': 'Expected Result',
    'recognition.simulationPlaceholder': 'Simulation Preview',
    'recognition.play': 'Play',
    'recognition.pause': 'Pause',
    'recognition.next': 'Next',
    'recognition.prev': 'Previous',
    'recognition.reset': 'Reset',
    'recognition.stepExplanation0': 'This is the original image that will be processed for character recognition.',
    'recognition.stepExplanation1': 'Standard degradation techniques (like Gaussian noise, blur) are applied to test how robust the recognition is under different conditions.',
    'recognition.stepExplanation2': 'Both the original and degraded images are preprocessed to enhance features that help with recognition.',
    'recognition.stepExplanation3': 'Characters are segmented from both the original and degraded images to be recognized individually.',
    'recognition.stepExplanation4': 'Each character segment is prepared for the CNN model by resizing, normalizing, and preparing the data format.',
    'recognition.stepExplanation5': 'The prepared character images are fed into the MobileNetV2 model to predict which character they represent.',
    'recognition.stepExplanation6': 'Individual character predictions are combined into complete text, and results from original vs. degraded images are compared.',

    // Results
    'results.title': 'Results',
    'results.originalImage': 'Original Image',
    'results.degradedImage': 'Degraded Image (Best Results)',
    'results.segmentedChars': 'Segmented Characters',
    'results.modelResults': 'Model Results',
    'results.extractedText': 'Extracted Text',
    'results.noText': 'No text detected',
    'results.confidence': 'Confidence:',
    'results.analysis': 'Analysis',
    'results.note': 'Note:',
    'results.noteDesc': 'The degraded image processing typically provides more accurate results with our OCR model. Pre-processing is not displayed as it often yields less accurate predictions.',
    'results.analysisResults': 'Analysis Results',
    'results.appliedTransformations': 'Applied transformations:',
    
    // New: Result Actions
    'results.actions': 'Save or Copy Results',
    'results.copy': 'Copy to Clipboard',
    'results.copied': 'Copied!',
    'results.copyFailed': 'Copy failed',
    'results.download': 'Download as Text File',
    'results.downloaded': 'Downloaded!',
    'results.downloadFailed': 'Download failed',
    'results.preview': 'Preview',
    
    // New: Guided Tour
    'guidedTour.segmentationTitle': 'Character Segmentation Process Guide',
    'guidedTour.recognitionTitle': 'Character Recognition Process Guide',
    'guidedTour.step': 'Step',
    'guidedTour.of': 'of',
    'guidedTour.prev': 'Previous',
    'guidedTour.next': 'Next',
    'guidedTour.play': 'Auto Play',
    'guidedTour.pause': 'Pause',
    'guidedTour.start': 'Start Tour',
    'guidedTour.exit': 'Exit Tour',
    'guidedTour.skipToEnd': 'Skip to End',
    'guidedTour.imageNotAvailable': 'Image preview not available',

    // Errors
    'error.selectImage': 'Please select an image first',
    'error.meetsRequirements': 'Please select or process an image that meets the requirements',
    'error.processing': 'An error occurred during recognition',

    // Home Page
    'home.hero.title1': 'Intelligent text extraction',
    'home.hero.title2': 'Advanced document recognition',
    'home.hero.title3': 'High precision OCR solution',
    'home.hero.title4': 'Optimized image processing',
    'home.hero.subtitle': 'OCRVision uses advanced image processing techniques and AI to extract text even from low-resolution images.',
    'home.hero.cta': 'Get Started Now',
    'home.hero.successText': 'Text extracted successfully',
    
    'home.features.title': 'Key Features',
    'home.features.feature1.title': 'Advanced Image Processing',
    'home.features.feature1.desc': 'Grayscale conversion, adaptive thresholding, and denoising to optimize image quality before extraction.',
    'home.features.feature2.title': 'High Precision OCR',
    'home.features.feature2.desc': 'Using Tesseract OCR for reliable text recognition, even on poor quality documents.',
    'home.features.feature3.title': 'Comparative Analysis',
    'home.features.feature3.desc': 'Visualize the difference between direct extraction and extraction with preprocessing to measure improvement.',
    
    'home.howItWorks.title': 'How It Works',
    'home.howItWorks.subtitle': 'Our application uses advanced image processing and computer vision techniques to optimize text recognition in low-quality images.',
    'home.howItWorks.step1.title': 'Image Acquisition',
    'home.howItWorks.step1.subtitle': 'Upload and Analysis',
    'home.howItWorks.step1.desc': 'The user uploads an image containing text. The system analyzes the image properties: resolution, color depth, and signal-to-noise ratio.',
    'home.howItWorks.step2.title': 'Advanced Preprocessing',
    'home.howItWorks.step2.subtitle': 'Image Enhancement',
    'home.howItWorks.step2.desc': 'Application of advanced techniques such as adaptive histogram equalization, noise reduction, and adaptive thresholding to improve text visibility.',
    'home.howItWorks.step3.title': 'OCR Extraction',
    'home.howItWorks.step3.subtitle': 'Text Recognition',
    'home.howItWorks.step3.desc': 'The OCR engine analyzes the preprocessed image to identify and extract text, comparing results before and after preprocessing.',
    'home.howItWorks.cta': 'Try It Now',

    // Footer
    'footer.copyright': '© {year} OCRVision Lab',
    'footer.developed': 'Developed by LAALIJI & HNIOUA',

    // Transformation Info Details
    'transform.techniques.title': 'Standard Image Degradation Techniques',
    'transform.techniques.description': 'These transformations will be applied to your image during processing to test model robustness.',
    'transform.gaussian.title': 'Gaussian Noise',
    'transform.gaussian.desc': 'Random noise is added to the image',
    'transform.saltpepper.title': 'Salt & Pepper',
    'transform.saltpepper.desc': 'Salt and pepper noise adds white and black pixels',
    'transform.brightness.title': 'Brightness',
    'transform.brightness.desc': 'Brightness is adjusted to simulate different lighting',
    'transform.blur.title': 'Blur',
    'transform.blur.desc': 'Gaussian blur simulates out of focus or low resolution',
    'transform.insight.title': 'Key Insight:',
    'transform.insight.desc': 'Our OCR model actually performs better with carefully degraded images! These transformations are not just for testing robustness - they often improve accuracy compared to preprocessing. This is because the model was trained on images with similar characteristics to the degraded versions.',
    
    // Image Validation Details
    'validation.characteristics': 'Image Characteristics:',
    'validation.dimensions': 'Dimensions:',
    'validation.aspectRatio': 'Aspect Ratio:',
    'validation.bitDepth': 'Bit Depth:',
    'validation.grayscale': 'Grayscale:',
    'validation.yes': 'Yes',
    'validation.no': 'No',
    'validation.doesNotMeet': 'Image does not meet requirements:',
    'validation.sizeRequirement': 'size must be between {min}x{min} and {max}x{max} pixels',
    'validation.mustBeGrayscale': 'must be grayscale',
    'validation.incompatible': 'Not Compatible',
    'validation.compatible': 'Compatible',

    // Home Recognition Methods Section
    'home.methods.title': 'Recognition Methods',
    'home.methods.subtitle': 'Choose the recognition approach that best fits your needs',
    'home.methods.fast.title': 'Fast Text Recognition',
    'home.methods.fast.description': 'Quick and efficient text extraction from images',
    'home.methods.fast.feature1': 'Direct text extraction with high accuracy',
    'home.methods.fast.feature2': 'Optimized for clear text in images',
    'home.methods.fast.feature3': 'Multiple OCR approaches compared',
    'home.methods.fast.feature4': 'Immediate results visualization',
    'home.methods.complete.title': 'Complete Process Simulation',
    'home.methods.complete.description': 'Step-by-step visualization of the OCR process',
    'home.methods.complete.feature1': 'Detailed processing visualization',
    'home.methods.complete.feature2': 'Character-by-character recognition',
    'home.methods.complete.feature3': 'Advanced image preprocessing',
    'home.methods.complete.feature4': 'Educational insights into OCR technology',
    'home.methods.complete.cta': 'Try Full Process',

    // Text Recognition
    'textRecognition.title': 'Text Recognition',
    'textRecognition.subtitle': 'Upload an image with text to extract its content using advanced OCR techniques',
    'textRecognition.uploadTitle': 'Upload an Image',
    'textRecognition.uploadInstruction': 'Click or drag image file here to upload',
    'textRecognition.uploadFormats': 'Supported formats: JPG, PNG, BMP, TIFF',
    'textRecognition.processing': 'Processing...',
    'textRecognition.process': 'Recognize Text',
    'textRecognition.resultsTitle': 'Recognition Results',
    'textRecognition.resultsSubtitle': 'Text extraction results with various OCR techniques',
    'textRecognition.recognizedText': 'Recognized Text',
    'textRecognition.noTextDetected': 'No text detected',
    'textRecognition.confidence': 'Confidence:',
    'textRecognition.originalImage': 'Original Image',
    'textRecognition.characterVisualization': 'Character Visualization',
    'textRecognition.visualizationDesc': 'Visualization of the character recognition process',
    'textRecognition.methodComparison': 'Method Comparison',
    'textRecognition.method': 'Method',
    'textRecognition.recognizedTextLabel': 'Recognized Text',
    'textRecognition.directOriginal': 'Direct (Original)',
    'textRecognition.directPreprocessed': 'Direct (Preprocessed)',
    'textRecognition.characterByCharacter': 'Character-by-Character',
    'textRecognition.wordSegmentation': 'Word Segmentation',
    'textRecognition.tesseractOCR': 'Tesseract OCR',
    'textRecognition.tryAnother': 'Try Another Image',
    'textRecognition.characterRecognitionProcess': 'Character Recognition Process',
    
    // Character Recognition Process Steps
    'textRecognition.process.step1.title': 'Image Preprocessing:',
    'textRecognition.process.step1.desc': 'The uploaded image is converted to grayscale, resized, and enhanced with adaptive thresholding to improve text visibility.',
    'textRecognition.process.step2.title': 'Character Segmentation:',
    'textRecognition.process.step2.desc': 'The preprocessed image is analyzed to identify and isolate individual characters using contour detection algorithms.',
    'textRecognition.process.step3.title': 'Character Preprocessing:',
    'textRecognition.process.step3.item1': 'Converted to grayscale',
    'textRecognition.process.step3.item2': 'Resized to 32x32 pixels',
    'textRecognition.process.step3.item3': 'Standardized with pixel values scaled between 0-1',
    'textRecognition.process.step3.item4': 'Reshaped to match the model\'s input requirements',
    'textRecognition.process.step4.title': 'Character Prediction:',
    'textRecognition.process.step4.desc': 'Each preprocessed character is fed into the trained neural network model (MobileNetV2) to predict the character.',
    'textRecognition.process.step5.title': 'Character Ordering:',
    'textRecognition.process.step5.desc': 'Predicted characters are arranged based on their spatial positions in the original image (sorted by x-coordinates).',
    'textRecognition.process.step6.title': 'Text Construction:',
    'textRecognition.process.step6.desc': 'The ordered characters are combined to form the final recognized text.',
  },
  fr: {
    // Commun
    'app.title': 'Application OCR Vision',
    'app.subtitle': 'Téléchargez votre image, suivez le processus étape par étape et obtenez le texte extrait par notre modèle CNN.',
    'button.next': 'Suivant',
    'button.previous': 'Précédent',
    'button.startAgain': 'Recommencer',
    'button.processImage': 'Traiter l\'image',
    'button.useOriginal': 'Utiliser l\'original',
    'button.useProcessed': 'Utiliser l\'image traitée',

    // Header
    'header.home': 'Accueil',
    'header.features': 'Fonctionnalités',
    'header.howItWorks': 'Comment ça marche',
    'header.process': 'Traitement',

    // Étapes
    'steps.upload': 'Télécharger l\'image',
    'steps.upload.desc': 'Fournir l\'image à traiter',
    'steps.validate': 'Valider et traiter',
    'steps.validate.desc': 'Vérifier que l\'image répond aux exigences',
    'steps.transform': 'Transformations',
    'steps.transform.desc': 'Techniques de dégradation d\'image',
    'steps.segment': 'Segmentation',
    'steps.segment.desc': 'Segmentation des caractères',
    'steps.recognize': 'Reconnaissance',
    'steps.recognize.desc': 'Reconnaissance des caractères par CNN',
    'steps.results': 'Résultats',
    'steps.results.desc': 'Voir le texte extrait et l\'analyse',

    // Étape de téléchargement
    'upload.title': 'Téléchargez votre image',
    'upload.requirements': 'Important : Pour de meilleurs résultats, veuillez télécharger :',
    'upload.requirement1': 'Une image en niveaux de gris contenant des caractères/chiffres clairs',
    'upload.requirement2': 'Taille d\'image entre 20x20 et 64x64 pixels',
    'upload.requirement3': 'Format de fichier : JPG, PNG, TIFF ou BMP',
    'upload.requirement4': 'Taille maximale : 5Mo',
    'upload.selected': 'Sélectionné :',

    // Étape de validation
    'validation.title': 'Validation et traitement de l\'image',
    'validation.about': 'À propos des exigences du jeu de données :',
    'validation.aboutDesc': 'Le modèle OCR a été entraîné sur des caractéristiques d\'image spécifiques. Pour des résultats précis, votre image doit correspondre à ces exigences.',
    'validation.req1': 'Images en niveaux de gris (canal unique)',
    'validation.req2': 'Taille standardisée (généralement 32x32 pixels)',
    'validation.req3': 'Caractères centrés',
    'validation.req4': 'Contraste constant',
    'validation.processDesc': 'Si votre image ne correspond pas à ces exigences, nous la traiterons pour améliorer la précision de la reconnaissance.',
    'validation.analyzing': 'Analyse des caractéristiques de l\'image...',
    'validation.uploadFirst': 'Veuillez télécharger une image à l\'étape précédente',
    'validation.meetsRequirements': 'L\'image répond à toutes les exigences du modèle.',
    'validation.note.title': 'Remarque :',
    'validation.note.text': 'Bien que l\'image ne corresponde pas parfaitement au format des données d\'entraînement du modèle, nous la traiterons tout de même et appliquerons des techniques de dégradation contrôlée qui donnent souvent de meilleurs résultats. Nous nous assurerons qu\'elle répond aux exigences de base avant de continuer.',

    // Résultats du traitement
    'processing.title': 'Résultats du traitement d\'image',
    'processing.original': 'Image originale',
    'processing.processed': 'Image traitée',
    'processing.notCompatible': 'Non compatible',
    'processing.compatible': 'Compatible',
    'processing.dimensions': 'Dimensions :',
    'processing.grayscale': 'Niveaux de gris :',
    'processing.note': 'Note importante :',
    'processing.noteDesc': 'Nous nous assurerons que votre image répond aux exigences de formatage, mais notre modèle OCR fonctionne en fait mieux avec des images légèrement dégradées. Après traitement, nous appliquerons des techniques de dégradation contrôlée pour améliorer la précision de la reconnaissance.',

    // Transformations
    'transform.title': 'Transformations d\'image',
    'transform.desc': 'Les techniques de dégradation standard suivantes seront appliquées pour tester la robustesse du modèle.',
    'transform.techniques': 'Techniques standard de dégradation d\'image',
    'transform.techniquesDesc': 'Ces transformations seront appliquées à votre image pendant le traitement pour tester la robustesse du modèle.',
    'transform.gaussianNoise': 'Bruit gaussien',
    'transform.gaussianNoiseDesc': 'Du bruit aléatoire est ajouté à l\'image',
    'transform.saltPepper': 'Bruit sel et poivre',
    'transform.saltPepperDesc': 'Le bruit sel et poivre ajoute des pixels blancs et noirs',
    'transform.brightness': 'Luminosité',
    'transform.brightnessDesc': 'La luminosité est ajustée pour simuler différents éclairages',
    'transform.blur': 'Flou',
    'transform.blurDesc': 'Le flou gaussien simule une mise au point ou une résolution faible',
    'transform.insight': 'Insight clé :',
    'transform.insightDesc': 'Notre modèle OCR fonctionne en fait mieux avec des images soigneusement dégradées ! Ces transformations ne servent pas seulement à tester la robustesse - elles améliorent souvent la précision par rapport au prétraitement. C\'est parce que le modèle a été entraîné sur des images aux caractéristiques similaires aux versions dégradées.',

    // Segmentation
    'segment.title': 'Segmentation des caractères',
    'segment.desc': 'Dans cette étape, l\'image est traitée pour identifier les caractères individuels :',
    'segment.process': 'Processus de segmentation :',
    'segment.step0': 'Image originale',
    'segment.step1': 'Convertir l\'image en niveaux de gris',
    'segment.step2': 'Appliquer un flou gaussien pour réduire le bruit',
    'segment.step3': 'Utiliser un seuillage adaptatif pour la binarisation',
    'segment.step4': 'Appliquer des opérations morphologiques (fermeture)',
    'segment.step5': 'Trouver des contours pour identifier les caractères potentiels',
    'segment.step6': 'Filtrer les contours selon la taille et le ratio d\'aspect',
    'segment.step7': 'Extraire les régions de caractères individuels',
    'segment.preview': 'La segmentation des caractères sera effectuée lors du traitement final',
    'segment.noImage': 'Aucune image téléchargée',
    'segment.stepExplanation0': 'Ceci est l\'image originale qui sera traitée.',
    'segment.stepExplanation1': 'La conversion en niveaux de gris simplifie l\'image en supprimant les informations de couleur, facilitant le traitement.',
    'segment.stepExplanation2': 'Le flou gaussien réduit le bruit et le niveau de détail, rendant la détection des caractères plus robuste.',
    'segment.stepExplanation3': 'Le seuillage adaptatif crée une image binaire, séparant le texte du fond en fonction des conditions d\'éclairage locales.',
    'segment.stepExplanation4': 'Les opérations morphologiques (comme la dilatation et l\'érosion) aident à combler les petits trous et à supprimer les petits bruits dans l\'image binaire.',
    'segment.stepExplanation5': 'La recherche de contours dans l\'image binaire identifie les limites potentielles des caractères.',
    'segment.stepExplanation6': 'Le filtrage des contours en fonction de la taille et du ratio d\'aspect aide à distinguer les caractères réels du bruit.',
    'segment.stepExplanation7': 'Les caractères sont extraits en tant que régions individuelles pour être traités par le modèle de reconnaissance.',

    // Reconnaissance
    'recognition.title': 'Reconnaissance des caractères',
    'recognition.processing': 'Exécution du modèle CNN pour la reconnaissance des caractères...',
    'recognition.process': 'Processus de reconnaissance :',
    'recognition.step0': 'Image originale',
    'recognition.step1': 'Appliquer des techniques de dégradation standard pour tester la robustesse',
    'recognition.step2': 'Prétraiter les images originales et dégradées',
    'recognition.step3': 'Segmenter les caractères des deux images',
    'recognition.step4': 'Pour chaque segment de caractère :',
    'recognition.step4a': 'Redimensionner à 32x32 pixels',
    'recognition.step4b': 'Normaliser les valeurs de pixels',
    'recognition.step4c': 'Injecter dans le modèle MobileNetV2',
    'recognition.step4d': 'Obtenir la classe prédite et la confiance',
    'recognition.step5': 'Combiner les prédictions de caractères en texte',
    'recognition.step6': 'Comparer les résultats entre les images originales et prétraitées',
    'recognition.stepLabel': 'Étape',
    'recognition.currentStep': 'Étape actuelle',
    'recognition.expectedResult': 'Résultat attendu',
    'recognition.simulationPlaceholder': 'Aperçu de simulation',
    'recognition.play': 'Lancer',
    'recognition.pause': 'Pause',
    'recognition.next': 'Suivant',
    'recognition.prev': 'Précédent',
    'recognition.reset': 'Réinitialiser',
    'recognition.stepExplanation0': 'Ceci est l\'image originale qui sera traitée pour la reconnaissance des caractères.',
    'recognition.stepExplanation1': 'Des techniques de dégradation standard (comme le bruit gaussien, le flou) sont appliquées pour tester la robustesse de la reconnaissance dans différentes conditions.',
    'recognition.stepExplanation2': 'Les images originales et dégradées sont prétraitées pour améliorer les caractéristiques qui aident à la reconnaissance.',
    'recognition.stepExplanation3': 'Les caractères sont segmentés à partir des images originales et dégradées pour être reconnus individuellement.',
    'recognition.stepExplanation4': 'Chaque segment de caractère est préparé pour le modèle CNN en le redimensionnant, en le normalisant et en préparant le format de données.',
    'recognition.stepExplanation5': 'Les images de caractères préparées sont injectées dans le modèle MobileNetV2 pour prédire quel caractère elles représentent.',
    'recognition.stepExplanation6': 'Les prédictions de caractères individuels sont combinées en texte complet, et les résultats des images originales et dégradées sont comparés.',

    // Résultats
    'results.title': 'Résultats',
    'results.originalImage': 'Image originale',
    'results.degradedImage': 'Image dégradée (Meilleurs résultats)',
    'results.segmentedChars': 'Caractères segmentés',
    'results.modelResults': 'Résultats du modèle',
    'results.extractedText': 'Texte extrait',
    'results.noText': 'Aucun texte détecté',
    'results.confidence': 'Confiance :',
    'results.analysis': 'Analyse',
    'results.note': 'Note :',
    'results.noteDesc': 'Le traitement d\'image dégradée fournit généralement des résultats plus précis avec notre modèle OCR. Le prétraitement n\'est pas affiché car il donne souvent des prédictions moins précises.',
    'results.analysisResults': 'Résultats d\'analyse',
    'results.appliedTransformations': 'Transformations appliquées :',

    // Erreurs
    'error.selectImage': 'Veuillez d\'abord sélectionner une image',
    'error.meetsRequirements': 'Veuillez sélectionner ou traiter une image qui répond aux exigences',
    'error.processing': 'Une erreur s\'est produite lors de la reconnaissance',

    // Page d'accueil
    'home.hero.title1': 'Extraction intelligente de texte',
    'home.hero.title2': 'Reconnaissance avancée de documents',
    'home.hero.title3': 'Solution OCR de haute précision',
    'home.hero.title4': 'Traitement d\'images optimisé',
    'home.hero.subtitle': 'OCRVision utilise des techniques avancées de traitement d\'image et l\'IA pour extraire du texte même à partir d\'images à faible résolution.',
    'home.hero.cta': 'Commencer maintenant',
    'home.hero.successText': 'Texte extrait avec succès',
    
    'home.features.title': 'Fonctionnalités principales',
    'home.features.feature1.title': 'Traitement d\'images avancé',
    'home.features.feature1.desc': 'Conversion en niveaux de gris, seuillage adaptatif, et débruitage pour optimiser la qualité de l\'image avant extraction.',
    'home.features.feature2.title': 'OCR haute précision',
    'home.features.feature2.desc': 'Utilisation de Tesseract OCR pour une reconnaissance de texte fiable, même sur des documents de qualité médiocre.',
    'home.features.feature3.title': 'Analyse comparative',
    'home.features.feature3.desc': 'Visualisez la différence entre l\'extraction directe et l\'extraction avec prétraitement pour mesurer l\'amélioration.',
    
    'home.howItWorks.title': 'Comment ça marche',
    'home.howItWorks.subtitle': 'Notre application utilise des techniques avancées de traitement d\'images et de vision par ordinateur pour optimiser la reconnaissance de texte dans des images de faible qualité.',
    'home.howItWorks.step1.title': 'Acquisition d\'image',
    'home.howItWorks.step1.subtitle': 'Téléchargement et analyse',
    'home.howItWorks.step1.desc': 'L\'utilisateur télécharge une image contenant du texte. Le système analyse les propriétés de l\'image : résolution, profondeur de couleur, et rapport signal/bruit.',
    'home.howItWorks.step2.title': 'Prétraitement avancé',
    'home.howItWorks.step2.subtitle': 'Amélioration de l\'image',
    'home.howItWorks.step2.desc': 'Application de techniques avancées comme l\'égalisation adaptative d\'histogramme, la réduction de bruit et le seuillage adaptatif pour améliorer la visibilité du texte.',
    'home.howItWorks.step3.title': 'Extraction OCR',
    'home.howItWorks.step3.subtitle': 'Reconnaissance de texte',
    'home.howItWorks.step3.desc': 'Le moteur OCR analyse l\'image prétraitée pour identifier et extraire le texte, avec comparaison des résultats avant et après prétraitement.',
    'home.howItWorks.cta': 'Essayer maintenant',

    // Footer
    'footer.copyright': '© {year} OCRVision Lab',
    'footer.developed': 'Développé par LAALIJI & HNIOUA',

    // Détails des transformations
    'transform.techniques.title': 'Techniques standard de dégradation d\'image',
    'transform.techniques.description': 'Ces transformations seront appliquées à votre image pendant le traitement pour tester la robustesse du modèle.',
    'transform.gaussian.title': 'Bruit gaussien',
    'transform.gaussian.desc': 'Du bruit aléatoire est ajouté à l\'image',
    'transform.saltpepper.title': 'Sel et poivre',
    'transform.saltpepper.desc': 'Le bruit sel et poivre ajoute des pixels blancs et noirs',
    'transform.brightness.title': 'Luminosité',
    'transform.brightness.desc': 'La luminosité est ajustée pour simuler différents éclairages',
    'transform.blur.title': 'Flou',
    'transform.blur.desc': 'Le flou gaussien simule une mise au point ou une résolution faible',
    'transform.insight.title': 'Point clé :',
    'transform.insight.desc': 'Notre modèle OCR fonctionne mieux avec des images soigneusement dégradées ! Ces transformations ne servent pas seulement à tester la robustesse - elles améliorent souvent la précision par rapport au prétraitement. C\'est parce que le modèle a été entraîné sur des images aux caractéristiques similaires aux versions dégradées.',
    
    // Détails de validation d'image
    'validation.characteristics': 'Caractéristiques de l\'image :',
    'validation.dimensions': 'Dimensions :',
    'validation.aspectRatio': 'Ratio d\'aspect :',
    'validation.bitDepth': 'Profondeur de bits :',
    'validation.grayscale': 'Niveaux de gris :',
    'validation.yes': 'Oui',
    'validation.no': 'Non',
    'validation.doesNotMeet': 'L\'image ne répond pas aux exigences :',
    'validation.sizeRequirement': 'la taille doit être entre {min}x{min} et {max}x{max} pixels',
    'validation.mustBeGrayscale': 'doit être en niveaux de gris',
    'validation.incompatible': 'Non compatible',
    'validation.compatible': 'Compatible',

    // New: Result Actions
    'results.actions': 'Sauvegarder ou Copier les Résultats',
    'results.copy': 'Copier dans le Presse-papier',
    'results.copied': 'Copié !',
    'results.copyFailed': 'Échec de la copie',
    'results.download': 'Télécharger en fichier texte',
    'results.downloaded': 'Téléchargé !',
    'results.downloadFailed': 'Échec du téléchargement',
    'results.preview': 'Aperçu',
    
    // New: Guided Tour
    'guidedTour.segmentationTitle': 'Guide du Processus de Segmentation des Caractères',
    'guidedTour.recognitionTitle': 'Guide du Processus de Reconnaissance des Caractères',
    'guidedTour.step': 'Étape',
    'guidedTour.of': 'sur',
    'guidedTour.prev': 'Précédent',
    'guidedTour.next': 'Suivant',
    'guidedTour.play': 'Lecture automatique',
    'guidedTour.pause': 'Pause',
    'guidedTour.start': 'Démarrer le guide',
    'guidedTour.exit': 'Quitter le guide',
    'guidedTour.skipToEnd': 'Passer à la fin',
    'guidedTour.imageNotAvailable': 'Aperçu de l\'image non disponible',

    // Home Recognition Methods Section
    'home.methods.title': 'Choisissez Votre Méthode de Reconnaissance',
    'home.methods.subtitle': 'Sélectionnez entre la reconnaissance rapide de texte ou la simulation détaillée du processus caractère par caractère',
    'home.methods.fast.title': 'Reconnaissance Rapide de Texte',
    'home.methods.fast.description': 'Traitez rapidement les images et obtenez des résultats textuels avec une grande précision. Parfait pour extraire du texte de documents et d\'images.',
    'home.methods.fast.feature1': 'Extraction directe de texte',
    'home.methods.fast.feature2': 'Modes de reconnaissance multiples',
    'home.methods.fast.feature3': 'Comparaison des méthodes',
    'home.methods.fast.feature4': 'Visualisation de la confiance',
    'home.methods.fast.cta': 'Essayer la Reconnaissance de Texte',
    'home.methods.complete.title': 'Simulation Complète du Processus',
    'home.methods.complete.description': 'Découvrez le pipeline OCR complet, du prétraitement de l\'image à la reconnaissance des caractères avec des étapes détaillées.',
    'home.methods.complete.feature1': 'Visualisation étape par étape',
    'home.methods.complete.feature2': 'Transformations d\'image',
    'home.methods.complete.feature3': 'Segmentation des caractères',
    'home.methods.complete.feature4': 'Analyse détaillée',
    'home.methods.complete.cta': 'Essayer le Processus Complet',

    // Text Recognition
    'textRecognition.title': 'Reconnaissance de Texte',
    'textRecognition.subtitle': 'Téléchargez une image contenant du texte pour extraire son contenu à l\'aide de techniques OCR avancées',
    'textRecognition.uploadTitle': 'Télécharger une Image',
    'textRecognition.uploadInstruction': 'Cliquez ou glissez-déposez un fichier image ici',
    'textRecognition.uploadFormats': 'Formats supportés : JPG, PNG, BMP, TIFF',
    'textRecognition.processing': 'Traitement en cours...',
    'textRecognition.process': 'Reconnaître le Texte',
    'textRecognition.resultsTitle': 'Résultats de la Reconnaissance',
    'textRecognition.resultsSubtitle': 'Résultats d\'extraction de texte avec diverses techniques OCR',
    'textRecognition.recognizedText': 'Texte Reconnu',
    'textRecognition.noTextDetected': 'Aucun texte détecté',
    'textRecognition.confidence': 'Confiance :',
    'textRecognition.originalImage': 'Image Originale',
    'textRecognition.characterVisualization': 'Visualisation des Caractères',
    'textRecognition.visualizationDesc': 'Visualisation du processus de reconnaissance des caractères',
    'textRecognition.methodComparison': 'Comparaison des Méthodes',
    'textRecognition.method': 'Méthode',
    'textRecognition.recognizedTextLabel': 'Texte Reconnu',
    'textRecognition.directOriginal': 'Direct (Original)',
    'textRecognition.directPreprocessed': 'Direct (Prétraité)',
    'textRecognition.characterByCharacter': 'Caractère par Caractère',
    'textRecognition.wordSegmentation': 'Segmentation de Mots',
    'textRecognition.tesseractOCR': 'Tesseract OCR',
    'textRecognition.tryAnother': 'Essayer une Autre Image',
    'textRecognition.characterRecognitionProcess': 'Processus de Reconnaissance des Caractères',
    
    // Character Recognition Process Steps
    'textRecognition.process.step1.title': 'Prétraitement d\'Image :',
    'textRecognition.process.step1.desc': 'L\'image téléchargée est convertie en niveaux de gris, redimensionnée, et améliorée avec un seuillage adaptatif pour améliorer la visibilité du texte.',
    'textRecognition.process.step2.title': 'Segmentation des Caractères :',
    'textRecognition.process.step2.desc': 'L\'image prétraitée est analysée pour identifier et isoler les caractères individuels à l\'aide d\'algorithmes de détection de contours.',
    'textRecognition.process.step3.title': 'Prétraitement des Caractères :',
    'textRecognition.process.step3.item1': 'Conversion en niveaux de gris',
    'textRecognition.process.step3.item2': 'Redimensionnement à 32x32 pixels',
    'textRecognition.process.step3.item3': 'Standardisation avec des valeurs de pixels entre 0 et 1',
    'textRecognition.process.step3.item4': 'Restructuration pour correspondre aux exigences d\'entrée du modèle',
    'textRecognition.process.step4.title': 'Prédiction des Caractères :',
    'textRecognition.process.step4.desc': 'Chaque caractère prétraité est soumis au modèle de réseau neuronal (MobileNetV2) pour prédire le caractère.',
    'textRecognition.process.step5.title': 'Ordonnancement des Caractères :',
    'textRecognition.process.step5.desc': 'Les caractères prédits sont organisés en fonction de leurs positions spatiales dans l\'image originale (triés par coordonnées x).',
    'textRecognition.process.step6.title': 'Construction du Texte :',
    'textRecognition.process.step6.desc': 'Les caractères ordonnés sont combinés pour former le texte final reconnu.',
  }
};

const LanguageContext = createContext<LanguageContextType | null>(null);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState<Language>('en');

  const t = (key: string, params?: Record<string, any>): string => {
    let translatedText = translations[language][key] || key;
    
    // Replace parameters if provided
    if (params) {
      Object.keys(params).forEach(param => {
        translatedText = translatedText.replace(new RegExp(`{${param}}`, 'g'), params[param]);
      });
    }
    
    return translatedText;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = (): LanguageContextType => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}; 
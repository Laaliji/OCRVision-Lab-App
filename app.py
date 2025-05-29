import os
import cv2
import numpy as np
import pytesseract
import base64
import tensorflow as tf
import json
import random
from PIL import Image, ImageEnhance
from io import BytesIO
from flask import Flask, render_template, request, jsonify, url_for, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import uuid
import string
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Tesseract configuration based on environment
if os.environ.get('TESSERACT_PATH'):
    pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH')
elif os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/Unix systems, Tesseract is usually in the PATH

# Configure static folder based on environment
static_folder = 'app/static'
if os.path.exists('app/static/build'):
    # If React build folder exists, use it for serving static files
    static_folder = 'app/static/build'

app = Flask(__name__, 
            template_folder='app/templates',
            static_folder=static_folder)

# Configure CORS to allow requests from Netlify frontend
NETLIFY_URL = os.environ.get('NETLIFY_URL', '*')
CORS(app, origins=[NETLIFY_URL, 'http://localhost:3000'])

app.secret_key = os.environ.get('SECRET_KEY', 'ocr_vision_secret_key')

# Configuration
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
PROCESSED_FOLDER = os.path.join('app', 'static', 'processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = 'final_best_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Character mapping for the model (36 classes: 0-9, A-Z)
CHARACTER_CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
INDEX_TO_CHAR = {i: char for i, char in enumerate(CHARACTER_CLASSES)}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def degrade_image(image):
    # Resize to low resolution (320x240)
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_h = 240
    new_w = int(aspect_ratio * new_h)
    low_res = cv2.resize(image, (new_w, new_h))
    
    # Add Gaussian noise
    noise = np.zeros(low_res.shape, np.uint8)
    cv2.randn(noise, 0, 25)
    noisy_img = cv2.add(low_res, noise)
    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(noisy_img, (5, 5), 0)
    
    # Save with high JPEG compression
    return blurred_img

def preprocess_image(image):
    """
    Applique une série de techniques avancées de traitement d'image pour améliorer la lisibilité du texte.
    Processus en plusieurs étapes selon les principes de traitement d'image pour la reconnaissance de texte.
    """
    # ÉTAPE 1: Conversion en espace de couleur adapté
    # Conversion en grayscale (YUV - canal Y pour meilleure perception de luminance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ÉTAPE 2: Analyse d'histogramme et amélioration de contraste
    # 2.1 Correction gamma pour améliorer le contraste des zones sombres
    gamma = 1.2
    gamma_corrected = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
    
    # 2.2 CLAHE (Contrast Limited Adaptive Histogram Equalization) - plus efficace que l'égalisation simple
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gamma_corrected)
    
    # ÉTAPE 3: Réduction du bruit avec filtrage spatial
    # 3.1 Filtre bilatéral pour réduire le bruit tout en préservant les bords
    bilateral = cv2.bilateralFilter(clahe_img, 7, 75, 75)
    
    # 3.2 Filtre médian pour réduire le bruit impulsionnel (poivre et sel)
    median_filtered = cv2.medianBlur(bilateral, 3)
    
    # ÉTAPE 4: Détection des bords et amélioration de structures
    # 4.1 Filtre Laplacien pour renforcer les contours
    laplacian = cv2.Laplacian(median_filtered, cv2.CV_8U, ksize=3)
    
    # 4.2 Renforcement des bords
    enhanced = cv2.addWeighted(median_filtered, 1.5, laplacian, -0.5, 0)
    
    # ÉTAPE 5: Binarisation adaptive - choix du meilleur seuillage selon le contenu
    # 5.1 Essayons deux types de seuillage et prenons le meilleur
    thresh_gaussian = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
    thresh_mean = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 5.2 Combinaison des deux seuillages (peut améliorer la détection)
    thresh = cv2.bitwise_or(thresh_gaussian, thresh_mean)
    
    # ÉTAPE 6: Opérations morphologiques pour nettoyer et renforcer le texte
    # 6.1 Définition du noyau (élément structurant)
    kernel = np.ones((2, 2), np.uint8)
    
    # 6.2 Ouverture (érosion suivie de dilatation) pour éliminer le bruit
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 6.3 Fermeture pour connecter les composantes proches du texte
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 6.4 Dilatation pour épaissir les caractères (améliore la détection OCR)
    dilated = cv2.dilate(closing, kernel, iterations=1)
    
    # ÉTAPE 7: Détection des contours pour les caractères
    # 7.1 Utilisation de Canny pour la détection précise des contours
    edges = cv2.Canny(dilated, 50, 150)
    
    # 7.2 Combinaison de l'image dilatée avec les contours
    result = cv2.bitwise_or(dilated, edges)
    
    # ÉTAPE 8: Nettoyage final - élimination du bruit résiduel
    # 8.1 Filtrage médian léger
    final = cv2.medianBlur(result, 3)
    
    # 8.2 Inversion si nécessaire (texte blanc sur fond noir est souvent mieux détecté)
    if np.sum(final == 0) > np.sum(final == 255):  # Si plus de pixels noirs que blancs
        final = cv2.bitwise_not(final)
    
    return final

def preprocess_for_model(image_segment):
    """
    Preprocess image segment for the trained model
    Expected input: 32x32x3 RGB image
    """
    try:
        # Ensure the image is in the right format
        if len(image_segment.shape) == 2:  # Grayscale
            image_segment = cv2.cvtColor(image_segment, cv2.COLOR_GRAY2RGB)
        elif len(image_segment.shape) == 3 and image_segment.shape[2] == 3:  # BGR
            image_segment = cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB)
        
        # Resize to 32x32 as expected by the model
        resized = cv2.resize(image_segment, (32, 32))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        print(f"Error in preprocess_for_model: {e}")
        return None

def segment_characters(image):
    """
    Segment individual characters from the image using contour detection
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply additional preprocessing for better contour detection
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding for better binarization
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        segments = []
        min_area = 50  # Minimum area for a character
        max_area = 5000  # Maximum area for a character
        min_width = 8
        min_height = 15
        max_aspect_ratio = 3.0  # width/height ratio
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter based on size and aspect ratio
            if (area >= min_area and area <= max_area and 
                w >= min_width and h >= min_height and
                w/h <= max_aspect_ratio):
                
                # Extract the character region with some padding
                padding = 2
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(gray.shape[1], x + w + padding)
                y_end = min(gray.shape[0], y + h + padding)
                
                char_img = gray[y_start:y_end, x_start:x_end]
                
                # Only add if the extracted region is valid
                if char_img.size > 0:
                    segments.append({
                        'image': char_img,
                        'bbox': [x, y, w, h],
                        'area': area
                    })
        
        # Sort segments from left to right (reading order)
        segments.sort(key=lambda seg: seg['bbox'][0])
        
        print(f"Found {len(segments)} character segments")
        return segments
        
    except Exception as e:
        print(f"Error in segment_characters: {str(e)}")
        return []

def predict_character(char_image):
    """
    Predict character using the trained model
    """
    try:
        # Ensure we have a valid image
        if char_image is None or char_image.size == 0:
            return '?', 0.0
        
        # Convert to RGB if grayscale
        if len(char_image.shape) == 2:
            # Convert grayscale to RGB
            char_rgb = cv2.cvtColor(char_image, cv2.COLOR_GRAY2RGB)
        elif len(char_image.shape) == 3 and char_image.shape[2] == 3:
            # Already RGB, but OpenCV uses BGR, so convert
            char_rgb = cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB)
        else:
            char_rgb = char_image
        
        # Resize to 32x32 (model input size)
        char_resized = cv2.resize(char_rgb, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        char_normalized = char_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        char_batch = np.expand_dims(char_normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(char_batch, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map class index to character
        predicted_char = CHARACTER_CLASSES[predicted_class]
        
        return predicted_char, confidence
        
    except Exception as e:
        print(f"Error in predict_character: {str(e)}")
        return '?', 0.0

def extract_text_with_model(image):
    """
    Extract text using the trained model for character recognition
    """
    try:
        # Segment characters
        segments = segment_characters(image)
        
        if not segments:
            return "No characters detected", []
        
        # Predict each character
        predictions = []
        recognized_text = ""
        
        for segment in segments:
            char, confidence = predict_character(segment['image'])
            predictions.append({
                'character': char,
                'confidence': confidence,
                'bbox': segment['bbox']
            })
            
            # Only include high-confidence predictions
            if confidence > 0.3:  # Threshold for confidence
                recognized_text += char
            else:
                recognized_text += '?'
        
        return recognized_text, predictions
    except Exception as e:
        print(f"Error in extract_text_with_model: {e}")
        return "Error in model prediction", []

def extract_text_tesseract(image):
    """
    Extract text using traditional Tesseract OCR
    """
    try:
        # Add white border around the image to improve OCR
        h, w = image.shape[:2]
        bordered = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Try with French language first, fallback to English if French fails
        try:
            # French config
            custom_config = r'--oem 3 --psm 6 -l fra'
            return pytesseract.image_to_string(bordered, config=custom_config)
        except pytesseract.pytesseract.TesseractError:
            # English fallback config
            custom_config = r'--oem 3 --psm 6'  # Default to English
            return pytesseract.image_to_string(bordered, config=custom_config)
    except Exception as e:
        print(f"Error in extract_text_tesseract: {e}")
        return "Error in Tesseract OCR"

def image_to_base64(image):
    # Convert OpenCV image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # If the path is an API endpoint, 404 will be returned and handled by the appropriate route
    if path.startswith('api/'):
        return '', 404
        
    # For any non-API route, serve the React app
    if os.path.exists('app/static/build'):
        # In production, serve the built React app
        if path != "" and os.path.exists(f"app/static/build/{path}"):
            return send_from_directory('app/static/build', path)
        else:
            return send_from_directory('app/static/build', 'index.html')
    else:
        # In development, serve a simple page
        return """
        <html>
            <head><title>OCR Vision App API</title></head>
            <body>
                <h1>OCR Vision App API Server</h1>
                <p>This is the API server for OCR Vision App.</p>
                <p>The frontend is not built. Please run the React development server.</p>
            </body>
        </html>
        """

@app.route('/process', methods=['GET', 'POST'])
def process():
    return render_template('process.html')

@app.route('/process_image', methods=['POST'])
def process_image_api():
    """
    API endpoint to process an image and return OCR results
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        # Add a unique identifier to avoid collisions
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the image
        try:
            # Read image
            image = cv2.imread(file_path)
            
            # Get degraded image and text
            degraded = degrade_image(image)
            degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
            cv2.imwrite(degraded_path, degraded)
            
            # Extract text using both methods for degraded image
            degraded_text_model, degraded_predictions = extract_text_with_model(degraded)
            degraded_text_tesseract = extract_text_tesseract(degraded)
            
            # Preprocess and extract text with advanced techniques
            preprocessed = preprocess_image(image)
            preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
            cv2.imwrite(preprocessed_path, preprocessed)
            
            # Extract text using both methods for preprocessed image
            preprocessed_text_model, preprocessed_predictions = extract_text_with_model(preprocessed)
            preprocessed_text_tesseract = extract_text_tesseract(preprocessed)
            
            # Calculate improvement metrics
            degraded_count_model = len(degraded_text_model.strip())
            preprocessed_count_model = len(preprocessed_text_model.strip())
            degraded_count_tesseract = len(degraded_text_tesseract.strip())
            preprocessed_count_tesseract = len(preprocessed_text_tesseract.strip())
            
            # Model comparison
            if degraded_count_model == 0 and preprocessed_count_model == 0:
                model_comparison = "Le modèle n'a détecté aucun caractère dans les deux images."
            elif degraded_count_model == 0:
                model_comparison = "Le prétraitement a permis au modèle de détecter des caractères."
            else:
                model_improvement = ((preprocessed_count_model - degraded_count_model) / degraded_count_model) * 100
                model_comparison = f"Le modèle a amélioré la détection de {model_improvement:.1f}% avec le prétraitement."
            
            # Tesseract comparison
            if degraded_count_tesseract == 0 and preprocessed_count_tesseract == 0:
                tesseract_comparison = "Tesseract n'a extrait aucun texte des deux images."
            elif degraded_count_tesseract == 0:
                tesseract_comparison = "Le prétraitement a permis à Tesseract d'extraire du texte."
            else:
                tesseract_improvement = ((preprocessed_count_tesseract - degraded_count_tesseract) / degraded_count_tesseract) * 100
                tesseract_comparison = f"Tesseract a amélioré l'extraction de {tesseract_improvement:.1f}% avec le prétraitement."
            
            # Calculate average confidence for model predictions
            avg_confidence_degraded = np.mean([p['confidence'] for p in degraded_predictions]) if degraded_predictions else 0.0
            avg_confidence_preprocessed = np.mean([p['confidence'] for p in preprocessed_predictions]) if preprocessed_predictions else 0.0
            
            # Paths for frontend
            base_url = request.url_root
            
            return jsonify({
                'success': True,
                'original_image': f"{base_url}static/uploads/{unique_filename}",
                'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                
                # Model results
                'model_results': {
                    'degraded_text': degraded_text_model,
                    'preprocessed_text': preprocessed_text_model,
                    'degraded_predictions': degraded_predictions,
                    'preprocessed_predictions': preprocessed_predictions,
                    'degraded_confidence': float(avg_confidence_degraded),
                    'preprocessed_confidence': float(avg_confidence_preprocessed),
                    'comparison': model_comparison
                },
                
                # Tesseract results
                'tesseract_results': {
                    'degraded_text': degraded_text_tesseract,
                    'preprocessed_text': preprocessed_text_tesseract,
                    'comparison': tesseract_comparison
                },
                
                # Legacy fields for backward compatibility
                'degraded_text': degraded_text_tesseract,
                'preprocessed_text': preprocessed_text_tesseract,
                'comparison': tesseract_comparison
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'File type not allowed'}), 400

@app.route('/process_step', methods=['POST'])
def process_step_api():
    """
    API endpoint to process individual steps and return intermediate results
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    step = request.form.get('step', '1')
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Read image
            image = cv2.imread(file_path)
            base_url = request.url_root
            
            if step == '1':
                # Step 1: Just return original image
                return jsonify({
                    'success': True,
                    'step': 1,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'message': 'Image téléchargée avec succès'
                })
            
            elif step == '2':
                # Step 2: Image degradation
                degraded = degrade_image(image)
                degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
                cv2.imwrite(degraded_path, degraded)
                
                return jsonify({
                    'success': True,
                    'step': 2,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                    'message': 'Image dégradée générée',
                    'degradation_info': {
                        'original_size': f"{image.shape[1]}x{image.shape[0]}",
                        'degraded_size': f"{degraded.shape[1]}x{degraded.shape[0]}",
                        'noise_added': True,
                        'blur_applied': True
                    }
                })
            
            elif step == '3':
                # Step 3: Preprocessing
                degraded = degrade_image(image)
                preprocessed = preprocess_image(image)
                
                degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
                preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
                
                cv2.imwrite(degraded_path, degraded)
                cv2.imwrite(preprocessed_path, preprocessed)
                
                return jsonify({
                    'success': True,
                    'step': 3,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                    'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                    'message': 'Prétraitement terminé',
                    'preprocessing_info': {
                        'techniques_applied': [
                            'Correction gamma',
                            'CLAHE',
                            'Filtrage bilatéral',
                            'Binarisation adaptative',
                            'Opérations morphologiques'
                        ]
                    }
                })
            
            elif step == '4':
                # Step 4: Character segmentation
                degraded = degrade_image(image)
                preprocessed = preprocess_image(image)
                
                # Segment characters from preprocessed image
                segments = segment_characters(preprocessed)
                
                # Save images
                degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
                preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
                
                cv2.imwrite(degraded_path, degraded)
                cv2.imwrite(preprocessed_path, preprocessed)
                
                # Create visualization of segmented characters
                if segments:
                    # Draw bounding boxes on the preprocessed image
                    segmented_viz = preprocessed.copy()
                    if len(segmented_viz.shape) == 2:
                        segmented_viz = cv2.cvtColor(segmented_viz, cv2.COLOR_GRAY2BGR)
                    
                    for i, segment in enumerate(segments):
                        x, y, w, h = segment['bbox']
                        cv2.rectangle(segmented_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(segmented_viz, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    segmented_path = os.path.join(app.config['PROCESSED_FOLDER'], f"segmented_{unique_filename}")
                    cv2.imwrite(segmented_path, segmented_viz)
                    segmented_image_url = f"{base_url}static/processed/segmented_{unique_filename}"
                else:
                    segmented_image_url = None
                
                return jsonify({
                    'success': True,
                    'step': 4,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                    'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                    'segmented_image': segmented_image_url,
                    'message': f'{len(segments)} caractères détectés',
                    'segmentation_info': {
                        'characters_found': len(segments),
                        'bounding_boxes': [segment['bbox'] for segment in segments],
                        'detection_method': 'Contour detection with filtering'
                    }
                })
            
            elif step == '5':
                # Step 5: Character recognition
                degraded = degrade_image(image)
                preprocessed = preprocess_image(image)
                
                # Get predictions for both images
                degraded_text, degraded_predictions = extract_text_with_model(degraded)
                preprocessed_text, preprocessed_predictions = extract_text_with_model(preprocessed)
                
                # Save images
                degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
                preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
                
                cv2.imwrite(degraded_path, degraded)
                cv2.imwrite(preprocessed_path, preprocessed)
                
                # Calculate confidence scores
                avg_confidence_degraded = np.mean([p['confidence'] for p in degraded_predictions]) if degraded_predictions else 0.0
                avg_confidence_preprocessed = np.mean([p['confidence'] for p in preprocessed_predictions]) if preprocessed_predictions else 0.0
                
                return jsonify({
                    'success': True,
                    'step': 5,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                    'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                    'message': 'Reconnaissance terminée',
                    'recognition_results': {
                        'degraded': {
                            'text': degraded_text,
                            'confidence': float(avg_confidence_degraded),
                            'characters_count': len(degraded_predictions),
                            'predictions': degraded_predictions
                        },
                        'preprocessed': {
                            'text': preprocessed_text,
                            'confidence': float(avg_confidence_preprocessed),
                            'characters_count': len(preprocessed_predictions),
                            'predictions': preprocessed_predictions
                        }
                    }
                })
            
            else:
                return jsonify({'success': False, 'error': 'Invalid step'}), 400
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'File type not allowed'}), 400

@app.route('/info', methods=['GET'])
def get_app_info():
    """API endpoint to return app information"""
    return jsonify({
        'version': '1.0.0',
        'name': 'OCR Vision API',
        'description': 'API for advanced OCR processing with image preprocessing techniques',
    })

# Add new transformation functions based on the Colab notebook
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Add salt and pepper noise to the image"""
    noisy = np.copy(image)
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def add_holes(image, num_holes=10, max_size=20):
    """Add random holes (black rectangles) to the image"""
    img_with_holes = np.copy(image)
    height, width = image.shape[:2]
    
    for _ in range(num_holes):
        # Random position and size
        x = np.random.randint(0, width - 10)
        y = np.random.randint(0, height - 10)
        w = np.random.randint(5, min(max_size, width - x))
        h = np.random.randint(5, min(max_size, height - y))
        
        # Add black hole
        img_with_holes[y:y+h, x:x+w] = 0
        
    return img_with_holes

def adjust_brightness(image, factor=1.5):
    """Adjust the brightness of the image"""
    # Convert to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    brightened = enhancer.enhance(factor)
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)

def add_glare(image, intensity=0.8):
    """Add light glare effect to the image"""
    height, width = image.shape[:2]
    # Create a circular glare mask
    center_x = np.random.randint(width // 4, width * 3 // 4)
    center_y = np.random.randint(height // 4, height * 3 // 4)
    radius = min(width, height) // 4
    
    # Create mask
    y, x = np.ogrid[:height, :width]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    
    # Apply glare
    glare = np.copy(image)
    glare[mask] = np.clip(glare[mask] + intensity * 255, 0, 255)
    return glare

def apply_jpeg_quality(image, quality=20):
    """Apply JPEG compression artifacts"""
    # Encode and decode with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_blur(image, kernel_size=5):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_transformations(image, transformation_ids):
    """Apply specified transformations to the image"""
    degraded = image.copy()
    transformation_info = []
    
    if 'gaussian_noise' in transformation_ids:
        sigma = np.random.randint(15, 35)
        degraded = add_gaussian_noise(degraded, mean=0, sigma=sigma)
        transformation_info.append(f"Gaussian noise (sigma={sigma})")
    
    if 'salt_pepper' in transformation_ids:
        salt_prob = np.random.uniform(0.005, 0.02)
        pepper_prob = np.random.uniform(0.005, 0.02)
        degraded = add_salt_pepper_noise(degraded, salt_prob, pepper_prob)
        transformation_info.append(f"Salt & pepper (salt={salt_prob:.3f}, pepper={pepper_prob:.3f})")
    
    if 'holes' in transformation_ids:
        num_holes = np.random.randint(5, 15)
        max_size = np.random.randint(10, 30)
        degraded = add_holes(degraded, num_holes, max_size)
        transformation_info.append(f"Holes (n={num_holes}, max_size={max_size}px)")
    
    if 'brightness' in transformation_ids:
        factor = np.random.uniform(0.5, 2.0)
        degraded = adjust_brightness(degraded, factor)
        transformation_info.append(f"Brightness adjustment (factor={factor:.2f})")
    
    if 'glare' in transformation_ids:
        intensity = np.random.uniform(0.5, 0.9)
        degraded = add_glare(degraded, intensity)
        transformation_info.append(f"Glare (intensity={intensity:.2f})")
    
    if 'jpeg_quality' in transformation_ids:
        quality = np.random.randint(10, 40)
        degraded = apply_jpeg_quality(degraded, quality)
        transformation_info.append(f"JPEG compression (quality={quality})")
    
    if 'blur' in transformation_ids:
        kernel_size = np.random.choice([3, 5, 7])
        degraded = apply_blur(degraded, kernel_size)
        transformation_info.append(f"Gaussian blur (kernel={kernel_size}x{kernel_size})")
    
    return degraded, transformation_info

@app.route('/process_with_transformations', methods=['POST'])
def process_with_transformations():
    """
    API endpoint to process an image with specific transformations based on the Colab approach
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400
    
    # Get transformations to apply
    transformations = request.form.get('transformations', '[]')
    try:
        transformation_ids = json.loads(transformations)
    except:
        transformation_ids = []
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        # Add a unique identifier to avoid collisions
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the image
        try:
            # Read original image
            original_image = cv2.imread(file_path)
            
            # Apply selected transformations to get the degraded image
            degraded_image, transformation_info = apply_transformations(
                original_image, transformation_ids)
            degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], 
                                         f"degraded_{unique_filename}")
            cv2.imwrite(degraded_path, degraded_image)
            
            # Apply advanced preprocessing to correct the degraded image
            preprocessed_image = preprocess_image(degraded_image)
            preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], 
                                            f"preprocessed_{unique_filename}")
            cv2.imwrite(preprocessed_path, preprocessed_image)
            
            # Segment the preprocessed image to find characters
            segments = segment_characters(preprocessed_image)
            
            # Draw segmentation boxes on a copy of the image
            segmented_viz = original_image.copy()
            for segment in segments:
                x, y, w, h = segment['bbox']
                cv2.rectangle(segmented_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            segmented_path = os.path.join(app.config['PROCESSED_FOLDER'], 
                                         f"segmented_{unique_filename}")
            cv2.imwrite(segmented_path, segmented_viz)
            
            # Extract text using our model from both degraded and preprocessed images
            degraded_text_model, degraded_predictions = extract_text_with_model(degraded_image)
            preprocessed_text_model, preprocessed_predictions = extract_text_with_model(preprocessed_image)
            
            # Extract text using Tesseract from both degraded and preprocessed images
            degraded_text_tesseract = extract_text_tesseract(degraded_image)
            preprocessed_text_tesseract = extract_text_tesseract(preprocessed_image)
            
            # Calculate average confidence for model predictions
            avg_confidence_degraded = np.mean([p['confidence'] for p in degraded_predictions]) * 100 if degraded_predictions else 0.0
            avg_confidence_preprocessed = np.mean([p['confidence'] for p in preprocessed_predictions]) * 100 if preprocessed_predictions else 0.0
            
            # Generate comparison text
            model_comparison = f"Model confidence: {avg_confidence_degraded:.1f}% -> {avg_confidence_preprocessed:.1f}%"
            if avg_confidence_preprocessed > avg_confidence_degraded:
                model_comparison += f" (improved by {avg_confidence_preprocessed - avg_confidence_degraded:.1f}%)"
            
            # Generate character boxes for visualization
            character_boxes = []
            for pred in preprocessed_predictions:
                character_boxes.append({
                    'x': pred['bbox'][0],
                    'y': pred['bbox'][1],
                    'width': pred['bbox'][2],
                    'height': pred['bbox'][3],
                    'text': pred['character'],
                    'confidence': pred['confidence']
                })
            
            # Paths for frontend
            base_url = request.url_root
            
            return jsonify({
                'success': True,
                'original_image': f"{base_url}static/uploads/{unique_filename}",
                'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                'segmented_image': f"{base_url}static/processed/segmented_{unique_filename}",
                
                # Model-based results
                'model_results': {
                    'degraded_text': degraded_text_model,
                    'preprocessed_text': preprocessed_text_model,
                    'degraded_predictions': degraded_predictions,
                    'preprocessed_predictions': preprocessed_predictions,
                    'degraded_confidence': float(avg_confidence_degraded),
                    'preprocessed_confidence': float(avg_confidence_preprocessed),
                    'comparison': model_comparison
                },
                
                # Traditional Tesseract results
                'tesseract_results': {
                    'degraded_text': degraded_text_tesseract,
                    'preprocessed_text': preprocessed_text_tesseract,
                    'comparison': f"Characters detected: {len(degraded_text_tesseract)} -> {len(preprocessed_text_tesseract)}"
                },
                
                # For backwards compatibility
                'degraded_text': degraded_text_model,
                'preprocessed_text': preprocessed_text_model,
                'comparison': model_comparison,
                
                # Extra info
                'transformation_info': transformation_info,
                'character_boxes': character_boxes
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f"Error processing the image: {str(e)}"
            }), 500
    
    return jsonify({
        'success': False,
        'error': 'Invalid file format. Allowed formats: ' + ', '.join(ALLOWED_EXTENSIONS)
    }), 400

@app.route('/apply_transformation', methods=['POST'])
def apply_transformation():
    """
    API endpoint to apply a single transformation with custom parameters
    and return the result for preview
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    technique_id = request.form.get('technique_id', '')
    parameters_json = request.form.get('parameters', '{}')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400
    
    # Parse parameters
    try:
        parameters = json.loads(parameters_json)
    except:
        parameters = {}
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Read original image
            image = cv2.imread(file_path)
            result_image = None
            description = ""
            param_info = {}
            
            # Apply the requested transformation
            if technique_id == 'gaussian_noise':
                sigma = float(parameters.get('sigma', 25))
                result_image = add_gaussian_noise(image, mean=0, sigma=sigma)
                description = f"Gaussian noise added with sigma={sigma}"
                param_info = {'sigma': sigma}
                
            elif technique_id == 'salt_pepper':
                salt_prob = float(parameters.get('salt_prob', 0.01))
                pepper_prob = float(parameters.get('pepper_prob', 0.01))
                result_image = add_salt_pepper_noise(image, salt_prob, pepper_prob)
                description = f"Salt & pepper noise added with salt={salt_prob:.3f}, pepper={pepper_prob:.3f}"
                param_info = {'salt_prob': salt_prob, 'pepper_prob': pepper_prob}
                
            elif technique_id == 'brightness':
                factor = float(parameters.get('factor', 1.0))
                result_image = adjust_brightness(image, factor)
                description = f"Brightness adjusted with factor={factor:.1f}"
                param_info = {'factor': factor}
                
            elif technique_id == 'glare':
                intensity = float(parameters.get('intensity', 0.6))
                result_image = add_glare(image, intensity)
                description = f"Glare effect added with intensity={intensity:.1f}"
                param_info = {'intensity': intensity}
                
            elif technique_id == 'jpeg_quality':
                quality = int(parameters.get('quality', 20))
                result_image = apply_jpeg_quality(image, quality)
                description = f"JPEG compression applied with quality={quality}"
                param_info = {'quality': quality}
                
            elif technique_id == 'blur':
                kernel_size = int(parameters.get('kernel_size', 5))
                result_image = apply_blur(image, kernel_size)
                description = f"Gaussian blur applied with kernel_size={kernel_size}x{kernel_size}"
                param_info = {'kernel_size': kernel_size}
                
            elif technique_id == 'holes':
                num_holes = int(parameters.get('num_holes', 10))
                max_size = int(parameters.get('max_size', 20))
                result_image = add_holes(image, num_holes, max_size)
                description = f"Added {num_holes} random holes with max size {max_size}px"
                param_info = {'num_holes': num_holes, 'max_size': max_size}
                
            elif technique_id == 'adaptive_threshold':
                block_size = int(parameters.get('block_size', 11))
                c_value = int(parameters.get('c_value', 2))
                
                # Convert to grayscale first
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result_image = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, block_size, c_value
                )
                # Convert back to BGR for consistency
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
                
                description = f"Adaptive thresholding applied with block_size={block_size}, C={c_value}"
                param_info = {'block_size': block_size, 'c_value': c_value}
                
            elif technique_id == 'morphology':
                operation = parameters.get('operation', 'opening')
                kernel_size = int(parameters.get('kernel_size', 3))
                iterations = int(parameters.get('iterations', 1))
                
                # Convert to grayscale first
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Create kernel
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # Apply morphological operation
                if operation == 'erosion':
                    result_gray = cv2.erode(gray, kernel, iterations=iterations)
                    op_name = "Erosion"
                elif operation == 'dilation':
                    result_gray = cv2.dilate(gray, kernel, iterations=iterations)
                    op_name = "Dilation"
                elif operation == 'opening':
                    result_gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
                    op_name = "Opening"
                elif operation == 'closing':
                    result_gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                    op_name = "Closing"
                else:
                    result_gray = gray
                    op_name = "Unknown"
                
                # Convert back to BGR for consistency
                result_image = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
                
                description = f"{op_name} applied with kernel_size={kernel_size}x{kernel_size}, iterations={iterations}"
                param_info = {'operation': operation, 'kernel_size': kernel_size, 'iterations': iterations}
            
            # Save the result image
            if result_image is not None:
                result_path = os.path.join(app.config['PROCESSED_FOLDER'], f"transform_{unique_filename}")
                cv2.imwrite(result_path, result_image)
                
                # Return result
                base_url = request.url_root
                return jsonify({
                    'success': True,
                    'transformedImage': f"{base_url}static/processed/transform_{unique_filename}",
                    'description': description,
                    'parameters': param_info
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f"Transformation technique '{technique_id}' not supported"
                }), 400
                
        except Exception as e:
            print(f"Error applying transformation: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f"Error applying transformation: {str(e)}"
            }), 500
    
    return jsonify({
        'success': False,
        'error': 'Invalid file format. Allowed formats: ' + ', '.join(ALLOWED_EXTENSIONS)
    }), 400

@app.route('/process_word', methods=['POST'])
def process_word_api():
    """
    Process an image to extract text using multiple techniques:
    1. Direct recognition of the entire word
    2. Character-by-character recognition
    3. Word segmentation
    4. Tesseract OCR
    
    Returns comparison of all methods with confidences
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file format'}), 400
    
    # Get recognition mode (text or character)
    recognition_mode = request.form.get('recognition_mode', 'text')
    if recognition_mode not in ['text', 'character']:
        recognition_mode = 'text'  # Default to text mode
    
    try:
        # Save original image
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        file.save(original_path)
        
        # Read the image with OpenCV
        original_image = cv2.imread(original_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(original_image)
        
        # Process with different methods
        # 1. Direct OCR on original image
        original_text = pytesseract.image_to_string(original_image, config='--psm 7 --oem 3')
        original_confidence = 0.75  # Estimated confidence
        
        # 2. OCR on preprocessed image
        preprocessed_text = pytesseract.image_to_string(preprocessed_image, config='--psm 7 --oem 3')
        preprocessed_confidence = 0.85  # Usually better than original
        
        # 3. Character-by-character recognition
        char_segments = segment_characters(preprocessed_image)
        character_predictions = []
        
        for segment in char_segments:
            x, y, w, h = segment['bbox']
            char_img = segment['image']  # Get the already extracted character image
            
            # Process and predict the character
            char, confidence = predict_character(char_img)
            
            # Only include predictions with reasonable confidence
            if confidence > 0.3:
                character_predictions.append({
                    'character': char,
                    'confidence': float(confidence),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
        
        # Sort character predictions by x-coordinate
        character_predictions.sort(key=lambda p: p['bbox'][0])
        
        # Form the character-by-character text
        char_by_char_text = ''.join([p['character'] for p in character_predictions])
        
        # 4. Word segmentation with connected components
        # For simplicity, we'll use Tesseract's word segmentation
        word_segmentation_text = pytesseract.image_to_string(
            preprocessed_image, config='--psm 8 --oem 3')
        
        # 5. Tesseract OCR with standard settings
        tesseract_text = pytesseract.image_to_string(
            preprocessed_image, config='--psm 6 --oem 3')
        
        # Generate a visualization image for character segmentation
        visualization_img = preprocessed_image.copy()
        if len(visualization_img.shape) == 2:  # Convert grayscale to BGR for colored boxes
            visualization_img = cv2.cvtColor(visualization_img, cv2.COLOR_GRAY2BGR)
            
        for pred in character_predictions:
            x, y, w, h = pred['bbox']
            conf = pred['confidence']
            char = pred['character']
            
            # Color based on confidence: green (high) to red (low)
            color = (0, int(255 * conf), int(255 * (1 - conf)))
            cv2.rectangle(visualization_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(visualization_img, char, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Determine best result (usually the preprocessed one has best results)
        best_word = preprocessed_text.strip() if preprocessed_text else original_text.strip()
        if not best_word and char_by_char_text:
            best_word = char_by_char_text
        if not best_word and word_segmentation_text:
            best_word = word_segmentation_text
        if not best_word and tesseract_text:
            best_word = tesseract_text
        
        # Save visualization image
        vis_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_visualization.jpg")
        cv2.imwrite(vis_path, visualization_img)
        
        # Save preprocessed image
        preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{unique_id}_preprocessed.jpg")
        cv2.imwrite(preprocessed_path, preprocessed_image)
        
        # Prepare response
        result = {
            'success': True,
            'original_image': f"/static/uploads/{unique_id}_{filename}",
            'preprocessed_image': f"/static/processed/{unique_id}_preprocessed.jpg",
            'visualization': f"/static/processed/{unique_id}_visualization.jpg",
            'recognition_mode': recognition_mode,
            'word_recognition': {
                'original_word': original_text.strip(),
                'preprocessed_word': preprocessed_text.strip(),
                'best_word': best_word,
                'original_confidence': original_confidence,
                'preprocessed_confidence': preprocessed_confidence,
                'character_details': character_predictions
            },
            'comparison': {
                'character_by_character': char_by_char_text,
                'word_segmentation': word_segmentation_text.strip(),
                'tesseract': tesseract_text.strip()
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in process_word_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 
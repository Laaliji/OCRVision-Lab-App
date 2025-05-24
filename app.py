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
from flask import Flask, render_template, request, jsonify, url_for, flash, redirect
from werkzeug.utils import secure_filename
from flask_cors import CORS
import uuid
import string
import traceback

# Chemin vers l'exécutable Tesseract (vérifiez le chemin réel après installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Si l'installation est dans un autre emplacement, utilisez plutôt:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')
CORS(app)  # Activer CORS pour toutes les routes
app.secret_key = 'ocr_vision_secret_key'

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

def extract_text_as_words(image):
    """
    Extract text as complete words by first segmenting into words, then characters
    This improves OCR accuracy for text rather than isolated characters
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create a copy of the image for processing
        image_copy = image.copy()
        
        # Binarize the image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate the image to connect characters into words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Horizontal kernel for words
        dilated_words = cv2.dilate(binary, kernel, iterations=1)
        
        # Find word contours
        word_contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort words from left to right and top to bottom
        def sort_contours(cnts):
            # Sort by top-to-bottom first
            sorted_by_y = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
            
            # Group contours that are roughly on the same line
            y_groups = {}
            for cnt in sorted_by_y:
                x, y, w, h = cv2.boundingRect(cnt)
                found = False
                for group_y in y_groups.keys():
                    if abs(y - group_y) < h * 0.5:  # If within 50% of height, consider same line
                        y_groups[group_y].append(cnt)
                        found = True
                        break
                if not found:
                    y_groups[y] = [cnt]
            
            # Sort each line from left to right
            result = []
            for y in sorted(y_groups.keys()):
                line_cnts = sorted(y_groups[y], key=lambda c: cv2.boundingRect(c)[0])
                result.extend(line_cnts)
            
            return result
        
        word_contours = sort_contours(word_contours)
        
        # Process each word
        all_predictions = []
        recognized_text = ""
        
        for word_contour in word_contours:
            # Get word bounding box
            x, y, w, h = cv2.boundingRect(word_contour)
            
            # Skip very small contours (noise)
            if w < 10 or h < 10:
                continue
                
            # Extract word region
            word_image = image_copy[y:y+h, x:x+w]
            
            # Process for better character segmentation
            word_gray = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY) if len(word_image.shape) == 3 else word_image
            _, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Use a smaller kernel for character segmentation
            char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            word_binary = cv2.morphologyEx(word_binary, cv2.MORPH_CLOSE, char_kernel)
            
            # Find character contours
            char_contours, _ = cv2.findContours(word_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort characters from left to right
            char_contours = sorted(char_contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # Process each character in the word
            word_text = ""
            word_predictions = []
            
            for char_contour in char_contours:
                # Get character bounding box
                cx, cy, cw, ch = cv2.boundingRect(char_contour)
                
                # Skip very small contours (noise)
                if cw < 3 or ch < 3:
                    continue
                
                # Add padding around character
                padding = 2
                cx_start = max(0, cx - padding)
                cy_start = max(0, cy - padding)
                cx_end = min(word_binary.shape[1], cx + cw + padding)
                cy_end = min(word_binary.shape[0], cy + ch + padding)
                
                # Extract character region
                char_image = word_gray[cy_start:cy_end, cx_start:cx_end]
                
                if char_image.size == 0:
                    continue
                
                # Predict character
                char, confidence = predict_character(char_image)
                
                # Add prediction info
                word_predictions.append({
                    'character': char,
                    'confidence': confidence,
                    'bbox': [x + cx, y + cy, cw, ch]  # Global coordinates
                })
                
                # Add character to word text if confidence is high enough
                if confidence > 0.3:
                    word_text += char
                else:
                    word_text += '?'
            
            # Add predictions from this word
            all_predictions.extend(word_predictions)
            
            # Add word to recognized text with a space
            if word_text:
                recognized_text += word_text + " "
        
        # Remove trailing space
        recognized_text = recognized_text.strip()
        
        return recognized_text, all_predictions
    except Exception as e:
        print(f"Error in extract_text_as_words: {e}")
        traceback.print_exc()
        return "Error in word-based text recognition", []

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

def predict_word_from_image(image, target_size=(32, 32)):
    """
    Predict a word directly from an image containing multiple characters.
    This function segments the image into characters and recognizes each character
    in context of the word.
    
    Args:
        image (np.array): Image in BGR or grayscale format
        target_size (tuple): Size to resize each character image to
        
    Returns:
        str: Predicted word
        list: List of characters with their bounding boxes and confidence
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize the image (invert for black text on white background)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of the characters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        char_contours = []
        min_area = 50  # Minimum area for a character
        max_area = 5000  # Maximum area for a character
        min_width = 8
        min_height = 15
        max_aspect_ratio = 3.0  # width/height ratio
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter based on size and aspect ratio
            if (area >= min_area and area <= max_area and 
                w >= min_width and h >= min_height and
                w/h <= max_aspect_ratio):
                char_contours.append(contour)
        
        # Sort contours from left to right (reading order)
        bounding_boxes = [cv2.boundingRect(c) for c in char_contours]
        sorted_contours = [c for _, c in sorted(zip(bounding_boxes, char_contours), key=lambda b: b[0][0])]
        
        predicted_word = ""
        char_predictions = []
        
        # Process each character
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract the character region with padding
            padding = 2
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)
            
            char_img = gray[y_start:y_end, x_start:x_end]
            
            if char_img.size == 0:
                continue
            
            # Resize to target size
            resized = cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)
            
            # Prepare image for model prediction
            if len(resized.shape) == 2:  # Grayscale
                # Convert to RGB (3 channels) if model expects it
                char_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            else:
                char_rgb = resized
            
            # Normalize pixel values
            char_normalized = char_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            char_batch = np.expand_dims(char_normalized, axis=0)
            
            # Make prediction
            predictions = model.predict(char_batch, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Map class index to character
            predicted_char = CHARACTER_CLASSES[predicted_class]
            
            # Add to result
            predicted_word += predicted_char
            char_predictions.append({
                'character': predicted_char,
                'confidence': confidence,
                'bbox': [x, y, w, h]
            })
        
        return predicted_word, char_predictions
    except Exception as e:
        print(f"Error in predict_word_from_image: {e}")
        traceback.print_exc()
        return "", []

def image_to_base64(image):
    # Convert OpenCV image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    return render_template('process.html')

@app.route('/word_recognition', methods=['GET'])
def word_recognition():
    return render_template('word_recognition.html')

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
            degraded_text_words, degraded_predictions_words = extract_text_as_words(degraded)
            degraded_text_tesseract = extract_text_tesseract(degraded)
            
            # Preprocess and extract text with advanced techniques
            preprocessed = preprocess_image(image)
            preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
            cv2.imwrite(preprocessed_path, preprocessed)
            
            # Extract text using both methods for preprocessed image
            preprocessed_text_model, preprocessed_predictions = extract_text_with_model(preprocessed)
            preprocessed_text_words, preprocessed_predictions_words = extract_text_as_words(preprocessed)
            preprocessed_text_tesseract = extract_text_tesseract(preprocessed)
            
            # Calculate improvement metrics
            degraded_count_model = len(degraded_text_model.strip())
            preprocessed_count_model = len(preprocessed_text_model.strip())
            degraded_count_tesseract = len(degraded_text_tesseract.strip())
            preprocessed_count_tesseract = len(preprocessed_text_tesseract.strip())
            
            # Word-based metrics
            degraded_count_words = len(degraded_text_words.strip())
            preprocessed_count_words = len(preprocessed_text_words.strip())
            
            # Model comparison
            if degraded_count_model == 0 and preprocessed_count_model == 0:
                model_comparison = "Le modèle n'a détecté aucun caractère dans les deux images."
            elif degraded_count_model == 0:
                model_comparison = "Le prétraitement a permis au modèle de détecter des caractères."
            else:
                model_improvement = ((preprocessed_count_model - degraded_count_model) / degraded_count_model) * 100
                model_comparison = f"Le modèle a amélioré la détection de {model_improvement:.1f}% avec le prétraitement."
            
            # Words model comparison
            if degraded_count_words == 0 and preprocessed_count_words == 0:
                words_model_comparison = "Le modèle par mots n'a détecté aucun caractère dans les deux images."
            elif degraded_count_words == 0:
                words_model_comparison = "Le prétraitement a permis au modèle par mots de détecter des caractères."
            else:
                words_model_improvement = ((preprocessed_count_words - degraded_count_words) / degraded_count_words) * 100
                words_model_comparison = f"Le modèle par mots a amélioré la détection de {words_model_improvement:.1f}% avec le prétraitement."
            
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
            
            # Calculate average confidence for word-based model predictions
            avg_confidence_degraded_words = np.mean([p['confidence'] for p in degraded_predictions_words]) if degraded_predictions_words else 0.0
            avg_confidence_preprocessed_words = np.mean([p['confidence'] for p in preprocessed_predictions_words]) if preprocessed_predictions_words else 0.0
            
            # Paths for frontend
            base_url = request.url_root
            
            return jsonify({
                'success': True,
                'original_image': f"{base_url}static/uploads/{unique_filename}",
                'degraded_image': f"{base_url}static/processed/degraded_{unique_filename}",
                'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                
                # Character model results
                'model_results': {
                    'degraded_text': degraded_text_model,
                    'preprocessed_text': preprocessed_text_model,
                    'degraded_predictions': degraded_predictions,
                    'preprocessed_predictions': preprocessed_predictions,
                    'degraded_confidence': float(avg_confidence_degraded),
                    'preprocessed_confidence': float(avg_confidence_preprocessed),
                    'comparison': model_comparison
                },
                
                # Word-based model results
                'words_model_results': {
                    'degraded_text': degraded_text_words,
                    'preprocessed_text': preprocessed_text_words,
                    'degraded_predictions': degraded_predictions_words,
                    'preprocessed_predictions': preprocessed_predictions_words,
                    'degraded_confidence': float(avg_confidence_degraded_words),
                    'preprocessed_confidence': float(avg_confidence_preprocessed_words),
                    'comparison': words_model_comparison
                },
                
                # Tesseract results
                'tesseract_results': {
                    'degraded_text': degraded_text_tesseract,
                    'preprocessed_text': preprocessed_text_tesseract,
                    'comparison': tesseract_comparison
                },
                
                # Legacy fields for backward compatibility
                'degraded_text': degraded_text_words,  # Using word-based as default
                'preprocessed_text': preprocessed_text_words,  # Using word-based as default
                'comparison': words_model_comparison  # Using word-based as default
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
            
            elif step == '6':
                # Step 6: Word segmentation
                degraded = degrade_image(image)
                preprocessed = preprocess_image(image)
                
                # Save images
                degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
                preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
                
                cv2.imwrite(degraded_path, degraded)
                cv2.imwrite(preprocessed_path, preprocessed)
                
                # First convert to binary
                gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Dilate to connect characters into words
                word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                dilated_words = cv2.dilate(binary, word_kernel, iterations=1)
                
                # Find word contours
                word_contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter small contours
                filtered_word_contours = []
                for contour in word_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w >= 10 and h >= 10:
                        filtered_word_contours.append(contour)
                
                # Create visualization with word bounding boxes
                word_viz = preprocessed.copy()
                for i, word_contour in enumerate(filtered_word_contours):
                    x, y, w, h = cv2.boundingRect(word_contour)
                    cv2.rectangle(word_viz, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(word_viz, f"W{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Save word segmentation visualization
                word_viz_path = os.path.join(app.config['PROCESSED_FOLDER'], f"word_segmented_{unique_filename}")
                cv2.imwrite(word_viz_path, word_viz)
                
                # Create visualization with both word and character segmentation
                word_char_viz = preprocessed.copy()
                char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                char_count = 0
                
                for word_idx, word_contour in enumerate(filtered_word_contours):
                    x, y, w, h = cv2.boundingRect(word_contour)
                    
                    # Draw word boundary
                    cv2.rectangle(word_char_viz, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(word_char_viz, f"W{word_idx+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Extract word region
                    word_image = gray[y:y+h, x:x+w]
                    
                    # Binarize
                    _, word_binary = cv2.threshold(word_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # Apply morphological operations
                    word_binary = cv2.morphologyEx(word_binary, cv2.MORPH_CLOSE, char_kernel)
                    
                    # Find character contours
                    char_contours, _ = cv2.findContours(word_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Sort characters from left to right
                    char_contours = sorted(char_contours, key=lambda c: cv2.boundingRect(c)[0])
                    
                    # Draw character boundaries inside word
                    for char_idx, char_contour in enumerate(char_contours):
                        cx, cy, cw, ch = cv2.boundingRect(char_contour)
                        # Skip very small contours (noise)
                        if cw < 3 or ch < 3:
                            continue
                        # Draw character bounding box (global coordinates)
                        cv2.rectangle(word_char_viz, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 255, 0), 1)
                        cv2.putText(word_char_viz, f"C{char_count}", (x + cx, y + cy-2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        char_count += 1
                
                # Save word-character segmentation visualization
                word_char_viz_path = os.path.join(app.config['PROCESSED_FOLDER'], f"word_char_segmented_{unique_filename}")
                cv2.imwrite(word_char_viz_path, word_char_viz)
                
                # Process text with word-based approach
                preprocessed_text, preprocessed_predictions = extract_text_as_words(preprocessed)
                
                # Calculate confidence scores
                avg_confidence = np.mean([p['confidence'] for p in preprocessed_predictions]) if preprocessed_predictions else 0.0
                
                return jsonify({
                    'success': True,
                    'step': 6,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                    'word_segmented_image': f"{base_url}static/processed/word_segmented_{unique_filename}",
                    'word_char_segmented_image': f"{base_url}static/processed/word_char_segmented_{unique_filename}",
                    'message': f'{len(filtered_word_contours)} mots et {char_count} caractères détectés',
                    'word_segmentation_info': {
                        'words_found': len(filtered_word_contours),
                        'characters_found': char_count,
                        'text': preprocessed_text,
                        'confidence': float(avg_confidence),
                        'predictions': preprocessed_predictions
                    }
                })
            
            elif step == '7':
                # Step 7: Tesseract recognition and comparison
                degraded = degrade_image(image)
                preprocessed = preprocess_image(image)
                
                # Save images
                degraded_path = os.path.join(app.config['PROCESSED_FOLDER'], f"degraded_{unique_filename}")
                preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
                
                cv2.imwrite(degraded_path, degraded)
                cv2.imwrite(preprocessed_path, preprocessed)
                
                # Get text from all three methods
                char_text, char_predictions = extract_text_with_model(preprocessed)
                word_text, word_predictions = extract_text_as_words(preprocessed)
                tesseract_text = extract_text_tesseract(preprocessed)
                
                # Calculate confidence scores
                char_confidence = np.mean([p['confidence'] for p in char_predictions]) if char_predictions else 0.0
                word_confidence = np.mean([p['confidence'] for p in word_predictions]) if word_predictions else 0.0
                
                # Create a comparison visualization
                comparison_viz = np.ones((400, 800, 3), dtype=np.uint8) * 255
                
                # Add method titles
                cv2.putText(comparison_viz, "Comparison of OCR Methods", (250, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Add method 1: Character-by-character
                cv2.putText(comparison_viz, "1. Character-by-character:", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(comparison_viz, f"Text: {char_text[:50]}{'...' if len(char_text) > 50 else ''}", (70, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(comparison_viz, f"Confidence: {char_confidence:.2f}", (70, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add method 2: Word-based
                cv2.putText(comparison_viz, "2. Word-based approach:", (50, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(comparison_viz, f"Text: {word_text[:50]}{'...' if len(word_text) > 50 else ''}", (70, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(comparison_viz, f"Confidence: {word_confidence:.2f}", (70, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add method 3: Tesseract
                cv2.putText(comparison_viz, "3. Tesseract OCR:", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(comparison_viz, f"Text: {tesseract_text[:50]}{'...' if len(tesseract_text) > 50 else ''}", (70, 310), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add analysis
                cv2.putText(comparison_viz, "Analysis:", (50, 350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Determine which method has higher confidence
                if word_confidence > char_confidence:
                    best_method = "Word-based approach"
                    improvement = ((word_confidence - char_confidence) / char_confidence) * 100 if char_confidence > 0 else 100
                    analysis_text = f"Word-based is {improvement:.1f}% more confident"
                else:
                    best_method = "Character-by-character"
                    improvement = ((char_confidence - word_confidence) / word_confidence) * 100 if word_confidence > 0 else 100
                    analysis_text = f"Character-by-character is {improvement:.1f}% more confident"
                
                cv2.putText(comparison_viz, f"Best method: {best_method}", (70, 380), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(comparison_viz, analysis_text, (70, 410), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Save comparison visualization
                comparison_path = os.path.join(app.config['PROCESSED_FOLDER'], f"comparison_{unique_filename}")
                cv2.imwrite(comparison_path, comparison_viz)
                
                return jsonify({
                    'success': True,
                    'step': 7,
                    'original_image': f"{base_url}static/uploads/{unique_filename}",
                    'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                    'comparison_image': f"{base_url}static/processed/comparison_{unique_filename}",
                    'message': 'Reconnaissance complète avec plusieurs méthodes',
                    'recognition_results': {
                        'character_based': {
                            'text': char_text,
                            'confidence': float(char_confidence),
                            'characters_count': len(char_predictions)
                        },
                        'word_based': {
                            'text': word_text,
                            'confidence': float(word_confidence),
                            'characters_count': len(word_predictions)
                        },
                        'tesseract': {
                            'text': tesseract_text
                        },
                        'best_method': best_method
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
        'version': '1.2.0',
        'name': 'OCR Vision API',
        'description': 'API for advanced OCR processing with image preprocessing techniques',
        'features': [
            'Character-based OCR using CNN model',
            'Word-based OCR with contextual character recognition',
            'Direct word recognition with character-level visualization',
            'Tesseract OCR integration',
            'Image preprocessing with multiple techniques',
            'Image degradation simulation',
            'Step-by-step OCR process visualization',
            'Method comparison and analysis'
        ],
        'recognition_methods': {
            'character_based': 'Segments individual characters for recognition',
            'word_based': 'Segments text into words first, then characters, preserving context',
            'direct_word': 'Directly processes words with character-level recognition and visualization',
            'tesseract': 'Uses Tesseract OCR engine for comparison'
        },
        'endpoints': {
            '/process_image': 'Process image with multiple OCR methods',
            '/process_step': 'Step-by-step OCR process visualization',
            '/process_word': 'Direct word recognition with character visualization',
            '/process_with_transformations': 'Apply specific image transformations before OCR',
            '/apply_transformation': 'Apply individual transformations for testing',
            '/info': 'Get API information'
        },
        'steps': [
            'Image loading',
            'Image degradation',
            'Preprocessing',
            'Character segmentation',
            'Character recognition',
            'Word segmentation',
            'Word-based recognition',
            'Direct word recognition',
            'Method comparison'
        ]
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
    API endpoint to process an image and extract full words with direct word-based recognition
    or character-by-character recognition based on user selection
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400
    
    # Get the recognition mode (character or text)
    recognition_mode = request.form.get('recognition_mode', 'character')
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Read image
            original_image = cv2.imread(file_path)
            
            # Preprocess the image
            preprocessed = preprocess_image(original_image)
            preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"preprocessed_{unique_filename}")
            cv2.imwrite(preprocessed_path, preprocessed)
            
            # Process with different methods based on the selected mode
            if recognition_mode == 'text':
                # Text mode: Use direct word recognition
                original_word, original_chars = predict_word_from_image(original_image)
                preprocessed_word, preprocessed_chars = predict_word_from_image(preprocessed)
            else:
                # Character mode: Use character-by-character recognition
                char_text, char_predictions = extract_text_with_model(preprocessed)
                original_word = ''
                original_chars = []
                preprocessed_word = char_text
                preprocessed_chars = char_predictions
            
            # Always get other methods for comparison
            char_text, char_predictions = extract_text_with_model(preprocessed)
            word_text, word_predictions = extract_text_as_words(preprocessed)
            tesseract_text = extract_text_tesseract(preprocessed)
            
            # Create visualization of character segmentation
            visualization = original_image.copy()
            for char in preprocessed_chars:
                x, y, w, h = char['bbox']
                confidence = char['confidence']
                predicted_char = char['character']
                
                # Draw bounding box with color based on confidence
                # More green = higher confidence
                color = (0, int(confidence * 255), 0)
                cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
                
                # Add character label
                cv2.putText(visualization, predicted_char, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            viz_path = os.path.join(app.config['PROCESSED_FOLDER'], f"word_recognition_{unique_filename}")
            cv2.imwrite(viz_path, visualization)
            
            # Calculate confidence
            original_confidence = np.mean([c['confidence'] for c in original_chars]) if original_chars else 0.0
            preprocessed_confidence = np.mean([c['confidence'] for c in preprocessed_chars]) if preprocessed_chars else 0.0
            
            # Compare methods
            if preprocessed_confidence > original_confidence:
                best_word = preprocessed_word
                improvement = ((preprocessed_confidence - original_confidence) / original_confidence) * 100 if original_confidence > 0 else 100
                comparison = f"Preprocessing improved word recognition confidence by {improvement:.1f}%"
            else:
                best_word = original_word if original_word else preprocessed_word
                comparison = "Direct recognition performed better than with preprocessing"
            
            # Paths for frontend
            base_url = request.url_root
            
            return jsonify({
                'success': True,
                'original_image': f"{base_url}static/uploads/{unique_filename}",
                'preprocessed_image': f"{base_url}static/processed/preprocessed_{unique_filename}",
                'visualization': f"{base_url}static/processed/word_recognition_{unique_filename}",
                'recognition_mode': recognition_mode,
                
                # Word recognition results
                'word_recognition': {
                    'original_word': original_word,
                    'preprocessed_word': preprocessed_word,
                    'best_word': best_word,
                    'original_confidence': float(original_confidence),
                    'preprocessed_confidence': float(preprocessed_confidence),
                    'comparison': comparison,
                    'character_details': preprocessed_chars
                },
                
                # Results from other methods for comparison
                'comparison': {
                    'character_by_character': char_text,
                    'word_segmentation': word_text,
                    'tesseract': tesseract_text
                }
            })
            
        except Exception as e:
            print(f"Error in process_word_api: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080) 
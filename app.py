import os
import cv2
import numpy as np
import pytesseract
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify

# Chemin vers l'exécutable Tesseract (vérifiez le chemin réel après installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Si l'installation est dans un autre emplacement, utilisez plutôt:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')

# Configuration
UPLOAD_FOLDER = 'app/static/uploads'
PROCESSED_FOLDER = 'app/static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

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

def extract_text(image):
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

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save original image
        original_path = os.path.join(UPLOAD_FOLDER, 'original.jpg')
        file.save(original_path)
        
        # Read the image with OpenCV
        original_img = cv2.imread(original_path)
        
        # Degrade the image
        degraded_img = degrade_image(original_img)
        degraded_path = os.path.join(PROCESSED_FOLDER, 'degraded.jpg')
        cv2.imwrite(degraded_path, degraded_img)
        
        # Preprocess the degraded image
        preprocessed_img = preprocess_image(degraded_img)
        preprocessed_path = os.path.join(PROCESSED_FOLDER, 'preprocessed.jpg')
        cv2.imwrite(preprocessed_path, preprocessed_img)
        
        # Extract text from both degraded and preprocessed images
        degraded_text = extract_text(degraded_img)
        preprocessed_text = extract_text(preprocessed_img)
        
        # Convert images to base64 for frontend display
        original_base64 = image_to_base64(original_img)
        degraded_base64 = image_to_base64(degraded_img)
        preprocessed_base64 = image_to_base64(preprocessed_img)
        
        return jsonify({
            'original_image': original_base64,
            'degraded_image': degraded_base64,
            'preprocessed_image': preprocessed_base64,
            'degraded_text': degraded_text,
            'preprocessed_text': preprocessed_text
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080) 
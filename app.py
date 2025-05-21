import os
import cv2
import numpy as np
import pytesseract
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply bilateral filter for noise removal
    denoised = cv2.bilateralFilter(thresh, 9, 75, 75)
    
    # Optional: Enhance sharpness with convolution filter
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def extract_text(image):
    # Use Tesseract to extract text
    return pytesseract.image_to_string(image)

def image_to_base64(image):
    # Convert OpenCV image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
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
    app.run(debug=True) 
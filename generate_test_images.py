import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create directory for test images if it doesn't exist
os.makedirs('test_images', exist_ok=True)

def create_text_image(text, filename, font_size=60, font_color=(0, 0, 0), bg_color=(255, 255, 255), 
                     noise_level=0, blur_level=0, rotation=0, spacing=1):
    """
    Create an image with the given text
    
    Args:
        text (str): Text to render
        filename (str): Output filename
        font_size (int): Font size
        font_color (tuple): RGB color for text
        bg_color (tuple): RGB color for background
        noise_level (float): Salt and pepper noise level (0-1)
        blur_level (int): Gaussian blur kernel size (0 for no blur)
        rotation (float): Rotation angle in degrees
        spacing (int): Letter spacing multiplier
    """
    # Calculate image size based on text length and font size
    width = len(text) * font_size * spacing
    height = int(font_size * 2)
    
    # Create a blank image with background color
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use Arial font (Windows) or DejaVuSans (Linux)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
                font_size = 20  # Default font is smaller
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()
    
    # Draw text in the center of the image
    text_width = font_size * len(text) * spacing
    text_x = (width - text_width) // 2
    text_y = (height - font_size) // 2
    
    # Draw each character with proper spacing
    for i, char in enumerate(text):
        position = (text_x + i * font_size * spacing, text_y)
        draw.text(position, char, font=font, fill=font_color)
    
    # Apply rotation if specified
    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=True)
    
    # Convert to OpenCV format for further processing
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Apply Gaussian blur if specified
    if blur_level > 0:
        blur_level = blur_level if blur_level % 2 == 1 else blur_level + 1  # Must be odd
        img_cv = cv2.GaussianBlur(img_cv, (blur_level, blur_level), 0)
    
    # Apply salt and pepper noise if specified
    if noise_level > 0:
        noise = np.zeros(img_cv.shape, np.uint8)
        noise_amount = noise_level * 0.5  # Scale for reasonable noise
        
        # Salt (white) noise
        salt = np.random.random(img_cv.shape) < noise_amount
        noise[salt] = 255
        
        # Pepper (black) noise
        pepper = np.random.random(img_cv.shape) < noise_amount
        noise[pepper] = 0
        
        # Add noise to the image
        img_cv = cv2.add(img_cv, noise)
    
    # Save the image
    cv2.imwrite(f"test_images/{filename}", img_cv)
    print(f"Created image: test_images/{filename}")
    
    return img_cv

# Create test images with different text and properties

# Clean, simple uppercase text
create_text_image("KECH", "kech.png", font_size=80, spacing=1.2)

# Mixed case text with more characters
create_text_image("Abdessamad", "abdessamad.png", font_size=60, spacing=1.1)

# Uppercase text with slight noise
create_text_image("IAII", "iaii.png", font_size=80, noise_level=0.05, spacing=1.2)

# Text with spacing and rotation
create_text_image("OCR Test", "ocr_test.png", font_size=70, rotation=5, spacing=1.3)

# Text with blur effect
create_text_image("COMPUTER", "computer.png", font_size=75, blur_level=3, spacing=1.1)

# Text with high noise level
create_text_image("VISION", "vision.png", font_size=70, noise_level=0.15, spacing=1.2)

# Text with combined effects
create_text_image("AI PROJECT", "ai_project.png", font_size=65, 
                 noise_level=0.1, blur_level=3, rotation=3, spacing=1.25)

print("All test images created successfully!") 
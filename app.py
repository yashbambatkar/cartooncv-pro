from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import logging
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def resize_large_image(img, max_dimension=1200):
    """Resize image if too large"""
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

# ===== CARTOON FILTERS =====

def filter_standard_cartoon(img):
    """Standard cartoon effect"""
    img = resize_large_image(img)
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    
    data = img_smooth.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    img_quantized = result.reshape(img_smooth.shape)
    
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    cartoon = np.where(edges_colored == 255, 0, img_quantized)
    return cartoon

def filter_vibrant_cartoon(img):
    """More vibrant with stronger colors"""
    img = resize_large_image(img)
    img_smooth = cv2.bilateralFilter(img, 12, 100, 100)
    
    data = img_smooth.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    img_quantized = result.reshape(img_smooth.shape)
    
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    cartoon = np.where(edges_colored == 255, 0, img_quantized)
    
    # Enhance saturation
    hsv = cv2.cvtColor(cartoon.astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return cartoon

def filter_pencil_sketch(img):
    """Pencil sketch effect"""
    img = resize_large_image(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blurred = 255 - blurred
    
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    sketch_3channel = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    return sketch_3channel

def filter_oil_painting(img):
    """Oil painting effect"""
    img = resize_large_image(img)
    oil = cv2.xphoto.oilPainting(img, 7, 1)
    return oil

def filter_watercolor(img):
    """Watercolor effect"""
    img = resize_large_image(img)
    
    # Apply bilateral filter multiple times
    for i in range(6):
        img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Reduce colors
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    watercolor = result.reshape(img.shape)
    
    return watercolor

def filter_neon(img):
    """Neon effect with edge glow"""
    img = resize_large_image(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Create glow effect
    glow = cv2.GaussianBlur(edges_colored, (15, 15), 0)
    glow = cv2.multiply(glow, 2)
    
    # Enhance colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.5)
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    neon = cv2.add(img_bright, glow)
    return np.clip(neon, 0, 255).astype(np.uint8)

FILTERS = {
    'standard': filter_standard_cartoon,
    'vibrant': filter_vibrant_cartoon,
    'sketch': filter_pencil_sketch,
    'oil': filter_oil_painting,
    'watercolor': filter_watercolor,
    'neon': filter_neon
}

@app.route('/')
def home():
    """Serve main page"""
    return render_template('index_enhanced.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and apply filter"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        filter_type = request.form.get('filter', 'standard')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
        
        # Validate filter type
        if filter_type not in FILTERS:
            filter_type = 'standard'
        
        # Save and process
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        logger.info(f"Processing image: {filename} with filter: {filter_type}")
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Apply selected filter
        filter_func = FILTERS[filter_type]
        processed_img = filter_func(img)
        
        # Save output
        output_filename = f'{filter_type}_' + filename
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Convert if needed and save
        if len(processed_img.shape) == 2:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(output_path, processed_img)
        
        logger.info(f"Successfully processed: {output_filename}")
        
        return jsonify({
            'success': True,
            'output_path': f'/outputs/{output_filename}',
            'filename': output_filename,
            'filter_applied': filter_type
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/filters')
def get_filters():
    """Get available filters"""
    filters = {
        'standard': {
            'name': 'Standard Cartoon',
            'description': 'Classic cartoon effect with bold outlines',
            'icon': '🎨'
        },
        'vibrant': {
            'name': 'Vibrant',
            'description': 'Enhanced colors with stronger saturation',
            'icon': '🌈'
        },
        'sketch': {
            'name': 'Pencil Sketch',
            'description': 'Realistic pencil sketch effect',
            'icon': '✏️'
        },
        'oil': {
            'name': 'Oil Painting',
            'description': 'Classic oil painting style',
            'icon': '🖼️'
        },
        'watercolor': {
            'name': 'Watercolor',
            'description': 'Soft, artistic watercolor effect',
            'icon': '🎭'
        },
        'neon': {
            'name': 'Neon Glow',
            'description': 'Modern neon with glowing edges',
            'icon': '⚡'
        }
    }
    return jsonify(filters)

@app.route('/outputs/<filename>')
def download_file(filename):
    """Serve output images"""
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'available_filters': list(FILTERS.keys())
    }), 200

if __name__ == '__main__':
    logger.info("Starting CartoonCV Advanced Application")
    app.run(debug=True, host='0.0.0.0', port=5000)
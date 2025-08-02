#!/usr/bin/env python3
"""
Elephant Identification System - Backend Server
Handles both Siamese Network and YOLOv8 model inference
"""

import os
import sys
import json
import time
import uuid
import shutil
import zipfile
import logging
from pathlib import Path
from datetime import datetime

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import custom modules
try:
    from process_siamese import SiameseProcessor
    from process_yolo import YOLOProcessor
    from process_batch import BatchProcessor
    from utils.image_utils import load_and_preprocess_image, get_image_files
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the python-backend directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
MAX_CONTENT_LENGTH = 200 * 1024 * 1024 * 1024  # 200GB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global model instances
siamese_processor = None
yolo_processor = None
batch_processor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """Initialize all AI models"""
    global siamese_processor, yolo_processor, batch_processor

    logger.info("Initializing AI models...")

    try:
        # Initialize Siamese Network
        siamese_model_path = Path('models/siamese_best_model.pth')
        if siamese_model_path.exists():
            siamese_processor = SiameseProcessor(str(siamese_model_path))
            logger.info("✅ Siamese Network loaded successfully")
        else:
            logger.warning("⚠️ Siamese model not found at models/siamese_best_model.pth")

        # Initialize YOLOv8
        yolo_model_path = Path('models/yolo_best_model.pt')
        if yolo_model_path.exists():
            yolo_processor = YOLOProcessor(str(yolo_model_path))
            logger.info("✅ YOLOv8 model loaded successfully")
        else:
            logger.warning("⚠️ YOLOv8 model not found at models/yolo_best_model.pt")

        # Initialize batch processor
        batch_processor = BatchProcessor(siamese_processor, yolo_processor)
        logger.info("✅ Batch processor initialized")

    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        return False

    return True

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    try:
        info = {
            'siamese_status': 'Loaded' if siamese_processor else 'Not Available',
            'yolo_status': 'Loaded' if yolo_processor else 'Not Available',
            'dataset_size': getattr(siamese_processor, 'dataset_size', 0) if siamese_processor else 0,
            'gpu_available': torch.cuda.is_available(),
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
            'backend_version': '1.0.0',
            'models_ready': bool(siamese_processor or yolo_processor)
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-dataset', methods=['POST'])
def compare_with_dataset():
    """Compare uploaded image with Siamese network dataset"""
    if not siamese_processor:
        return jsonify({'error': 'Siamese network not available'}), 503

    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Get parameters
        threshold = float(request.form.get('threshold', 0.85))
        top_k = int(request.form.get('top_k', 10))

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(filepath)

        logger.info(f"Processing image: {filename}")

        # Process with Siamese network
        start_time = time.time()
        results = siamese_processor.compare_with_dataset(filepath, threshold, top_k)
        processing_time = time.time() - start_time

        # Clean up uploaded file
        os.remove(filepath)

        response = {
            'matches': results,
            'processing_time': round(processing_time, 2),
            'threshold_used': threshold,
            'total_comparisons': len(results)
        }

        logger.info(f"Siamese comparison completed in {processing_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in dataset comparison: {e}")
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-yolo', methods=['POST'])
def detect_with_yolo():
    """Detect elephants using YOLOv8 model"""
    if not yolo_processor:
        return jsonify({'error': 'YOLOv8 model not available'}), 503

    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Get parameters
        confidence = float(request.form.get('confidence', 0.5))
        iou_threshold = float(request.form.get('iou', 0.45))
        image_size = int(request.form.get('image_size', 640))

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(filepath)

        logger.info(f"Detecting elephants in: {filename}")

        # Process with YOLOv8
        start_time = time.time()
        results = yolo_processor.detect_elephants(
            filepath,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            image_size=image_size
        )
        processing_time = time.time() - start_time

        # Clean up uploaded file
        os.remove(filepath)

        response = {
            'detections': results['detections'],
            'annotated_image': results.get('annotated_image_base64'),
            'processing_time': round(processing_time, 2),
            'parameters': {
                'confidence_threshold': confidence,
                'iou_threshold': iou_threshold,
                'image_size': image_size
            }
        }

        logger.info(f"YOLOv8 detection completed in {processing_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in YOLOv8 detection: {e}")
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-batch', methods=['POST'])
def process_batch():
    """Process batch of images with selected model(s)"""
    if not batch_processor:
        return jsonify({'error': 'Batch processor not available'}), 503

    try:
        # Get parameters
        folder_path = request.form.get('folder_path')
        model_type = request.form.get('model_type', 'siamese')
        threshold = float(request.form.get('threshold', 0.85))
        max_groups = request.form.get('max_groups', '100')
        output_format = request.form.get('output_format', 'zip')

        if not folder_path or not os.path.exists(folder_path):
            return jsonify({'error': 'Invalid folder path'}), 400

        max_groups = int(max_groups) if max_groups != '0' else None

        logger.info(f"Starting batch processing: {folder_path}")
        logger.info(f"Model: {model_type}, Threshold: {threshold}, Max groups: {max_groups}")

        # Process batch
        start_time = time.time()
        results = batch_processor.process_folder(
            folder_path=folder_path,
            model_type=model_type,
            similarity_threshold=threshold,
            max_groups=max_groups,
            output_format=output_format
        )
        processing_time = time.time() - start_time

        # Generate download URL if file was created
        download_url = None
        if results.get('output_file'):
            download_filename = f"batch_results_{uuid.uuid4().hex[:8]}.zip"
            download_path = os.path.join(app.config['RESULTS_FOLDER'], download_filename)
            shutil.move(results['output_file'], download_path)
            download_url = f"/api/download/{download_filename}"

        response = {
            'total_images': results.get('total_images', 0),
            'groups_created': results.get('groups_created', 0),
            'processing_time': round(processing_time, 2),
            'accuracy': results.get('average_confidence', 0),
            'summary': results.get('summary', {}),
            'download_url': download_url,
            'model_used': model_type
        }

        logger.info(f"Batch processing completed in {processing_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed results"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'siamese': bool(siamese_processor),
            'yolo': bool(yolo_processor),
            'batch': bool(batch_processor)
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        current_time = time.time()

        # Clean upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                # Remove files older than 1 hour
                if current_time - os.path.getctime(filepath) > 3600:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old upload: {filename}")

        # Clean results folder
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            if os.path.isfile(filepath):
                # Remove files older than 24 hours
                if current_time - os.path.getctime(filepath) > 86400:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old result: {filename}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == '__main__':
    logger.info("🐘 Starting Elephant Identification System Backend v1.0.0")

    # Initialize models
    if not initialize_models():
        logger.error("❌ Failed to initialize models. Some features may not work.")

    # Clean up old files
    cleanup_temp_files()

    logger.info("🚀 Backend server starting on http://localhost:3001")

    try:
        app.run(
            host='127.0.0.1',
            port=3001,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("👋 Backend server shutting down gracefully")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

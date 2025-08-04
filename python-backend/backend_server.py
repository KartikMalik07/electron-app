#!/usr/bin/env python3
"""
Airavat - Backend Server v1.0.0
Enhanced startup with better error handling and diagnostics
"""

import os
import sys
import json
import time
import uuid
import shutil
import zipfile
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log', mode='w'),  # Overwrite log file each time
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False

    logger.info("✅ Python version compatible")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'torch', 'torchvision', 'cv2', 'PIL', 'numpy',
        'flask_cors', 'werkzeug'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            logger.info(f"✅ {package} - OK")
        except ImportError as e:
            missing_packages.append(package)
            logger.error(f"❌ {package} - MISSING")

    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    logger.info("✅ All required packages available")
    return True

def check_optional_packages():
    """Check optional packages and warn if missing"""
    optional_packages = {
        'ultralytics': 'YOLOv8 functionality',
        'efficientnet_pytorch': 'EfficientNet backbone for Siamese network'
    }

    for package, description in optional_packages.items():
        try:
            __import__(package)
            logger.info(f"✅ {package} - OK ({description})")
        except ImportError:
            logger.warning(f"⚠️  {package} - MISSING ({description})")

def setup_directories():
    """Create necessary directories"""
    directories = [
        'temp_uploads',
        'temp_results',
        'models',
        'dataset'
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Directory ready: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create {directory}: {e}")
            return False

    return True

def check_model_files():
    """Check if model files are available"""
    model_files = {
        'models/siamese_best_model.pth': 'Siamese Network model',
        'models/yolo_best_model.pt': 'YOLOv8 model'
    }

    available_models = 0

    for model_path, description in model_files.items():
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✅ {description}: {model_path} ({size_mb:.1f} MB)")
            available_models += 1
        else:
            logger.warning(f"⚠️  {description}: {model_path} - NOT FOUND")

    if available_models == 0:
        logger.warning("⚠️  No model files found - running in demo mode")
    else:
        logger.info(f"✅ {available_models}/{len(model_files)} model files available")

    return available_models > 0

# Now import Flask and other modules after checks
try:
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import torchvision.transforms as transforms

    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    from werkzeug.utils import secure_filename

    logger.info("✅ Core imports successful")
except ImportError as e:
    logger.error(f"❌ Critical import error: {e}")
    logger.error("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Import custom modules with error handling
try:
    from process_siamese import SiameseProcessor
    logger.info("✅ Siamese processor import successful")
except ImportError as e:
    logger.warning(f"⚠️  Siamese processor unavailable: {e}")
    SiameseProcessor = None

try:
    from process_yolo import YOLOProcessor
    logger.info("✅ YOLO processor import successful")
except ImportError as e:
    logger.warning(f"⚠️  YOLO processor unavailable: {e}")
    YOLOProcessor = None

try:
    from process_batch import BatchProcessor
    logger.info("✅ Batch processor import successful")
except ImportError as e:
    logger.warning(f"⚠️  Batch processor unavailable: {e}")
    BatchProcessor = None

try:
    from utils.image_utils import load_and_preprocess_image, get_image_files
    logger.info("✅ Image utilities import successful")
except ImportError as e:
    logger.warning(f"⚠️  Image utilities unavailable: {e}")

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

# Global model instances
siamese_processor = None
yolo_processor = None
batch_processor = None
startup_info = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """Initialize all AI models with enhanced error handling"""
    global siamese_processor, yolo_processor, batch_processor

    logger.info("🤖 Initializing AI models...")
    models_loaded = 0

    try:
        # Initialize Siamese Network
        if SiameseProcessor:
            siamese_model_path = Path('models/siamese_best_model.pth')
            if siamese_model_path.exists():
                try:
                    siamese_processor = SiameseProcessor(str(siamese_model_path))
                    logger.info("✅ Siamese Network loaded successfully")
                    models_loaded += 1
                except Exception as e:
                    logger.error(f"❌ Failed to load Siamese model: {e}")
            else:
                logger.warning("⚠️  Siamese model not found - creating demo processor")
                try:
                    siamese_processor = SiameseProcessor(None)  # Demo mode
                    logger.info("✅ Siamese Network running in demo mode")
                    models_loaded += 1
                except Exception as e:
                    logger.error(f"❌ Failed to create demo Siamese processor: {e}")
        else:
            logger.warning("⚠️  Siamese processor class not available")

        # Initialize YOLOv8
        if YOLOProcessor:
            yolo_model_path = Path('models/yolo_best_model.pt')
            if yolo_model_path.exists():
                try:
                    yolo_processor = YOLOProcessor(str(yolo_model_path))
                    logger.info("✅ YOLOv8 model loaded successfully")
                    models_loaded += 1
                except Exception as e:
                    logger.error(f"❌ Failed to load YOLOv8 model: {e}")
            else:
                logger.warning("⚠️  YOLOv8 model not found - trying default model")
                try:
                    yolo_processor = YOLOProcessor(None)  # Will use default YOLOv8n
                    logger.info("✅ YOLOv8 running with default model")
                    models_loaded += 1
                except Exception as e:
                    logger.error(f"❌ Failed to load default YOLOv8: {e}")
        else:
            logger.warning("⚠️  YOLO processor class not available")

        # Initialize batch processor
        if BatchProcessor:
            try:
                batch_processor = BatchProcessor(siamese_processor, yolo_processor)
                logger.info("✅ Batch processor initialized")
                models_loaded += 1
            except Exception as e:
                logger.error(f"❌ Failed to initialize batch processor: {e}")
        else:
            logger.warning("⚠️  Batch processor class not available")

    except Exception as e:
        logger.error(f"❌ Critical error during model initialization: {e}")
        logger.error(traceback.format_exc())

    logger.info(f"🎯 Model initialization complete: {models_loaded} components loaded")
    return models_loaded > 0

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app_name': 'Airavat',
            'version': '1.0.0',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'models_loaded': {
                'siamese': siamese_processor is not None,
                'yolo': yolo_processor is not None,
                'batch': batch_processor is not None
            },
            'system_info': {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'startup_info': startup_info
        }

        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

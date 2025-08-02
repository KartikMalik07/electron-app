#!/bin/bash

# Elephant Identification System v1.0.0 - Startup Script
# This script handles environment setup and launches the application

echo "🐘 Starting Elephant Identification System v1.0.0"
echo "================================================"

# Set environment variables for graphics compatibility
export LIBGL_ALWAYS_SOFTWARE=1
export ELECTRON_DISABLE_GPU=1
export ELECTRON_NO_SANDBOX=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
check_nodejs() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js found: $NODE_VERSION"

        # Check if version is >= 16
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1 | cut -d'v' -f2)
        if [ "$NODE_MAJOR" -lt 16 ]; then
            print_error "Node.js version 16 or higher required. Found: $NODE_VERSION"
            exit 1
        fi
    else
        print_error "Node.js not found. Please install Node.js 16+ from https://nodejs.org/"
        exit 1
    fi
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_status "Python found: $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version)
        print_status "Python found: $PYTHON_VERSION"
    else
        print_error "Python not found. Please install Python 3.8+ from https://python.org/"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."

    # Check available memory
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -g | awk 'NR==2{printf "%.0f", $2}')
        print_status "Available RAM: ${TOTAL_MEM}GB"

        if [ "$TOTAL_MEM" -lt 8 ]; then
            print_warning "Warning: Less than 8GB RAM detected. Application may run slowly."
        fi
    fi

    # Check available disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    print_status "Available disk space: $AVAILABLE_SPACE"

    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected - CUDA acceleration will be available"
    else
        print_warning "No NVIDIA GPU detected - will use CPU processing"
    fi
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python environment..."

    cd python-backend

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        print_status "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        exit 1
    fi

    cd ..
}

# Setup Node.js dependencies
setup_nodejs_env() {
    print_status "Setting up Node.js environment..."

    # Install npm dependencies
    npm install

    if [ $? -eq 0 ]; then
        print_status "Node.js dependencies installed successfully"
    else
        print_error "Failed to install Node.js dependencies"
        exit 1
    fi
}

# Check for model files
check_model_files() {
    print_status "Checking for model files..."

    SIAMESE_MODEL="python-backend/models/siamese_best_model.pth"
    YOLO_MODEL="python-backend/models/yolo_best_model.pt"

    if [ -f "$SIAMESE_MODEL" ]; then
        print_status "Siamese model found: $SIAMESE_MODEL"
    else
        print_warning "Siamese model not found at $SIAMESE_MODEL"
        print_warning "Please download the model from GitHub releases and place it in the models folder"
    fi

    if [ -f "$YOLO_MODEL" ]; then
        print_status "YOLOv8 model found: $YOLO_MODEL"
    else
        print_warning "YOLOv8 model not found at $YOLO_MODEL"
        print_warning "Please download the model from GitHub releases and place it in the models folder"
    fi
}

# Start the application
start_application() {
    print_status "Starting Elephant Identification System..."

    # Start Python backend in background
    print_status "Starting Python backend server..."
    cd python-backend
    source venv/bin/activate
    python backend_server.py &
    BACKEND_PID=$!
    cd ..

    # Wait a moment for backend to start
    sleep 3

    # Start Electron frontend
    print_status "Starting Electron frontend..."
    npm start &
    FRONTEND_PID=$!

    # Function to cleanup on exit
    cleanup() {
        print_status "Shutting down application..."
        if [ ! -z "$BACKEND_PID" ]; then
            kill $BACKEND_PID 2>/dev/null
        fi
        if [ ! -z "$FRONTEND_PID" ]; then
            kill $FRONTEND_PID 2>/dev/null
        fi
        exit 0
    }

    # Set trap to cleanup on script exit
    trap cleanup SIGINT SIGTERM

    print_status "Application started successfully!"
    print_status "Backend PID: $BACKEND_PID"
    print_status "Frontend PID: $FRONTEND_PID"
    print_status ""
    print_status "🎉 Elephant Identification System is now running!"
    print_status "📱 The application window should open automatically"
    print_status "🌐 If the window doesn't open, try: http://localhost:3001"
    print_status ""
    print_status "Press Ctrl+C to stop the application"
    print_status ""

    # Wait for processes to complete
    wait
}

# Main execution flow
main() {
    echo ""
    print_status "Initializing Elephant Identification System..."
    echo ""

    # Run all checks and setup
    check_nodejs
    check_python
    check_system_requirements

    # Setup environments (only if needed)
    if [ ! -d "node_modules" ] || [ ! -d "python-backend/venv" ]; then
        print_status "Setting up development environment..."
        setup_nodejs_env
        setup_python_env
    else
        print_status "Development environment already set up"
    fi

    # Check for model files
    check_model_files

    # Start the application
    start_application
}

# Help function
show_help() {
    echo "🐘 Elephant Identification System v1.0.0"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --setup        Only setup environment, don't start app"
    echo "  --check        Only check system requirements"
    echo "  --clean        Clean all dependencies and caches"
    echo ""
    echo "Examples:"
    echo "  $0              # Start the application"
    echo "  $0 --setup      # Setup environment only"
    echo "  $0 --check      # Check system requirements"
    echo "  $0 --clean      # Clean installation"
    echo ""
}

# Clean installation
clean_installation() {
    print_status "Cleaning installation..."

    # Remove Node.js dependencies
    if [ -d "node_modules" ]; then
        print_status "Removing Node.js dependencies..."
        rm -rf node_modules
    fi

    # Remove Python virtual environment
    if [ -d "python-backend/venv" ]; then
        print_status "Removing Python virtual environment..."
        rm -rf python-backend/venv
    fi

    # Remove temporary files
    if [ -d "python-backend/temp_uploads" ]; then
        rm -rf python-backend/temp_uploads
    fi

    if [ -d "python-backend/temp_results" ]; then
        rm -rf python-backend/temp_results
    fi

    # Remove log files
    if [ -f "python-backend/backend.log" ]; then
        rm python-backend/backend.log
    fi

    print_status "Cleanup completed!"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --setup)
        check_nodejs
        check_python
        setup_nodejs_env
        setup_python_env
        print_status "Setup completed successfully!"
        exit 0
        ;;
    --check)
        check_nodejs
        check_python
        check_system_requirements
        check_model_files
        print_status "System check completed!"
        exit 0
        ;;
    --clean)
        clean_installation
        exit 0
        ;;
    "")
        # No arguments, run main
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

#!/bin/bash

# Airavat v1.0.0 - Enhanced Startup Script
# Comprehensive environment setup and error handling

echo "🐘 Starting Airavat v1.0.0"
echo "==========================="

# Set environment variables for better compatibility
export LIBGL_ALWAYS_SOFTWARE=1
export ELECTRON_DISABLE_GPU=1
export ELECTRON_NO_SANDBOX=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
PYTHON_CMD=""
NODE_CMD="node"
NPM_CMD="npm"
BACKEND_PID=""
FRONTEND_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Enhanced Python detection
find_python() {
    local python_commands=("python3" "python" "py")

    print_status "Searching for Python installation..."

    for cmd in "${python_commands[@]}"; do
        if command_exists "$cmd"; then
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
            local major=$(echo "$version" | cut -d. -f1)
            local minor=$(echo "$version" | cut -d. -f2)

            print_debug "Found $cmd: version $version"

            # Check if version is >= 3.8
            if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
                PYTHON_CMD="$cmd"
                print_success "Using Python: $cmd (version $version)"
                return 0
            elif [ "$major" -gt 3 ]; then
                PYTHON_CMD="$cmd"
                print_success "Using Python: $cmd (version $version)"
                return 0
            else
                print_warning "$cmd version $version is too old (need 3.8+)"
            fi
        fi
    done

    print_error "No suitable Python installation found"
    print_error "Please install Python 3.8+ from https://python.org/"
    return 1
}

# Check Node.js
check_nodejs() {
    print_status "Checking Node.js installation..."

    if command_exists node; then
        NODE_VERSION=$(node --version)
        print_debug "Found Node.js: $NODE_VERSION"

        # Extract major version number
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1 | cut -d'v' -f2)
        if [ "$NODE_MAJOR" -lt 16 ]; then
            print_error "Node.js version 16+ required. Found: $NODE_VERSION"
            print_error "Please update Node.js from https://nodejs.org/"
            return 1
        fi

        print_success "Node.js OK: $NODE_VERSION"
    else
        print_error "Node.js not found"
        print_error "Please install Node.js 16+ from https://nodejs.org/"
        return 1
    fi

    if command_exists npm; then
        NPM_VERSION=$(npm --version)
        print_success "npm OK: v$NPM_VERSION"
    else
        print_error "npm not found (should come with Node.js)"
        return 1
    fi

    return 0
}

# Enhanced system requirements check
check_system_requirements() {
    print_status "Checking system requirements..."

    # Memory check
    if command_exists free; then
        TOTAL_MEM=$(free -g | awk 'NR==2{printf "%.0f", $2}')
        AVAILABLE_MEM=$(free -g | awk 'NR==2{printf "%.0f", $7}')
        print_debug "System Memory: ${TOTAL_MEM}GB total, ${AVAILABLE_MEM}GB available"

        if [ "$TOTAL_MEM" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected (${TOTAL_MEM}GB). Performance may be limited."
        else
            print_success "Memory OK: ${TOTAL_MEM}GB RAM"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS memory check
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($0/1024/1024/1024)}')
        print_debug "System Memory: ${TOTAL_MEM}GB"
        if [ "$TOTAL_MEM" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be limited."
        else
            print_success "Memory OK: ${TOTAL_MEM}GB RAM"
        fi
    fi

    # Disk space check
    AVAILABLE_SPACE=$(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    print_debug "Available disk space: $AVAILABLE_SPACE"

    # GPU check
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
        export CUDA_VISIBLE_DEVICES=0
    else
        print_warning "No NVIDIA GPU detected - will use CPU processing"
    fi

    # Check available ports
    if command_exists netstat; then
        if netstat -tuln | grep -q ":3001 "; then
            print_warning "Port 3001 appears to be in use. Backend may fail to start."
        else
            print_debug "Port 3001 available"
        fi
    fi
}

# Enhanced Python environment setup
setup_python_env() {
    print_status "Setting up Python environment..."

    cd python-backend || {
        print_error "python-backend directory not found"
        return 1
    }

    # Check if we're already in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_debug "Already in virtual environment: $VIRTUAL_ENV"
    else
        # Create virtual environment if it doesn't exist
        if [ ! -d "venv" ]; then
            print_status "Creating Python virtual environment..."
            $PYTHON_CMD -m venv venv
            if [ $? -ne 0 ]; then
                print_error "Failed to create virtual environment"
                cd ..
                return 1
            fi
        fi

        # Activate virtual environment
        print_debug "Activating virtual environment..."
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source venv/Scripts/activate
        else
            source venv/bin/activate
        fi
    fi

    # Upgrade pip
    print_debug "Upgrading pip..."
    python -m pip install --upgrade pip --quiet

    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in python-backend directory"
        cd ..
        return 1
    fi

    # Install requirements with progress
    print_status "Installing Python dependencies (this may take a while)..."

    # Count number of packages
    TOTAL_PACKAGES=$(grep -c -v '^\s*#' requirements.txt | grep -c -v '^\s*$')
    print_debug "Installing $TOTAL_PACKAGES packages..."

    # Install with verbose output in debug mode
    if [[ "${DEBUG:-}" == "1" ]]; then
        pip install -r requirements.txt
    else
        pip install -r requirements.txt --quiet --progress-bar off
    fi

    if [ $? -eq 0 ]; then
        print_success "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        print_error "Try running with DEBUG=1 for more details"
        cd ..
        return 1
    fi

    cd ..
    return 0
}

# Enhanced Node.js environment setup
setup_nodejs_env() {
    print_status "Setting up Node.js environment..."

    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        print_error "package.json not found"
        return 1
    fi

    # Install dependencies
    print_status "Installing Node.js dependencies..."
    if [[ "${DEBUG:-}" == "1" ]]; then
        npm install
    else
        npm install --silent
    fi

    if [ $? -eq 0 ]; then
        print_success "Node.js dependencies installed successfully"
    else
        print_error "Failed to install Node.js dependencies"
        return 1
    fi

    return 0
}

# Check for model files with detailed info
check_model_files() {
    print_status "Checking for AI model files..."

    local models_dir="python-backend/models"
    local models_found=0

    # Create models directory if it doesn't exist
    mkdir -p "$models_dir"

    # Check Siamese model
    local siamese_model="$models_dir/siamese_best_model.pth"
    if [ -f "$siamese_model" ]; then
        local size=$(du -h "$siamese_model" | cut -f1)
        print_success "Siamese model found: $size"
        models_found=$((models_found + 1))
    else
        print_warning "Siamese model not found: $siamese_model"
        print_warning "Application will run in demo mode for Siamese network"
    fi

    # Check YOLO model
    local yolo_model="$models_dir/yolo_best_model.pt"
    if [ -f "$yolo_model" ]; then
        local size=$(du -h "$yolo_model" | cut -f1)
        print_success "YOLOv8 model found: $size"
        models_found=$((models_found + 1))
    else
        print_warning "YOLOv8 model not found: $yolo_model"
        print_warning "Application will use default YOLOv8 model"
    fi

    if [ $models_found -eq 0 ]; then
        print_warning "No custom model files found"
        print_warning "For full functionality, download models from:"
        print_warning "https://github.com/yourusername/electron-app/releases"
    else
        print_success "$models_found/2 model files available"
    fi

    return 0
}

# Test backend connectivity
test_backend_connection() {
    print_status "Testing backend connection..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if command_exists curl; then
            if curl -s http://localhost:3001/api/health >/dev/null 2>&1; then
                print_success "Backend is responding"
                return 0
            fi
        elif command_exists wget; then
            if wget -q --spider http://localhost:3001/api/health 2>/dev/null; then
                print_success "Backend is responding"
                return 0
            fi
        else
            # Fallback: try to connect with Python
            if $PYTHON_CMD -c "
import urllib.request
try:
    urllib.request.urlopen('http://localhost:3001/api/health', timeout=1)
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
                print_success "Backend is responding"
                return 0
            fi
        fi

        print_debug "Backend not ready yet (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done

    print_warning "Backend may not be fully ready, but continuing..."
    return 1
}

# Start Python backend with enhanced monitoring
start_python_backend() {
    print_status "Starting Python backend server..."

    cd python-backend || {
        print_error "Failed to enter python-backend directory"
        return 1
    }

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        print_debug "Activating Python virtual environment..."
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source venv/Scripts/activate
        else
            source venv/bin/activate
        fi
    fi

    # Check if backend script exists
    if [ ! -f "backend_server.py" ]; then
        print_error "backend_server.py not found"
        cd ..
        return 1
    fi

    # Start backend in background
    print_debug "Launching backend_server.py..."

    if [[ "${DEBUG:-}" == "1" ]]; then
        # Debug mode: show output
        python backend_server.py &
        BACKEND_PID=$!
    else
        # Normal mode: redirect output to log file
        python backend_server.py > backend_startup.log 2>&1 &
        BACKEND_PID=$!
    fi

    cd ..

    if [ -z "$BACKEND_PID" ]; then
        print_error "Failed to start Python backend"
        return 1
    fi

    print_debug "Backend started with PID: $BACKEND_PID"

    # Wait a moment for backend to initialize
    sleep 5

    # Check if process is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process died shortly after startup"
        if [ -f "python-backend/backend_startup.log" ]; then
            print_error "Check python-backend/backend_startup.log for details"
        fi
        return 1
    fi

    # Test connection
    test_backend_connection

    return 0
}

# Start Electron frontend
start_electron_frontend() {
    print_status "Starting Electron frontend..."

    # Check if main entry point exists
    if [ ! -f "src/main.js" ]; then
        print_error "src/main.js not found"
        return 1
    fi

    # Start Electron
    print_debug "Launching Electron application..."

    if [[ "${DEBUG:-}" == "1" ]]; then
        # Debug mode: show output
        npm start &
        FRONTEND_PID=$!
    else
        # Normal mode: minimal output
        npm start > frontend_startup.log 2>&1 &
        FRONTEND_PID=$!
    fi

    if [ -z "$FRONTEND_PID" ]; then
        print_error "Failed to start Electron frontend"
        return 1
    fi

    print_debug "Frontend started with PID: $FRONTEND_PID"
    return 0
}

# Cleanup function
cleanup() {
    print_status "Shutting down application..."

    # Stop frontend
    if [ ! -z "$FRONTEND_PID" ]; then
        print_debug "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null

        # Wait for graceful shutdown
        sleep 2

        # Force kill if still running
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_debug "Force killing frontend..."
            kill -9 $FRONTEND_PID 2>/dev/null
        fi
    fi

    # Stop backend
    if [ ! -z "$BACKEND_PID" ]; then
        print_debug "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null

        # Wait for graceful shutdown
        sleep 3

        # Force kill if still running
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_debug "Force killing backend..."
            kill -9 $BACKEND_PID 2>/dev/null
        fi
    fi

    # Clean up log files if not in debug mode
    if [[ "${DEBUG:-}" != "1" ]]; then
        rm -f backend_startup.log frontend_startup.log
    fi

    print_success "Application shutdown complete"
    exit 0
}

# Main application startup
start_application() {
    print_status "🚀 Starting Airavat application..."

    # Set trap for cleanup on exit
    trap cleanup SIGINT SIGTERM EXIT

    # Start backend
    if ! start_python_backend; then
        print_error "Failed to start Python backend"
        return 1
    fi

    # Start frontend
    if ! start_electron_frontend; then
        print_error "Failed to start Electron frontend"
        return 1
    fi

    # Display success information
    echo ""
    print_success "🎉 Airavat is now running!"
    print_success "📱 The application window should open automatically"
    print_success "🌐 Backend API available at: http://localhost:3001"
    print_success "🖥️  Frontend PID: $FRONTEND_PID"
    print_success "🐍 Backend PID: $BACKEND_PID"
    echo ""
    print_status "📊 Application Status:"
    print_status "  • Frontend: Running"
    print_status "  • Backend: Running"
    print_status "  • API Endpoint: http://localhost:3001/api/health"
    echo ""
    print_status "Press Ctrl+C to stop the application"
    print_status "Check logs with: tail -f python-backend/backend.log"
    echo ""

    # Monitor processes
    while true; do
        # Check if backend is still running
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            print_error "Backend process has stopped unexpectedly"
            if [ -f "python-backend/backend.log" ]; then
                print_error "Last few lines from backend log:"
                tail -5 python-backend/backend.log
            fi
            break
        fi

        # Check if frontend is still running
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            print_warning "Frontend process has stopped"
            break
        fi

        sleep 5
    done

    return 0
}

# Setup only mode
setup_only() {
    print_status "Setting up development environment only..."

    if ! find_python; then
        return 1
    fi

    if ! check_nodejs; then
        return 1
    fi

    if ! setup_python_env; then
        return 1
    fi

    if ! setup_nodejs_env; then
        return 1
    fi

    check_model_files

    print_success "Environment setup completed!"
    print_status "Run './start.sh' to start the application"
    return 0
}

# System check only mode
system_check() {
    print_status "Running system diagnostics..."

    find_python
    check_nodejs
    check_system_requirements
    check_model_files

    # Test Python imports
    print_status "Testing Python imports..."
    cd python-backend 2>/dev/null || {
        print_error "python-backend directory not found"
        return 1
    }

    if [ -d "venv" ]; then
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source venv/Scripts/activate
        else
            source venv/bin/activate
        fi
    fi

    python -c "
import sys
print(f'Python version: {sys.version}')

# Test critical imports
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV: {e}')

try:
    from flask import Flask
    print('✅ Flask: Available')
except ImportError as e:
    print(f'❌ Flask: {e}')

try:
    from PIL import Image
    print('✅ Pillow: Available')
except ImportError as e:
    print(f'❌ Pillow: {e}')
"

    cd ..
    print_success "System check completed!"
    return 0
}

# Clean installation
clean_installation() {
    print_status "Cleaning installation..."

    # Stop any running processes
    cleanup 2>/dev/null || true

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
    print_status "Cleaning temporary files..."
    rm -rf python-backend/temp_uploads
    rm -rf python-backend/temp_results
    rm -rf python-backend/__pycache__
    find python-backend -name "*.pyc" -delete 2>/dev/null || true
    find python-backend -name "*.pyo" -delete 2>/dev/null || true

    # Remove log files
    rm -f python-backend/backend.log
    rm -f python-backend/backend_startup.log
    rm -f frontend_startup.log

    # Remove build artifacts
    rm -rf dist
    rm -rf build

    print_success "Cleanup completed!"
    return 0
}

# Help function
show_help() {
    echo "🐘 Airavat v1.0.0 - AI Elephant Identification System"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo "  --setup          Setup environment only (don't start app)"
    echo "  --check          Run system diagnostics only"
    echo "  --clean          Clean all dependencies and temporary files"
    echo "  --debug          Enable debug output"
    echo ""
    echo "Environment Variables:"
    echo "  DEBUG=1          Enable verbose debug output"
    echo "  NO_GPU=1         Disable GPU acceleration"
    echo ""
    echo "Examples:"
    echo "  $0               # Start the application normally"
    echo "  $0 --setup       # Setup environment only"
    echo "  $0 --check       # Check system requirements"
    echo "  $0 --clean       # Clean installation"
    echo "  DEBUG=1 $0       # Start with debug output"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check python-backend/backend.log for backend errors"
    echo "  • Ensure Python 3.8+ and Node.js 16+ are installed"
    echo "  • Run '$0 --check' to diagnose system issues"
    echo "  • Run '$0 --clean && $0 --setup' to reset environment"
    echo ""
}

# Main execution flow
main() {
    print_status "Initializing Airavat v1.0.0..."
    echo ""

    # Run prerequisite checks
    if ! find_python; then
        return 1
    fi

    if ! check_nodejs; then
        return 1
    fi

    check_system_requirements

    # Setup environments if needed
    local need_setup=false

    if [ ! -d "node_modules" ]; then
        print_debug "Node.js dependencies not found"
        need_setup=true
    fi

    if [ ! -d "python-backend/venv" ]; then
        print_debug "Python virtual environment not found"
        need_setup=true
    fi

    if [ "$need_setup" = true ]; then
        print_status "Setting up development environment..."

        if ! setup_nodejs_env; then
            return 1
        fi

        if ! setup_python_env; then
            return 1
        fi
    else
        print_success "Development environment already configured"
    fi

    # Check for model files
    check_model_files

    # Start the application
    if ! start_application; then
        print_error "Failed to start application"
        return 1
    fi

    return 0
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --setup)
        setup_only
        exit $?
        ;;
    --check)
        system_check
        exit $?
        ;;
    --clean)
        clean_installation
        exit $?
        ;;
    --debug)
        export DEBUG=1
        shift
        main "$@"
        exit $?
        ;;
    "")
        # No arguments, run main
        main
        exit $?
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

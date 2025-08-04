@echo off
setlocal enabledelayedexpansion

REM Airavat v1.0.0 - Windows Startup Script
echo.
echo 🐘 Starting Airavat v1.0.0
echo ===========================
echo.

REM Set environment variables for compatibility
set LIBGL_ALWAYS_SOFTWARE=1
set ELECTRON_DISABLE_GPU=1
set ELECTRON_NO_SANDBOX=1
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1

REM Colors for output (Windows 10+)
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "BLUE=[34m"
set "NC=[0m"

REM Global variables
set PYTHON_CMD=
set BACKEND_PID=
set FRONTEND_PID=

REM Function to print status messages
:print_status
echo %GREEN%[INFO]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARN]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

REM Check if Python is installed
:check_python
call :print_status "Checking Python installation..."

REM Try different Python commands
for %%p in (python python3 py) do (
    %%p --version >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%v in ('%%p --version 2^>^&1') do (
            set python_version=%%v
            for /f "tokens=1,2 delims=." %%a in ("!python_version!") do (
                set major=%%a
                set minor=%%b
                if !major! geq 3 if !minor! geq 8 (
                    set PYTHON_CMD=%%p
                    call :print_success "Found Python: %%p (!python_version!)"
                    goto :python_found
                ) else (
                    call :print_warning "%%p version !python_version! is too old (need 3.8+)"
                )
            )
        )
    )
)

call :print_error "No suitable Python installation found"
call :print_error "Please install Python 3.8+ from https://python.org/"
exit /b 1

:python_found
goto :eof

REM Check if Node.js is installed
:check_nodejs
call :print_status "Checking Node.js installation..."

node --version >nul 2>&1
if !errorlevel! neq 0 (
    call :print_error "Node.js not found"
    call :print_error "Please install Node.js 16+ from https://nodejs.org/"
    exit /b 1
)

for /f "tokens=1" %%v in ('node --version') do (
    set node_version=%%v
    set node_major=!node_version:~1,2!
    if !node_major! lss 16 (
        call :print_error "Node.js version 16+ required. Found: !node_version!"
        exit /b 1
    )
    call :print_success "Node.js OK: !node_version!"
)

npm --version >nul 2>&1
if !errorlevel! neq 0 (
    call :print_error "npm not found"
    exit /b 1
)

for /f "tokens=1" %%v in ('npm --version') do (
    call :print_success "npm OK: v%%v"
)

goto :eof

REM Check system requirements
:check_system_requirements
call :print_status "Checking system requirements..."

REM Check available memory (Windows)
for /f "skip=1 tokens=4" %%m in ('wmic computersystem get TotalPhysicalMemory') do (
    if not "%%m"=="" (
        set /a total_mem_gb=%%m/1024/1024/1024
        if !total_mem_gb! lss 8 (
            call :print_warning "Less than 8GB RAM detected (!total_mem_gb!GB). Performance may be limited."
        ) else (
            call :print_success "Memory OK: !total_mem_gb!GB RAM"
        )
        goto :mem_done
    )
)
:mem_done

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    for /f "skip=9 tokens=2,3,4,5,6,7,8,9" %%a in ('nvidia-smi') do (
        if "%%a" neq "" if "%%b" neq "" (
            call :print_success "GPU detected: %%a %%b %%c %%d"
            goto :gpu_done
        )
    )
) else (
    call :print_warning "No NVIDIA GPU detected - will use CPU processing"
)
:gpu_done

goto :eof

REM Setup Python environment
:setup_python_env
call :print_status "Setting up Python environment..."

cd python-backend
if !errorlevel! neq 0 (
    call :print_error "python-backend directory not found"
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    call :print_status "Creating Python virtual environment..."
    %PYTHON_CMD% -m venv venv
    if !errorlevel! neq 0 (
        call :print_error "Failed to create virtual environment"
        cd ..
        exit /b 1
    )
)

REM Activate virtual environment
call :print_status "Activating virtual environment..."
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip --quiet

REM Install requirements
if not exist "requirements.txt" (
    call :print_error "requirements.txt not found"
    cd ..
    exit /b 1
)

call :print_status "Installing Python dependencies (this may take a while)..."
pip install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    call :print_error "Failed to install Python dependencies"
    cd ..
    exit /b 1
)

call :print_success "Python dependencies installed successfully"
cd ..
goto :eof

REM Setup Node.js environment
:setup_nodejs_env
call :print_status "Setting up Node.js environment..."

if not exist "package.json" (
    call :print_error "package.json not found"
    exit /b 1
)

call :print_status "Installing Node.js dependencies..."
npm install --silent
if !errorlevel! neq 0 (
    call :print_error "Failed to install Node.js dependencies"
    exit /b 1
)

call :print_success "Node.js dependencies installed successfully"
goto :eof

REM Check for model files
:check_model_files
call :print_status "Checking for AI model files..."

set models_found=0

if not exist "python-backend\models" mkdir "python-backend\models"

if exist "python-backend\models\siamese_best_model.pth" (
    call :print_success "Siamese model found"
    set /a models_found+=1
) else (
    call :print_warning "Siamese model not found - will run in demo mode"
)

if exist "python-backend\models\yolo_best_model.pt" (
    call :print_success "YOLOv8 model found"
    set /a models_found+=1
) else (
    call :print_warning "YOLOv8 model not found - will use default model"
)

if !models_found! equ 0 (
    call :print_warning "No custom model files found"
    call :print_warning "Download models from: https://github.com/yourusername/electron-app/releases"
) else (
    call :print_success "!models_found!/2 model files available"
)

goto :eof

REM Start Python backend
:start_python_backend
call :print_status "Starting Python backend server..."

cd python-backend
if not exist "backend_server.py" (
    call :print_error "backend_server.py not found"
    cd ..
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv" (
    call venv\Scripts\activate.bat
)

REM Start backend
call :print_status "Launching backend server..."
start /b "" python backend_server.py
if !errorlevel! neq 0 (
    call :print_error "Failed to start Python backend"
    cd ..
    exit /b 1
)

cd ..

REM Wait for backend to start
call :print_status "Waiting for backend to initialize..."
timeout /t 8 /nobreak >nul

goto :eof

REM Start Electron frontend
:start_electron_frontend
call :print_status "Starting Electron frontend..."

if not exist "src\main.js" (
    call :print_error "src\main.js not found"
    exit /b 1
)

call :print_status "Launching Electron application..."
start /b "" npm start
if !errorlevel! neq 0 (
    call :print_error "Failed to start Electron frontend"
    exit /b 1
)

goto :eof

REM Main startup function
:start_application
call :print_status "🚀 Starting Airavat application..."

call :start_python_backend
if !errorlevel! neq 0 exit /b 1

call :start_electron_frontend
if !errorlevel! neq 0 exit /b 1

echo.
call :print_success "🎉 Airavat is now running!"
call :print_success "📱 The application window should open automatically"
call :print_success "🌐 Backend API available at: http://localhost:3001"
echo.
call :print_status "📊 Application Status:"
call :print_status "  • Frontend: Running"
call :print_status "  • Backend: Running"
call :print_status "  • API Endpoint: http://localhost:3001/api/health"
echo.
call :print_status "Press Ctrl+C to stop the application"
echo.

REM Keep the window open
pause

goto :eof

REM Clean installation
:clean_installation
call :print_status "Cleaning installation..."

REM Remove Node.js dependencies
if exist "node_modules" (
    call :print_status "Removing Node.js dependencies..."
    rmdir /s /q "node_modules"
)

REM Remove Python virtual environment
if exist "python-backend\venv" (
    call :print_status "Removing Python virtual environment..."
    rmdir /s /q "python-backend\venv"
)

REM Remove temporary files
call :print_status "Cleaning temporary files..."
if exist "python-backend\temp_uploads" rmdir /s /q "python-backend\temp_uploads"
if exist "python-backend\temp_results" rmdir /s /q "python-backend\temp_results"
if exist "python-backend\__pycache__" rmdir /s /q "python-backend\__pycache__"
if exist "python-backend\backend.log" del "python-backend\backend.log"
if exist "dist" rmdir /s /q "dist"

call :print_success "Cleanup completed!"
goto :eof

REM Setup only mode
:setup_only
call :print_status "Setting up development environment only..."

call :check_python
if !errorlevel! neq 0 exit /b 1

call :check_nodejs
if !errorlevel! neq 0 exit /b 1

call :setup_python_env
if !errorlevel! neq 0 exit /b 1

call :setup_nodejs_env
if !errorlevel! neq 0 exit /b 1

call :check_model_files

call :print_success "Environment setup completed!"
call :print_status "Run 'start.bat' to start the application"
goto :eof

REM Help function
:show_help
echo 🐘 Airavat v1.0.0 - AI Elephant Identification System
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --help           Show this help message
echo   --setup          Setup environment only (don't start app)
echo   --clean          Clean all dependencies and temporary files
echo.
echo Examples:
echo   %~nx0            # Start the application normally
echo   %~nx0 --setup    # Setup environment only
echo   %~nx0 --clean    # Clean installation
echo.
echo Troubleshooting:
echo   • Check python-backend\backend.log for backend errors
echo   • Ensure Python 3.8+ and Node.js 16+ are installed
echo   • Run '%~nx0 --clean' then '%~nx0 --setup' to reset environment
echo.
goto :eof

REM Main execution
:main
call :print_status "Initializing Airavat v1.0.0..."
echo.

REM Run prerequisite checks
call :check_python
if !errorlevel! neq 0 exit /b 1

call :check_nodejs
if !errorlevel! neq 0 exit /b 1

call :check_system_requirements

REM Setup environments if needed
set need_setup=0

if not exist "node_modules" (
    set need_setup=1
)

if not exist "python-backend\venv" (
    set need_setup=1
)

if !need_setup! equ 1 (
    call :print_status "Setting up development environment..."

    call :setup_nodejs_env
    if !errorlevel! neq 0 exit /b 1

    call :setup_python_env
    if !errorlevel! neq 0 exit /b 1
) else (
    call :print_success "Development environment already configured"
)

REM Check for model files
call :check_model_files

REM Start the application
call :start_application
if !errorlevel! neq 0 (
    call :print_error "Failed to start application"
    exit /b 1
)

goto :eof

REM Parse command line arguments
if "%~1"=="--help" (
    call :show_help
    exit /b 0
)

if "%~1"=="--setup" (
    call :setup_only
    exit /b !errorlevel!
)

if "%~1"=="--clean" (
    call :clean_installation
    exit /b !errorlevel!
)

if "%~1"=="" (
    call :main
    exit /b !errorlevel!
)

call :print_error "Unknown option: %~1"
echo.
call :show_help
exit /b 1

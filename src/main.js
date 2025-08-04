const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs-extra');
const os = require('os');

let mainWindow;
let pythonProcess = null;
let backendReady = false;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    icon: path.join(__dirname, '../assets/icon.png'),
    show: false,
    titleBarStyle: 'default',
    title: 'Airavat - AI Elephant Identification v1.0.0'
  });

  mainWindow.loadFile('src/frontend/index.html');

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();

    // Show loading message if backend isn't ready
    if (!backendReady) {
      showLoadingScreen();
    }
  });

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
    cleanup();
  });
}

function showLoadingScreen() {
  mainWindow.webContents.executeJavaScript(`
    document.body.innerHTML = \`
      <div style="display: flex; justify-content: center; align-items: center; height: 100vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Segoe UI', sans-serif;">
        <div style="text-align: center; color: white;">
          <div style="font-size: 3em; margin-bottom: 20px;">🐘</div>
          <h1 style="margin: 0 0 20px 0;">Starting Airavat...</h1>
          <p style="font-size: 1.2em; margin: 10px 0;">Initializing AI Backend</p>
          <div style="margin: 20px 0;">
            <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.3); border-top: 4px solid white; border-radius: 50%; animation: spin 1s linear infinite;"></div>
          </div>
          <p id="status" style="font-size: 1em; margin: 10px 0; opacity: 0.8;">Setting up Python environment...</p>
          <style>
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
          </style>
        </div>
      </div>
    \`;
  `);
}

function updateLoadingStatus(message) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.executeJavaScript(`
      const statusEl = document.getElementById('status');
      if (statusEl) statusEl.textContent = '${message}';
    `);
  }
}

function getResourcePath(relativePath) {
  if (app.isPackaged) {
    // In packaged app, resources are in different location
    return path.join(process.resourcesPath, relativePath);
  } else {
    // In development
    return path.join(__dirname, '..', relativePath);
  }
}

function getPythonCommand() {
  // Priority order for finding Python
  const commands = [];

  if (app.isPackaged) {
    // First try portable Python bundled with the app
    const portablePython = path.join(process.resourcesPath, 'python-portable', 'python', 'python.exe');
    if (fs.existsSync(portablePython)) {
      commands.push(portablePython);
    }

    // Try additional bundled Python locations
    const altPortablePython = path.join(process.resourcesPath, 'app.asar.unpacked', 'python-portable', 'python', 'python.exe');
    if (fs.existsSync(altPortablePython)) {
      commands.push(altPortablePython);
    }
  }

  // Fallback to system Python
  if (process.platform === 'win32') {
    commands.push('python', 'python3', 'py');
  } else {
    commands.push('python3', 'python');
  }

  return commands;
}

function checkPythonInstallation() {
  return new Promise((resolve, reject) => {
    const commands = getPythonCommand();
    let commandIndex = 0;

    function tryCommand() {
      if (commandIndex >= commands.length) {
        reject(new Error('No Python installation found'));
        return;
      }

      const cmd = commands[commandIndex];
      console.log(`Checking Python command: ${cmd}`);
      updateLoadingStatus(`Checking Python: ${path.basename(cmd)}...`);

      const testProcess = spawn(cmd, ['--version'], { stdio: 'pipe' });

      testProcess.on('close', (code) => {
        if (code === 0) {
          console.log(`Found working Python command: ${cmd}`);
          resolve(cmd);
        } else {
          commandIndex++;
          tryCommand();
        }
      });

      testProcess.on('error', () => {
        commandIndex++;
        tryCommand();
      });
    }

    tryCommand();
  });
}

async function installPythonDependencies(pythonCmd) {
  console.log('Installing Python dependencies...');
  updateLoadingStatus('Installing Python dependencies...');

  const backendDir = getResourcePath('python-backend');
  const requirementsPath = path.join(backendDir, 'requirements.txt');

  if (!fs.existsSync(requirementsPath)) {
    console.error('requirements.txt not found');
    return false;
  }

  return new Promise((resolve) => {
    const installProcess = spawn(pythonCmd, ['-m', 'pip', 'install', '-r', 'requirements.txt'], {
      cwd: backendDir,
      stdio: 'pipe'
    });

    installProcess.stdout.on('data', (data) => {
      console.log(`Pip install: ${data}`);
    });

    installProcess.stderr.on('data', (data) => {
      console.log(`Pip install (stderr): ${data}`);
    });

    installProcess.on('close', (code) => {
      console.log(`Pip install process exited with code ${code}`);
      resolve(code === 0);
    });

    installProcess.on('error', (error) => {
      console.error(`Pip install error: ${error}`);
      resolve(false);
    });
  });
}

async function setupModels() {
  console.log('Setting up AI models...');
  updateLoadingStatus('Setting up AI models...');

  const modelsDir = getResourcePath('models');
  const backendModelsDir = getResourcePath('python-backend/models');

  // Create backend models directory
  fs.ensureDirSync(backendModelsDir);

  if (fs.existsSync(modelsDir)) {
    // Copy models from bundled location to backend
    try {
      fs.copySync(modelsDir, backendModelsDir, { overwrite: true });
      console.log('✅ Models copied to backend directory');

      // Check which models are available
      const siameseModel = path.join(backendModelsDir, 'siamese_best_model.pth');
      const yoloModel = path.join(backendModelsDir, 'yolo_best_model.pt');

      if (fs.existsSync(siameseModel)) {
        const stats = fs.statSync(siameseModel);
        console.log(`✅ Siamese model available: ${(stats.size / 1024 / 1024).toFixed(1)}MB`);
      }

      if (fs.existsSync(yoloModel)) {
        const stats = fs.statSync(yoloModel);
        console.log(`✅ YOLO model available: ${(stats.size / 1024 / 1024).toFixed(1)}MB`);
      }

    } catch (error) {
      console.warn('Could not copy models:', error.message);
    }
  } else {
    console.log('⚠️ No models directory found - will run in demo mode');
  }

  // Create placeholder models if none exist (for demo mode)
  const siameseModel = path.join(backendModelsDir, 'siamese_best_model.pth');
  const yoloModel = path.join(backendModelsDir, 'yolo_best_model.pt');

  if (!fs.existsSync(siameseModel)) {
    fs.writeFileSync(siameseModel, '# Placeholder - Download real models from releases');
    console.log('Created Siamese model placeholder');
  }

  if (!fs.existsSync(yoloModel)) {
    fs.writeFileSync(yoloModel, '# Placeholder - Download real models from releases');
    console.log('Created YOLO model placeholder');
  }
}

async function startPythonBackend() {
  console.log('🐘 Starting Airavat Backend v1.0.0');
  updateLoadingStatus('Starting Python backend...');

  try {
    // Check Python installation
    const pythonCmd = await checkPythonInstallation();
    console.log(`Using Python command: ${pythonCmd}`);

    const backendDir = getResourcePath('python-backend');
    const backendScript = path.join(backendDir, 'backend_server.py');

    // Check if backend script exists
    if (!fs.existsSync(backendScript)) {
      throw new Error(`Backend script not found: ${backendScript}`);
    }

    // Setup models
    await setupModels();

    // Try to install dependencies (non-blocking for packaged apps)
    if (!app.isPackaged) {
      console.log('Checking Python dependencies...');
      updateLoadingStatus('Installing dependencies...');
      const depsInstalled = await installPythonDependencies(pythonCmd);
      if (!depsInstalled) {
        console.warn('⚠️  Some Python dependencies might be missing');
      }
    }

    // Create necessary directories
    const tempDirs = ['temp_uploads', 'temp_results', 'models'];
    tempDirs.forEach(dir => {
      const dirPath = path.join(backendDir, dir);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`Created directory: ${dir}`);
      }
    });

    // Set environment variables for Python
    const env = {
      ...process.env,
      PYTHONPATH: backendDir,
      PYTHONUNBUFFERED: '1',
      PYTHONDONTWRITEBYTECODE: '1'
    };

    // Add portable Python to PATH if available
    if (app.isPackaged && pythonCmd.includes('python-portable')) {
      const pythonPortableDir = path.dirname(pythonCmd);
      env.PATH = `${pythonPortableDir};${env.PATH}`;
    }

    // Start Python backend
    console.log('🚀 Launching Python backend server...');
    updateLoadingStatus('Launching backend server...');

    pythonProcess = spawn(pythonCmd, [backendScript], {
      cwd: backendDir,
      env,
      stdio: 'pipe'
    });

    let backendStarted = false;

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`🐍 Python: ${output.trim()}`);

      // Check if backend is ready
      if (output.includes('Backend server starting') ||
          output.includes('Running on') ||
          output.includes('Starting Flask server')) {
        backendStarted = true;
        backendReady = true;
        console.log('✅ Python backend is ready!');
        updateLoadingStatus('Backend ready! Loading interface...');

        // Reload the main window content
        if (mainWindow && !mainWindow.isDestroyed()) {
          setTimeout(() => {
            mainWindow.reload();
          }, 1000);
        }
      }

      // Update loading status based on output
      if (output.includes('Installing')) {
        updateLoadingStatus('Installing dependencies...');
      } else if (output.includes('Loading')) {
        updateLoadingStatus('Loading AI models...');
      } else if (output.includes('Server initialization complete')) {
        updateLoadingStatus('Server ready!');
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.error(`🐍 Python Error: ${output.trim()}`);

      // Check for import errors
      if (output.includes('ModuleNotFoundError') || output.includes('ImportError')) {
        console.error('❌ Missing Python dependencies detected');
        updateLoadingStatus('Some dependencies missing - will use demo mode');
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`🐍 Python process exited with code ${code}`);
      backendReady = false;

      if (code !== 0 && !backendStarted) {
        console.error('❌ Python backend failed to start');
        showBackendError();
      }
    });

    pythonProcess.on('error', (error) => {
      console.error(`🐍 Python process error: ${error.message}`);
      backendReady = false;
      showBackendError();
    });

    // Wait for backend to start (with timeout)
    return new Promise((resolve) => {
      let attempts = 0;
      const maxAttempts = 15; // 45 seconds total

      const checkBackend = setInterval(() => {
        attempts++;

        if (backendStarted) {
          clearInterval(checkBackend);
          resolve(true);
        } else if (attempts >= maxAttempts) {
          clearInterval(checkBackend);
          console.warn('⚠️  Backend startup timeout - continuing anyway');
          // The backend might still be starting in demo mode
          resolve(false);
        }
      }, 3000);
    });

  } catch (error) {
    console.error(`❌ Failed to start Python backend: ${error.message}`);
    showBackendError();
    return false;
  }
}

function showBackendError() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.executeJavaScript(`
      document.body.innerHTML = \`
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); font-family: 'Segoe UI', sans-serif;">
          <div style="text-align: center; color: white; max-width: 600px; padding: 40px;">
            <div style="font-size: 3em; margin-bottom: 20px;">⚠️</div>
            <h1 style="margin: 0 0 20px 0;">Backend Startup Failed</h1>
            <p style="font-size: 1.1em; margin: 20px 0; line-height: 1.6;">
              The AI backend failed to start. The app may run with limited functionality:
            </p>
            <ul style="text-align: left; display: inline-block; margin: 20px 0;">
              <li>✅ Interface will still work</li>
              <li>⚠️ AI features will use demo mode</li>
              <li>⚠️ No real elephant identification</li>
              <li>⚠️ Mock detection results only</li>
            </ul>
            <div style="margin: 30px 0;">
              <button onclick="location.reload()" style="background: rgba(255,255,255,0.2); color: white; border: 2px solid white; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1em; margin-right: 10px;">
                🔄 Retry
              </button>
              <button onclick="continueWithDemo()" style="background: rgba(255,255,255,0.2); color: white; border: 2px solid white; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1em;">
                📱 Continue in Demo Mode
              </button>
            </div>
            <p style="font-size: 0.9em; margin-top: 30px; opacity: 0.8;">
              Check console logs for detailed error information
            </p>
          </div>
        </div>
        <script>
          function continueWithDemo() {
            // Set a flag and reload
            localStorage.setItem('demo-mode', 'true');
            location.reload();
          }
        </script>
      \`;
    `);
  }
}

function cleanup() {
  if (pythonProcess) {
    console.log('🛑 Shutting down Python backend...');
    pythonProcess.kill('SIGTERM');

    // Force kill after 5 seconds
    setTimeout(() => {
      if (pythonProcess && !pythonProcess.killed) {
        pythonProcess.kill('SIGKILL');
      }
    }, 5000);
  }
}

// Enhanced IPC Handlers
ipcMain.handle('select-folder', async () => {
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
      title: 'Select Folder Containing Elephant Images'
    });

    if (!result.canceled && result.filePaths.length > 0) {
      return result.filePaths[0];
    }
    return null;
  } catch (error) {
    console.error('Error selecting folder:', error);
    return null;
  }
});

ipcMain.handle('select-files', async () => {
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile', 'multiSelections'],
      title: 'Select Elephant Images',
      filters: [
        { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });

    if (!result.canceled && result.filePaths.length > 0) {
      return result.filePaths;
    }
    return null;
  } catch (error) {
    console.error('Error selecting files:', error);
    return null;
  }
});

ipcMain.handle('get-system-info', async () => {
  try {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const cpus = os.cpus();

    return {
      totalMemory: Math.round(totalMem / 1024 / 1024 / 1024) + ' GB',
      freeMemory: Math.round(freeMem / 1024 / 1024 / 1024) + ' GB',
      usedMemory: Math.round((totalMem - freeMem) / 1024 / 1024 / 1024) + ' GB',
      cpuCount: cpus.length,
      cpuModel: cpus[0].model,
      platform: process.platform,
      arch: process.arch,
      nodeVersion: process.version,
      electronVersion: process.versions.electron,
      backendStatus: backendReady ? 'Ready' : 'Starting...',
      isPackaged: app.isPackaged,
      resourcesPath: app.isPackaged ? process.resourcesPath : 'Development'
    };
  } catch (error) {
    console.error('Error getting system info:', error);
    return {
      error: 'Failed to retrieve system information'
    };
  }
});

ipcMain.handle('check-backend-status', async () => {
  return {
    ready: backendReady,
    processRunning: pythonProcess && !pythonProcess.killed
  };
});

ipcMain.handle('restart-backend', async () => {
  console.log('Restarting backend...');

  if (pythonProcess) {
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
  }

  backendReady = false;
  const success = await startPythonBackend();
  return success;
});

// App event handlers
app.whenReady().then(async () => {
  console.log('🚀 Electron app ready, starting backend...');

  // Start backend first
  await startPythonBackend();

  // Create window
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  cleanup();

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', (event) => {
  if (pythonProcess && !pythonProcess.killed) {
    console.log('🛑 Gracefully shutting down backend...');
    pythonProcess.kill('SIGTERM');

    // Give it a moment to shut down
    event.preventDefault();
    setTimeout(() => {
      app.quit();
    }, 2000);
  }
});

// Handle protocol for development
if (process.defaultApp) {
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('airavat', process.execPath, [path.resolve(process.argv[1])]);
  }
} else {
  app.setAsDefaultProtocolClient('airavat');
}

// Add error handling for uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

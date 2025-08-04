#!/usr/bin/env node

/**
 * Setup Python Portable for Electron App
 * Downloads and configures portable Python with required dependencies
 */

const fs = require('fs-extra');
const path = require('path');
const https = require('https');
const { execSync, spawn } = require('child_process');
const os = require('os');

const PYTHON_VERSION = '3.11.7';
const PYTHON_PORTABLE_DIR = 'python-portable';

// Python download URLs
const PYTHON_URLS = {
  win32: {
    x64: `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip`,
    ia32: `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-win32.zip`
  }
};

async function downloadFile(url, outputPath) {
  return new Promise((resolve, reject) => {
    console.log(`Downloading: ${url}`);
    const file = fs.createWriteStream(outputPath);

    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Handle redirects
        return downloadFile(response.headers.location, outputPath);
      }

      if (response.statusCode !== 200) {
        reject(new Error(`Download failed: ${response.statusCode}`));
        return;
      }

      const totalSize = parseInt(response.headers['content-length'] || '0');
      let downloadedSize = 0;

      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        const percent = totalSize > 0 ? (downloadedSize / totalSize * 100).toFixed(1) : '?';
        process.stdout.write(`\rProgress: ${percent}% (${downloadedSize}/${totalSize} bytes)`);
      });

      response.pipe(file);

      file.on('finish', () => {
        console.log('\nDownload completed');
        file.close();
        resolve();
      });

      file.on('error', (err) => {
        fs.unlink(outputPath);
        reject(err);
      });
    }).on('error', reject);
  });
}

async function extractZip(zipPath, extractPath) {
  console.log(`Extracting: ${zipPath} to ${extractPath}`);

  if (process.platform === 'win32') {
    // Use PowerShell on Windows
    const command = `powershell -command "Expand-Archive -Path '${zipPath}' -DestinationPath '${extractPath}' -Force"`;
    execSync(command, { stdio: 'inherit' });
  } else {
    // Use unzip on Unix-like systems
    execSync(`unzip -q "${zipPath}" -d "${extractPath}"`, { stdio: 'inherit' });
  }
}

async function setupPythonPortable() {
  try {
    console.log('🐍 Setting up portable Python environment...');

    const platform = os.platform();
    const arch = os.arch();

    if (platform !== 'win32') {
      console.log('⚠️  Portable Python setup is primarily for Windows builds');
      console.log('On Unix systems, the system Python will be used');
      return;
    }

    // Clean existing portable Python
    if (fs.existsSync(PYTHON_PORTABLE_DIR)) {
      console.log('Cleaning existing portable Python...');
      fs.removeSync(PYTHON_PORTABLE_DIR);
    }

    fs.ensureDirSync(PYTHON_PORTABLE_DIR);

    // Get download URL
    const downloadUrl = PYTHON_URLS[platform][arch === 'x64' ? 'x64' : 'ia32'];
    if (!downloadUrl) {
      throw new Error(`No Python download available for ${platform} ${arch}`);
    }

    // Download Python
    const zipPath = path.join(PYTHON_PORTABLE_DIR, 'python.zip');
    await downloadFile(downloadUrl, zipPath);

    // Extract Python
    const pythonDir = path.join(PYTHON_PORTABLE_DIR, 'python');
    fs.ensureDirSync(pythonDir);
    await extractZip(zipPath, pythonDir);

    // Clean up zip
    fs.removeSync(zipPath);

    console.log('✅ Portable Python extracted');

    // Configure Python
    await configurePython(pythonDir);

    // Install pip
    await installPip(pythonDir);

    // Install requirements
    await installRequirements(pythonDir);

    console.log('✅ Portable Python setup completed!');

  } catch (error) {
    console.error('❌ Error setting up portable Python:', error.message);
    process.exit(1);
  }
}

async function configurePython(pythonDir) {
  console.log('Configuring Python...');

  // Enable site-packages by modifying python311._pth
  const pthFiles = fs.readdirSync(pythonDir).filter(f => f.endsWith('._pth'));

  if (pthFiles.length > 0) {
    const pthPath = path.join(pythonDir, pthFiles[0]);
    let pthContent = fs.readFileSync(pthPath, 'utf8');

    // Enable site-packages
    pthContent = pthContent.replace('#import site', 'import site');
    pthContent += '\nLib\\site-packages\n';

    fs.writeFileSync(pthPath, pthContent);
    console.log('✅ Python path configuration updated');
  }

  // Create Lib directory structure
  const libDir = path.join(pythonDir, 'Lib');
  const sitePackagesDir = path.join(libDir, 'site-packages');
  fs.ensureDirSync(sitePackagesDir);
}

async function installPip(pythonDir) {
  console.log('Installing pip...');

  const pythonExe = path.join(pythonDir, 'python.exe');

  // Download get-pip.py
  const getPipPath = path.join(pythonDir, 'get-pip.py');
  await downloadFile('https://bootstrap.pypa.io/get-pip.py', getPipPath);

  // Install pip
  execSync(`"${pythonExe}" "${getPipPath}"`, {
    stdio: 'inherit',
    cwd: pythonDir
  });

  // Clean up
  fs.removeSync(getPipPath);

  console.log('✅ pip installed');
}

async function installRequirements(pythonDir) {
  console.log('Installing Python requirements...');

  const pythonExe = path.join(pythonDir, 'python.exe');
  const requirementsPath = path.join(process.cwd(), 'python-backend', 'requirements.txt');

  if (!fs.existsSync(requirementsPath)) {
    console.log('⚠️  requirements.txt not found, skipping package installation');
    return;
  }

  try {
    // Install requirements with portable Python
    execSync(`"${pythonExe}" -m pip install -r "${requirementsPath}" --no-warn-script-location`, {
      stdio: 'inherit',
      cwd: pythonDir
    });

    console.log('✅ Python requirements installed');
  } catch (error) {
    console.log('⚠️  Some packages may have failed to install');
    console.log('The app will fall back to mock implementations');
  }
}

// Create batch file for easy Python access
async function createPythonBatch(pythonDir) {
  const batchContent = `@echo off
set PYTHONPATH=%~dp0
set PYTHONDONTWRITEBYTECODE=1
"%~dp0python.exe" %*
`;

  fs.writeFileSync(path.join(pythonDir, 'python-portable.bat'), batchContent);
}

if (require.main === module) {
  setupPythonPortable();
}

module.exports = { setupPythonPortable };

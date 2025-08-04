#!/usr/bin/env node

/**
 * Download AI Models Script
 * Downloads the required model files from GitHub releases
 */

const fs = require('fs-extra');
const path = require('path');
const https = require('https');

// Configuration
const GITHUB_REPO = 'yourusername/electron-app'; // Replace with your actual repo
const MODELS_DIR = 'models';
const MODELS_CONFIG = {
  'siamese_best_model.pth': {
    url: `https://github.com/${GITHUB_REPO}/releases/download/models-v1.0.0/siamese_best_model.pth`,
    description: 'Siamese Neural Network model for elephant identification',
    required: false // Can work without it (demo mode)
  },
  'yolo_best_model.pt': {
    url: `https://github.com/${GITHUB_REPO}/releases/download/models-v1.0.0/yolo_best_model.pt`,
    description: 'YOLOv8 model for elephant detection',
    required: false // Can work without it (uses default YOLOv8)
  }
};

async function downloadFile(url, outputPath) {
  return new Promise((resolve, reject) => {
    console.log(`📥 Downloading: ${path.basename(outputPath)}`);

    const file = fs.createWriteStream(outputPath);

    const request = https.get(url, (response) => {
      // Handle redirects
      if (response.statusCode === 302 || response.statusCode === 301) {
        file.close();
        fs.removeSync(outputPath);
        return downloadFile(response.headers.location, outputPath)
          .then(resolve)
          .catch(reject);
      }

      if (response.statusCode !== 200) {
        file.close();
        fs.removeSync(outputPath);
        reject(new Error(`Download failed: HTTP ${response.statusCode}`));
        return;
      }

      const totalSize = parseInt(response.headers['content-length'] || '0');
      let downloadedSize = 0;

      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        if (totalSize > 0) {
          const percent = (downloadedSize / totalSize * 100).toFixed(1);
          const mbDownloaded = (downloadedSize / 1024 / 1024).toFixed(1);
          const mbTotal = (totalSize / 1024 / 1024).toFixed(1);
          process.stdout.write(`\r   Progress: ${percent}% (${mbDownloaded}/${mbTotal} MB)`);
        }
      });

      response.pipe(file);

      file.on('finish', () => {
        console.log('\n   ✅ Download completed');
        file.close();
        resolve();
      });

      file.on('error', (err) => {
        fs.removeSync(outputPath);
        reject(err);
      });
    });

    request.on('error', (err) => {
      file.close();
      fs.removeSync(outputPath);
      reject(err);
    });

    request.setTimeout(30000, () => {
      request.destroy();
      file.close();
      fs.removeSync(outputPath);
      reject(new Error('Download timeout'));
    });
  });
}

function formatFileSize(bytes) {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

async function checkExistingModels() {
  console.log('🔍 Checking existing models...');

  const existingModels = [];
  const modelsDir = MODELS_DIR;

  if (!fs.existsSync(modelsDir)) {
    fs.ensureDirSync(modelsDir);
  }

  for (const [filename, config] of Object.entries(MODELS_CONFIG)) {
    const filePath = path.join(modelsDir, filename);

    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      const size = formatFileSize(stats.size);

      // Check if file is too small (likely a placeholder)
      if (stats.size < 1024 * 1024) { // Less than 1MB
        console.log(`   ⚠️  ${filename} exists but seems too small (${size}) - will re-download`);
        fs.removeSync(filePath);
      } else {
        console.log(`   ✅ ${filename} exists (${size})`);
        existingModels.push(filename);
      }
    } else {
      console.log(`   ❌ ${filename} not found`);
    }
  }

  return existingModels;
}

async function downloadModels() {
  try {
    console.log('🤖 AI Models Download Manager');
    console.log('================================');

    // Check existing models
    const existingModels = await checkExistingModels();

    const modelsToDownload = Object.keys(MODELS_CONFIG).filter(
      filename => !existingModels.includes(filename)
    );

    if (modelsToDownload.length === 0) {
      console.log('✅ All models are already present!');
      return;
    }

    console.log(`\n📦 Downloading ${modelsToDownload.length} model(s)...`);

    let successCount = 0;
    let failCount = 0;

    for (const filename of modelsToDownload) {
      const config = MODELS_CONFIG[filename];
      const outputPath = path.join(MODELS_DIR, filename);

      try {
        console.log(`\n📄 ${config.description}`);
        await downloadFile(config.url, outputPath);

        // Verify file size
        const stats = fs.statSync(outputPath);
        if (stats.size < 1024) { // Less than 1KB probably means error page
          throw new Error('Downloaded file is too small');
        }

        console.log(`   ✅ ${filename} downloaded successfully (${formatFileSize(stats.size)})`);
        successCount++;

      } catch (error) {
        console.log(`   ❌ Failed to download ${filename}: ${error.message}`);

        if (config.required) {
          failCount++;
        } else {
          console.log(`   ℹ️  ${filename} is optional - app will work in demo mode`);
        }

        // Clean up failed download
        if (fs.existsSync(outputPath)) {
          fs.removeSync(outputPath);
        }
      }
    }

    console.log('\n📊 Download Summary:');
    console.log(`   ✅ Successfully downloaded: ${successCount}`);
    console.log(`   ❌ Failed downloads: ${failCount}`);
    console.log(`   📁 Models directory: ${path.resolve(MODELS_DIR)}`);

    if (failCount > 0) {
      console.log('\n⚠️  Some models failed to download, but the app can still run:');
      console.log('   • Siamese Network: Will use demo mode with mock results');
      console.log('   • YOLOv8: Will use default pre-trained model');
    }

    // Create model info file
    await createModelInfo(successCount, failCount);

    console.log('\n🎉 Model download process completed!');

  } catch (error) {
    console.error('❌ Error in model download process:', error.message);
    process.exit(1);
  }
}

async function createModelInfo(successCount, failCount) {
  const modelInfo = {
    download_date: new Date().toISOString(),
    models_downloaded: successCount,
    models_failed: failCount,
    models_available: {},
    readme: 'This file contains information about downloaded AI models'
  };

  // Check which models are available
  for (const [filename, config] of Object.entries(MODELS_CONFIG)) {
    const filePath = path.join(MODELS_DIR, filename);
    const available = fs.existsSync(filePath);

    modelInfo.models_available[filename] = {
      available,
      description: config.description,
      size: available ? formatFileSize(fs.statSync(filePath).size) : null
    };
  }

  // Write model info
  const infoPath = path.join(MODELS_DIR, 'models_info.json');
  fs.writeJsonSync(infoPath, modelInfo, { spaces: 2 });

  // Create README
  const readmePath = path.join(MODELS_DIR, 'README.md');
  const readmeContent = `# AI Models

This directory contains the AI models used by Airavat:

## Available Models

${Object.entries(modelInfo.models_available).map(([filename, info]) =>
  `- **${filename}**: ${info.available ? '✅' : '❌'} ${info.description}${info.size ? ` (${info.size})` : ''}`
).join('\n')}

## Notes

- If models are missing, the app will run in demo mode
- Models can be downloaded from: https://github.com/${GITHUB_REPO}/releases
- Total models downloaded: ${successCount}/${Object.keys(MODELS_CONFIG).length}

Generated on: ${new Date().toISOString()}
`;

  fs.writeFileSync(readmePath, readmeContent);
}

// Alternative: Download from direct URLs (if you host models elsewhere)
const ALTERNATIVE_URLS = {
  'siamese_best_model.pth': 'https://your-hosting-service.com/models/siamese_best_model.pth',
  'yolo_best_model.pt': 'https://your-hosting-service.com/models/yolo_best_model.pt'
};

async function downloadFromAlternativeSource() {
  console.log('📡 Trying alternative download sources...');

  for (const [filename, url] of Object.entries(ALTERNATIVE_URLS)) {
    const outputPath = path.join(MODELS_DIR, filename);

    if (fs.existsSync(outputPath)) {
      console.log(`   ⏭️  ${filename} already exists, skipping`);
      continue;
    }

    try {
      await downloadFile(url, outputPath);
      console.log(`   ✅ Downloaded ${filename} from alternative source`);
    } catch (error) {
      console.log(`   ❌ Alternative download failed for ${filename}: ${error.message}`);
    }
  }
}

if (require.main === module) {
  downloadModels();
}

module.exports = { downloadModels, downloadFromAlternativeSource };

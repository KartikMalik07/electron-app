

// Configuration for connecting to FastAPI Render backend
const BACKEND_CONFIG = {
    // Your Render backend URL (replace with your actual URL)
    RENDER_URL: 'https://your-app-name.onrender.com',

    // Local development URL
    LOCAL_URL: 'http://localhost:8000',  // FastAPI default port

    // API endpoints
    ENDPOINTS: {
        HEALTH: '/api/health',
        MODEL_INFO: '/api/model-info',
        COMPARE_DATASET: '/api/compare-dataset',
        DETECT_YOLO: '/api/detect-yolo',
        PROCESS_BATCH: '/api/process-batch',
        DOCS: '/docs',  // FastAPI auto-generated docs
        REDOC: '/redoc'  // Alternative docs
    },

    // Timeouts
    TIMEOUT: 30000, // 30 seconds for API calls
    HEALTH_CHECK_INTERVAL: 30000, // 30 seconds

    // File upload limits
    MAX_FILE_SIZE: 200 * 1024 * 1024, // 200MB
};

// Auto-detect backend URL
function getBackendUrl() {
    // Check if we're in development (localhost)
    const isDevelopment = window.location.hostname === 'localhost' ||
                         window.location.hostname === '127.0.0.1';

    if (isDevelopment) {
        return BACKEND_CONFIG.LOCAL_URL;
    }

    // In production, use the Render URL
    return BACKEND_CONFIG.RENDER_URL;
}

// Enhanced API client for FastAPI with better error handling
class AiravataFastAPI {
    constructor() {
        this.baseURL = getBackendUrl();
        this.isOnline = false;
        this.lastHealthCheck = null;
        this.modelInfo = null;

        // Start health monitoring
        this.startHealthMonitoring();
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;

        const defaultOptions = {
            timeout: BACKEND_CONFIG.TIMEOUT,
            headers: {
                'Accept': 'application/json',
            }
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            // Add timeout to fetch
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), requestOptions.timeout);

            const response = await fetch(url, {
                ...requestOptions,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                // FastAPI returns detailed error information
                let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    if (errorData.detail) {
                        errorDetail = errorData.detail;
                    }
                } catch (e) {
                    // Fallback to status text if JSON parsing fails
                }
                throw new Error(errorDetail);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }

        } catch (error) {
            console.error(`API Error for ${endpoint}:`, error);

            if (error.name === 'AbortError') {
                throw new Error('Request timeout - backend may be starting up');
            } else if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
                throw new Error('Cannot connect to backend - check if server is running');
            } else {
                throw error;
            }
        }
    }

    async healthCheck() {
        try {
            const health = await this.makeRequest(BACKEND_CONFIG.ENDPOINTS.HEALTH);
            this.isOnline = true;
            this.lastHealthCheck = new Date();
            return health;
        } catch (error) {
            this.isOnline = false;
            console.warn('Backend health check failed:', error.message);
            return null;
        }
    }

    startHealthMonitoring() {
        // Initial health check
        this.healthCheck();

        // Periodic health checks
        setInterval(() => {
            this.healthCheck();
        }, BACKEND_CONFIG.HEALTH_CHECK_INTERVAL);
    }

    async getModelInfo() {
        const info = await this.makeRequest(BACKEND_CONFIG.ENDPOINTS.MODEL_INFO);
        this.modelInfo = info;
        return info;
    }

    async compareWithDataset(imageFile, threshold = 0.85, topK = 10) {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('threshold', threshold.toString());
        formData.append('top_k', topK.toString());

        return await this.makeRequest(BACKEND_CONFIG.ENDPOINTS.COMPARE_DATASET, {
            method: 'POST',
            body: formData
        });
    }

    async detectElephants(imageFile, confidence = 0.5, iou = 0.45, imageSize = 640) {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('confidence', confidence.toString());
        formData.append('iou', iou.toString());
        formData.append('image_size', imageSize.toString());

        return await this.makeRequest(BACKEND_CONFIG.ENDPOINTS.DETECT_YOLO, {
            method: 'POST',
            body: formData
        });
    }

    async processBatch(folderPath, modelType, threshold, maxGroups, outputFormat) {
        const formData = new FormData();
        formData.append('folder_path', folderPath);
        formData.append('model_type', modelType);
        formData.append('threshold', threshold.toString());
        formData.append('max_groups', maxGroups.toString());
        formData.append('output_format', outputFormat);

        return await this.makeRequest(BACKEND_CONFIG.ENDPOINTS.PROCESS_BATCH, {
            method: 'POST',
            body: formData
        });
    }

    getStatus() {
        return {
            isOnline: this.isOnline,
            baseURL: this.baseURL,
            lastHealthCheck: this.lastHealthCheck,
            modelInfo: this.modelInfo
        };
    }

    // FastAPI specific methods
    openDocs() {
        window.open(`${this.baseURL}${BACKEND_CONFIG.ENDPOINTS.DOCS}`, '_blank');
    }

    openRedoc() {
        window.open(`${this.baseURL}${BACKEND_CONFIG.ENDPOINTS.REDOC}`, '_blank');
    }
}

// Create global API instance
const api = new AiravataFastAPI();

// Enhanced error handling and user feedback
function showBackendStatus() {
    const status = api.getStatus();
    const statusContainer = document.getElementById('modelStatus');

    if (!statusContainer) return;

    if (status.isOnline) {
        const docsLink = `${status.baseURL}/docs`;
        statusContainer.innerHTML = `
            <p><span class="status-indicator status-online"></span>FastAPI Backend Online</p>
            <p><strong>Server:</strong> ${status.baseURL}</p>
            <p><strong>Last Check:</strong> ${status.lastHealthCheck ? status.lastHealthCheck.toLocaleTimeString() : 'Never'}</p>
            <p style="font-size: 0.9em; color: #666;">Real AI models active</p>
            <p style="font-size: 0.8em; margin-top: 10px;">
                <a href="${docsLink}" target="_blank" style="color: #667eea; text-decoration: none;">
                    📚 View API Documentation
                </a>
            </p>
        `;
    } else {
        statusContainer.innerHTML = `
            <p><span class="status-indicator status-offline"></span>FastAPI Backend Offline</p>
            <p><strong>Trying:</strong> ${status.baseURL}</p>
            <p style="color: #e74c3c;">Connection failed - server may be starting up</p>
        `;
    }
}

// File upload validation with FastAPI-specific error handling
function validateImageFile(file) {
    if (!file) {
        throw new Error('No file selected');
    }

    if (file.size > BACKEND_CONFIG.MAX_FILE_SIZE) {
        throw new Error(`File too large. Maximum size is ${BACKEND_CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB`);
    }

    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        throw new Error('Invalid file type. Please use JPG, PNG, BMP, or TIFF');
    }

    return true;
}

// Enhanced processing functions for FastAPI
async function processSiameseImageReal(file) {
    const progressContainer = document.getElementById('siameseProgress');
    const resultsContainer = document.getElementById('siameseResults');
    const progressFill = document.getElementById('siameseProgressFill');
    const progressText = document.getElementById('siameseProgressText');

    progressContainer.style.display = 'block';
    resultsContainer.style.display = 'none';

    try {
        // Validate file
        validateImageFile(file);

        // Animate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            progressFill.style.width = progress + '%';
            progressText.textContent = `Analyzing elephant features... ${progress}%`;

            if (progress >= 90) {
                clearInterval(progressInterval);
            }
        }, 200);

        // Get parameters
        const threshold = parseFloat(document.getElementById('siameseThreshold').value);
        const topK = parseInt(document.getElementById('siameseTopK').value);

        // Call FastAPI endpoint
        const results = await api.compareWithDataset(file, threshold, topK);

        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Real AI analysis complete!';

        displaySiameseResults(results);

        setTimeout(() => {
            progressContainer.style.display = 'none';
        }, 1000);

    } catch (error) {
        console.error('Error processing image:', error);
        progressText.textContent = 'Analysis failed: ' + error.message;

        setTimeout(() => {
            progressContainer.style.display = 'none';
        }, 3000);

        showErrorMessage('siameseResults', error.message);
    }
}

async function processYoloImageReal(file) {
    const progressContainer = document.getElementById('yoloProgress');
    const resultsContainer = document.getElementById('yoloResults');
    const progressFill = document.getElementById('yoloProgressFill');
    const progressText = document.getElementById('yoloProgressText');

    progressContainer.style.display = 'block';
    resultsContainer.style.display = 'none';

    try {
        validateImageFile(file);

        // Animate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 8;
            progressFill.style.width = progress + '%';
            progressText.textContent = `Running YOLOv8 detection... ${progress}%`;

            if (progress >= 90) {
                clearInterval(progressInterval);
            }
        }, 150);

        // Get parameters
        const confidence = parseFloat(document.getElementById('yoloConfidence').value);
        const iou = parseFloat(document.getElementById('yoloIou').value);
        const imageSize = parseInt(document.getElementById('yoloImageSize').value);

        // Call FastAPI endpoint
        const results = await api.detectElephants(file, confidence, iou, imageSize);

        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Real YOLOv8 detection complete!';

        displayYoloResults(results);

        setTimeout(() => {
            progressContainer.style.display = 'none';
        }, 1000);

    } catch (error) {
        console.error('Error processing image:', error);
        progressText.textContent = 'Detection failed: ' + error.message;

        setTimeout(() => {
            progressContainer.style.display = 'none';
        }, 3000);

        showErrorMessage('yoloResults', error.message);
    }
}

// Update file handling for drag & drop
function handleFileInput(files, processingFunction) {
    if (files && files.length > 0) {
        const file = files[0];
        processingFunction(file);
    }
}

// Enhanced status monitoring
function updateBackendStatus() {
    showBackendStatus();

    // Update model info if backend is online
    if (api.isOnline) {
        updateModelInfo();
    }
}

async function updateModelInfo() {
    try {
        const modelInfo = await api.getModelInfo();
        const statusContainer = document.getElementById('modelStatus');

        if (statusContainer && modelInfo) {
            statusContainer.innerHTML += `
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">
                    <p><strong>Siamese Network:</strong> ${modelInfo.siamese_status}</p>
                    <p><strong>YOLOv8 Model:</strong> ${modelInfo.yolo_status}</p>
                    <p><strong>Dataset Size:</strong> ${modelInfo.dataset_size} elephants</p>
                    <p><strong>Device:</strong> ${modelInfo.device}</p>
                    ${modelInfo.cuda_available ? '<p style="color: #27ae60;">🚀 GPU Acceleration Active</p>' : '<p style="color: #f39c12;">⚡ CPU Processing</p>'}
                </div>
            `;
        }
    } catch (error) {
        console.warn('Failed to get model info:', error);
    }
}

// Connection retry logic
async function retryConnection(maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        console.log(`Connection attempt ${i + 1}/${maxRetries}`);

        const health = await api.healthCheck();
        if (health) {
            console.log('✅ Connected to FastAPI backend');
            updateBackendStatus();
            return true;
        }

        // Wait before retry (exponential backoff)
        const delay = Math.pow(2, i) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    console.warn('❌ Failed to connect after', maxRetries, 'attempts');
    return false;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🐘 Airavat Frontend v1.0.0 - Connecting to FastAPI Backend');
    console.log('Backend URL:', api.baseURL);

    // Try to connect with retries
    const connected = await retryConnection(5);

    if (!connected) {
        showConnectionError();
    }

    // Set up periodic status updates
    setInterval(updateBackendStatus, BACKEND_CONFIG.HEALTH_CHECK_INTERVAL);
});

function showConnectionError() {
    const container = document.querySelector('.container');
    if (container) {
        const errorBanner = document.createElement('div');
        errorBanner.style.cssText = `
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        `;
        errorBanner.innerHTML = `
            <h3>⚠️ FastAPI Backend Connection Failed</h3>
            <p>Cannot connect to the AI backend at: <strong>${api.baseURL}</strong></p>
            <p>The backend may be starting up or there may be a network issue.</p>
            <div style="margin-top: 15px;">
                <button onclick="location.reload()" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px;">
                    🔄 Retry Connection
                </button>
                <button onclick="api.openDocs()" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    📚 API Docs
                </button>
            </div>
        `;
        container.insertBefore(errorBanner, container.firstChild);
    }
}

// Update existing functions to use the new API
function processSiameseImage(file) {
    return processSiameseImageReal(file);
}

function processYoloImage(file) {
    return processYoloImageReal(file);
}

// Update file selection handlers
async function selectSiameseImage() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
        if (e.target.files[0]) {
            processSiameseImage(e.target.files[0]);
        }
    };
    input.click();
}

async function selectYoloImage() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
        if (e.target.files[0]) {
            processYoloImage(e.target.files[0]);
        }
    };
    input.click();
}

// Update drag and drop handlers
function handleSiameseImageDrop(files) {
    handleFileInput(files, processSiameseImage);
}

function handleYoloImageDrop(files) {
    handleFileInput(files, processYoloImage);
}

// Export for global use
window.AiravataAPI = api;
window.BACKEND_CONFIG = BACKEND_CONFIG;

#!/usr/bin/env python3
"""
Model Downloader for Render Deployment
Downloads model files during build process
"""
import os
import json
import urllib.request
import hashlib
import sys

def download_file(url, filename):
    """Download file with progress"""
    print(f"Downloading {filename}...")
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def verify_file_hash(filename, expected_hash):
    """Verify file integrity"""
    if not expected_hash:
        return True
    hash_sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    actual_hash = hash_sha256.hexdigest()
    if actual_hash == expected_hash:
        print(f"✅ {filename} hash verified")
        return True
    else:
        print(f"❌ {filename} hash mismatch")
        return False

def main():
    # Load model configuration
    if not os.path.exists('model_config.json'):
        print("❌ model_config.json not found")
        return False
    with open('model_config.json', 'r') as f:
        config = json.load(f)

    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.chdir('models')

    # Download each model
    success = True
    for filename, url in config['download_urls'].items():
        if not download_file(url, filename):
            success = False
            continue
        # Verify hash if available
        expected_hash = config.get('model_hashes', {}).get(filename)
        if not verify_file_hash(filename, expected_hash):
            success = False

    if success:
        print("\n🎉 All models downloaded successfully!")
        return True
    else:
        print("\n❌ Some models failed to download")
        return False

if __name__ == '__main__':
    sys.exit(0 if main() else 1)

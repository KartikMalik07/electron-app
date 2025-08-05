#!/usr/bin/env python3
"""
Airavat Model Upload Tool (GitHub Releases Only)
Uploads .pth/.pt model files to GitHub Releases for Render deployment
"""
import os
import sys
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_model_files():
    """Check if model files exist and get their info"""
    model_files = {
        'siamese_best_model.pth': None,
        'yolo_best_model.pt': None
    }
    for filename in model_files.keys():
        # Check in multiple locations
        possible_paths = [
            f'models/{filename}',
            f'python-backend/models/{filename}',
            filename
        ]
        for path in possible_paths:
            if os.path.exists(path):
                stat = os.stat(path)
                model_files[filename] = {
                    'path': path,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'hash': get_file_hash(path)
                }
                break
    return model_files


def get_file_hash(filepath):
    """Get SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def upload_to_github_releases(model_files, repo_name, token):
    """Upload models to GitHub releases"""
    try:
        from github import Github
    except ImportError:
        logger.error("PyGithub not installed. Run: pip install PyGithub")
        return False

    try:
        g = Github(token)
        repo = g.get_repo(repo_name)

        # Create or get release
        release_tag = "models-v1.0.0"
        try:
            release = repo.get_release(release_tag)
            logger.info(f"Found existing release: {release_tag}")
        except Exception:
            logger.info(f"Creating new release: {release_tag}")
            release = repo.create_git_release(
                tag=release_tag,
                name="AI Models v1.0.0",
                message="Elephant identification AI models"
            )

        # Upload each model file
        for filename, info in model_files.items():
            if info is None:
                logger.warning(f"Skipping {filename} - file not found")
                continue

            logger.info(f"Uploading {filename} ({info['size_mb']:.1f}MB)...")

            # Check if asset already exists
            existing_asset = None
            for asset in release.get_assets():
                if asset.name == filename:
                    existing_asset = asset
                    break

            if existing_asset:
                logger.info(f"Deleting existing {filename}...")
                existing_asset.delete_asset()

            # Upload new asset
            with open(info['path'], 'rb') as f:
                release.upload_asset(
                    path=info['path'],
                    name=filename,
                    content_type='application/octet-stream'
                )
            logger.info(f"✅ Successfully uploaded {filename}")

        # Create download URLs
        download_urls = {}
        for asset in release.get_assets():
            download_urls[asset.name] = asset.browser_download_url

        # Save config
        config = {
            'storage_type': 'github_releases',
            'repo_name': repo_name,
            'release_tag': release_tag,
            'download_urls': download_urls,
            'upload_date': str(datetime.now()),
            'model_hashes': {name: info['hash'] for name, info in model_files.items() if info}
        }
        with open('model_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("✅ All models uploaded successfully!")
        logger.info("📁 Model URLs saved to model_config.json")
        return True

    except Exception as e:
        logger.error(f"❌ GitHub upload failed: {e}")
        return False


def create_model_downloader():
    """Create a script that Render can use to download models"""
    script_content = '''#!/usr/bin/env python3
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
            print(f"\\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\\n❌ Failed to download {filename}: {e}")
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
        print("\\n🎉 All models downloaded successfully!")
        return True
    else:
        print("\\n❌ Some models failed to download")
        return False

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
'''
    with open('download_models.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    logger.info("✅ Created download_models.py script")


def main():
    """Main function"""
    logger.info("🐘 Airavat Model Upload Tool (GitHub Releases Only)")
    logger.info("=" * 50)

    # Check model files
    logger.info("🔍 Checking for model files...")
    model_files = check_model_files()
    found_models = [name for name, info in model_files.items() if info is not None]

    if not found_models:
        logger.error("❌ No model files found!")
        logger.error("Expected files: siamese_best_model.pth, yolo_best_model.pt")
        return False

    logger.info(f"✅ Found {len(found_models)} model file(s):")
    for name, info in model_files.items():
        if info:
            logger.info(f"  • {name}: {info['size_mb']:.1f} MB")

    # GitHub-specific setup
    repo_name = os.getenv('GITHUB_REPO')
    token = os.getenv('GITHUB_TOKEN')

    if not repo_name or not token:
        logger.error("❌ Missing environment variables!")
        logger.info("Please set: GITHUB_REPO and GITHUB_TOKEN")
        logger.info("Example: export GITHUB_REPO='your-username/your-repo'")
        return False

    logger.info(f"📦 Uploading models to GitHub repo: {repo_name}")

    # Upload to GitHub Releases
    success = upload_to_github_releases(model_files, repo_name, token)

    if success:
        # Create downloader script
        create_model_downloader()
        logger.info("\n🎉 Upload complete!")
        logger.info("➡️  Next steps:")
        logger.info("1. Commit 'model_config.json' and 'download_models.py' to your repo")
        logger.info("2. Update Render build command to:")
        logger.info("   python download_models.py && pip install -r requirements.txt")
        logger.info("3. Deploy!")
        return True
    else:
        logger.error("❌ Upload failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

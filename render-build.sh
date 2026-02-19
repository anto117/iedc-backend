#!/usr/bin/env bash
# exit on error
set -o errexit

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🎥 Downloading Portable FFmpeg..."
# Download a pre-compiled, portable version of FFmpeg
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz

# Extract the ffmpeg tool and place it directly in your app folder
cp ffmpeg-*-amd64-static/ffmpeg .
chmod +x ffmpeg

# Clean up the downloaded zip files to save server space
rm ffmpeg-release-amd64-static.tar.xz
rm -rf ffmpeg-*-amd64-static

echo "✅ Build Complete!"
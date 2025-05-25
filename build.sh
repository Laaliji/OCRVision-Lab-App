#!/usr/bin/env bash
# Install Tesseract OCR
apt-get update -y
apt-get install -y tesseract-ocr
apt-get install -y libtesseract-dev

# Tell Python where to find Tesseract
export TESSERACT_PATH=/usr/bin/tesseract

# Exit successfully
exit 0 
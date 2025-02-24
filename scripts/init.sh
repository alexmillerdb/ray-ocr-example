#!/bin/bash
set -x
apt-get update
apt-get install -y poppler-utils
apt-get install -y tesseract-ocr
apt-get install -y libmagic-dev
#!/usr/bin/env bash
set -euo pipefail

# Download the Colorectal Cancer WSI / EBHI-Seg dataset from Kaggle.
# Requires that you have configured the Kaggle CLI with your API credentials.
#
# Usage:
#   bash scripts/download_kaggle.sh

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw/colorectal-cancer-wsi"
mkdir -p "${DATA_DIR}"

echo "Downloading Colorectal Cancer WSI dataset to: ${DATA_DIR}"
kaggle datasets download -d mahdiislam/colorectal-cancer-wsi -p "${DATA_DIR}" --unzip

echo "Download complete."



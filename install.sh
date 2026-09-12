#!/usr/bin/env bash
set -e
python3 -m pip install -r requirements.txt
# CPU-only torch hint (no CUDA): python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0

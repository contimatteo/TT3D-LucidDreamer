#!/bin/bash

pip install -U pip wheel setuptools
# pip install torch==2.0.1 torchvision==0.15.2 xformers==0.0.20 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r custom_requirements.txt
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/

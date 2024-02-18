#!/bin/bash

pip install -U pip wheel setuptools
pip install torch==2.0.1 torchvision==0.15.2 triton --index-url https://download.pytorch.org/whl/cu118
pip install ninja
# pip install --force-reinstall -U setuptools
# pip install --force-reinstall -U pip
pip install -r custom_requirements.txt 
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install trimesh

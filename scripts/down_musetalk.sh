#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="checkpoints"

# Create necessary directories
mkdir -p ${CheckpointsDir}/musetalkV15 ${CheckpointsDir}/face-parse-bisent ${CheckpointsDir}/sd-vae ${CheckpointsDir}/whisper

# Install required packages
pip install -U "huggingface_hub[cli]"
pip install gdown

# Set HuggingFace mirror endpoint
#export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.5 weights (unet.pth)
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download Face Parse Bisent weights
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

echo "✅ All weights have been downloaded successfully!"
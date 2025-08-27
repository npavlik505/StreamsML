#!/usr/bin/env bash
set -e

# Install system packages using apt
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y \
    curl \
    libssl-dev \
    pkg-config \
    git \
    python3 \
    python3-dev \
    python3-pip \
    libhdf5-mpi-dev \
    build-essential \
    libopenmpi-dev \
    cargo
else
  echo "apt-get not found. Please install required packages manually." >&2
  exit 1
fi

# Upgrade pip and install python dependencies
pip3 install --upgrade pip
pip3 install mpi4py h5py numpy

# Install just using cargo
if ! command -v just >/dev/null 2>&1; then
  cargo install just
fi

# Set environment variables for streams repositories
STREAMS_DIR_PATH="$(pwd)/streams"
STREAMS_UTILS_DIR_PATH="$(pwd)/streams-utils"

if [ -d "$STREAMS_DIR_PATH" ] && ! grep -q "STREAMS_DIR" ~/.bashrc; then
  echo "export STREAMS_DIR=\"$STREAMS_DIR_PATH\"" >> ~/.bashrc
fi
if [ -d "$STREAMS_UTILS_DIR_PATH" ] && ! grep -q "STREAMS_UTILS_DIR" ~/.bashrc; then
  echo "export STREAMS_UTILS_DIR=\"$STREAMS_UTILS_DIR_PATH\"" >> ~/.bashrc
fi

echo "Dependencies installed. Restart your shell to load environment variables."

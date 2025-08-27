#!/usr/bin/env bash
set -e

# Install system packages using apt (linux) or brew (mac)
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
elif command -v brew >/dev/null 2>&1; then
  brew update
  brew install \
    curl \
    openssl \
    pkg-config \
    git \
    python \
    hdf5 \
    open-mpi \
    rust \
    just
else
  echo "No supported package manager found (apt-get or brew).\n" \
       "Install required system packages manually." >&2
fi

# Create and activate a virtual environment
python3 -m venv streamsenv
source streamsenv/bin/activate

# Upgrade pip and install python dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r ./streams-utils/requirements.txt

# Install just using cargo
if ! command -v just >/dev/null 2>&1; then
  cargo install just
fi


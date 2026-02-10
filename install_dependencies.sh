#!/usr/bin/env bash
set -e

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "Please run :source ./install_dependencies.sh"
	echo "This script must be sourced to update PATH in your current shell."
	exit 1
fi

# Install system packages using apt (linux) or brew (mac)
PYTHON_BIN="python3"

if command -v apt-get >/dev/null 2>&1; then
  IS_UBUNTU=false
  FUSE_PACKAGE="fuse3"
  if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID}" == "ubuntu" ]]; then
      IS_UBUNTU=true
      UBUNTU_VERSION="${VERSION_ID%%.*}"
      if [[ -n "${UBUNTU_VERSION}" && "${UBUNTU_VERSION}" -lt 24 ]]; then
        FUSE_PACKAGE="fuse"
      fi
    fi
  fi

  if [[ "${FUSE_PACKAGE}" == "fuse3" ]] && dpkg -s fuse >/dev/null 2>&1; then
    sudo apt-get remove -y fuse
  fi

  sudo apt-get update -y
  APT_EXTRA_PACKAGES=()
  if [[ "${IS_UBUNTU}" == "true" ]]; then
    APT_EXTRA_PACKAGES+=(python3-mpi4py)
  fi

  sudo apt-get install -y \
    ca-certificates \
    cmake \
    curl \
    libssl-dev \
    pkg-config \
    git \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    "${FUSE_PACKAGE}" \
    squashfuse \
    uidmap \
    libhdf5-mpi-dev \
    build-essential \
    libopenmpi-dev \
    "${APT_EXTRA_PACKAGES[@]}"
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python3"
  fi
elif command -v brew >/dev/null 2>&1; then
  brew update
  brew install \
    curl \
    openssl \
    pkg-config \
    git \
    python \
    hdf5 \
    open-mpi
else
  echo "No supported package manager found (apt-get or brew).\n" \
       "Install required system packages manually." >&2
fi

# Configure Apptainer temp/cache dirs for unprivileged installs
APPTAINER_TMPDIR="${HOME}/AppTmpDir"
mkdir -p "${APPTAINER_TMPDIR}"
export APPTAINER_TMPDIR

APPTAINER_CACHEDIR="${HOME}/.cache/apptainer"
mkdir -p "${APPTAINER_CACHEDIR}"
export APPTAINER_CACHEDIR

# Install Apptainer prebuilt binaries if missing
if ! command -v apptainer >/dev/null 2>&1; then
  APPTAINER_VERSION="1.3.4"
  APPTAINER_PREFIX="${HOME}/.local/apptainer"
  APPTAINER_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
  APPTAINER_ARCH="$(uname -m)"

  case "${APPTAINER_ARCH}" in
    x86_64) APPTAINER_ARCH="amd64" ;;
    aarch64|arm64) APPTAINER_ARCH="arm64" ;;
    *)
      echo "Unsupported architecture for Apptainer binaries: ${APPTAINER_ARCH}" >&2
      APPTAINER_ARCH=""
      ;;
  esac

  if [[ "${APPTAINER_OS}" == "linux" && -n "${APPTAINER_ARCH}" ]]; then
    APPTAINER_TARBALL="apptainer-${APPTAINER_VERSION}-${APPTAINER_OS}-${APPTAINER_ARCH}.tar.gz"
    APPTAINER_URL="https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/${APPTAINER_TARBALL}"
    APPTAINER_DOWNLOAD_PATH="${APPTAINER_TMPDIR}/${APPTAINER_TARBALL}"

    mkdir -p "${APPTAINER_PREFIX}"
    echo "Downloading Apptainer ${APPTAINER_VERSION} from ${APPTAINER_URL}"
    curl -L "${APPTAINER_URL}" -o "${APPTAINER_DOWNLOAD_PATH}"
    tar -xzf "${APPTAINER_DOWNLOAD_PATH}" -C "${APPTAINER_PREFIX}"

    APPTAINER_EXTRACT_DIR=""
    for candidate in "${APPTAINER_PREFIX}"/apptainer-"${APPTAINER_VERSION}"*; do
      if [[ -d "${candidate}" ]]; then
        APPTAINER_EXTRACT_DIR="${candidate}"
        break
      fi
    done

    if [[ -n "${APPTAINER_EXTRACT_DIR}" && -d "${APPTAINER_EXTRACT_DIR}/bin" ]]; then
      APPTAINER_BIN_DIR="${APPTAINER_EXTRACT_DIR}/bin"
    else
      APPTAINER_BIN_DIR="${APPTAINER_PREFIX}/bin"
    fi

    export PATH="${APPTAINER_BIN_DIR}:${PATH}"
    echo "Apptainer installed to ${APPTAINER_PREFIX}."

    APPTAINER_PROFILE_BIN_DIR="${APPTAINER_BIN_DIR/#${HOME}/\$HOME}"
    APPTAINER_PATH_LINE="export PATH=\"${APPTAINER_PROFILE_BIN_DIR}:\$PATH\""
    PROFILE_FILES=("${HOME}/.profile" "${HOME}/.bashrc")

    for profile_file in "${PROFILE_FILES[@]}"; do
      if [[ -f "${profile_file}" ]] && ! grep -Fxq "${APPTAINER_PATH_LINE}" "${profile_file}"; then
        echo "${APPTAINER_PATH_LINE}" >> "${profile_file}"
        echo "Added Apptainer PATH update to ${profile_file}. Run 'source ${profile_file}' or open a new shell."
      fi
    done
  else
    echo "Skipping Apptainer install (unsupported OS or architecture)." >&2
  fi
fi

# Create and activate a virtual environment
PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="$(${PYTHON_BIN} -c 'import sys; print(sys.version_info.major)')"
PYTHON_MINOR="$(${PYTHON_BIN} -c 'import sys; print(sys.version_info.minor)')"
if [[ "${PYTHON_MAJOR}" -ne 3 || "${PYTHON_MINOR}" -lt 9 || "${PYTHON_MINOR}" -gt 12 ]]; then
  echo "Unsupported Python ${PYTHON_VERSION}. Supported versions are >=3.9 and <=3.12." >&2
  echo "Install Python 3.10+ and re-run: source ./install_dependencies.sh" >&2
  return 1
fi

${PYTHON_BIN} -m venv streamsenv
source streamsenv/bin/activate

# Upgrade pip and install python dependencies
streamsenv/bin/python -m pip install --upgrade "pip==24.0" "setuptools<70" "wheel"

if command -v dpkg >/dev/null 2>&1 && dpkg -s python3-mpi4py >/dev/null 2>&1; then
  MPI4PY_FILTERED_REQUIREMENTS="$(mktemp)"
  grep -Ev '^mpi4py([<>=!~].*)?$' ./streams-utils/venv_reqs.txt > "${MPI4PY_FILTERED_REQUIREMENTS}"
  streamsenv/bin/python -m pip install --no-cache-dir -r "${MPI4PY_FILTERED_REQUIREMENTS}"
  rm -f "${MPI4PY_FILTERED_REQUIREMENTS}"
else
  streamsenv/bin/python -m pip install --no-cache-dir -r ./streams-utils/venv_reqs.txt
fi

# Install just using cargo (project-local Rust/Cargo dirs: .rustup, .cargo).
# Delete these directories to reset the local Rust toolchain.
export RUSTUP_HOME="${PWD}/.rustup"
export CARGO_HOME="${PWD}/.cargo"
export PATH="${CARGO_HOME}/bin:${PATH}"

if ! command -v rustup >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
  export RUSTUP_INIT_SKIP_PATH_CHECK=1
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  unset RUSTUP_INIT_SKIP_PATH_CHECK
fi

RUST_TOOLCHAIN_VERSION="1.75.0"
JUST_VERSION="1.38.0"
RUSTUP_BOOTSTRAP_TOOLCHAIN="stable"

rustup toolchain install "${RUST_TOOLCHAIN_VERSION}" --profile minimal
rustup toolchain install "${RUSTUP_BOOTSTRAP_TOOLCHAIN}" --profile minimal
rustup default "${RUST_TOOLCHAIN_VERSION}"

if ! command -v just >/dev/null 2>&1; then
  cargo +${RUSTUP_BOOTSTRAP_TOOLCHAIN} install just --version "${JUST_VERSION}"
fi

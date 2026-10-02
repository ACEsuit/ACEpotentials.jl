#!/bin/bash
# =============================================================================
# Build Portable ACE Plugin using Apptainer Container
# =============================================================================
#
# This script builds a portable ACE deployment inside an Apptainer container
# based on manylinux_2_28, ensuring compatibility with glibc 2.28+.
#
# Usage:
#   ./build_portable.sh config.yaml [output_dir]
#
# Arguments:
#   config.yaml  - YAML configuration file (same format as deploy_model.jl)
#   output_dir   - Optional output directory (default: current directory)
#
# Requirements:
#   - Apptainer (or Singularity) installed
#   - Internet access (for first container build)
#
# The script will:
#   1. Build the container image if not present
#   2. Run the build inside the container
#   3. Output a portable tarball with glibc 2.28 baseline
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Get script directory (where this script and container def live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="${SCRIPT_DIR}/../container"
CONTAINER_DEF="${CONTAINER_DIR}/portable_build.def"
CONTAINER_SIF="${CONTAINER_DIR}/portable_build.sif"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 config.yaml [output_dir]"
    echo ""
    echo "Build a portable ACE deployment using manylinux_2_28 container."
    echo ""
    echo "Arguments:"
    echo "  config.yaml  - YAML configuration file"
    echo "  output_dir   - Optional output directory (default: current directory)"
    exit 1
fi

CONFIG_FILE="$1"
OUTPUT_DIR="${2:-.}"

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Get absolute paths
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)" || mkdir -p "$OUTPUT_DIR" && OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

log_info "Configuration: $CONFIG_FILE"
log_info "Output directory: $OUTPUT_DIR"

# Check for Apptainer/Singularity
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    log_error "Neither apptainer nor singularity found in PATH"
    log_error "Please install Apptainer: https://apptainer.org/docs/admin/main/installation.html"
    exit 1
fi

log_info "Using container runtime: $CONTAINER_CMD"

# Build container if needed
if [ ! -f "$CONTAINER_SIF" ]; then
    log_info "Container image not found, building..."
    log_info "This may take 5-10 minutes on first run..."

    if [ ! -f "$CONTAINER_DEF" ]; then
        log_error "Container definition not found: $CONTAINER_DEF"
        exit 1
    fi

    # Build the container
    cd "$CONTAINER_DIR"
    $CONTAINER_CMD build "$CONTAINER_SIF" "$CONTAINER_DEF"

    log_info "Container built successfully: $CONTAINER_SIF"
else
    log_info "Using existing container: $CONTAINER_SIF"
fi

# Create temporary workspace for the build
WORKSPACE=$(mktemp -d)
trap "rm -rf $WORKSPACE" EXIT

log_info "Build workspace: $WORKSPACE"

# Copy necessary files to workspace
# The container will bind-mount the ACEpotentials repo and workspace
REPO_ROOT="${SCRIPT_DIR}/../.."

# Find data files referenced in config (if local paths)
# For now, we'll just bind-mount the entire repo and config location

log_info "Starting portable build inside container..."

# Run the build inside the container
# Bind mounts:
#   - /repo: ACEpotentials.jl repository (read-only for source)
#   - /workspace: Build output directory (read-write)
#   - /config: Directory containing config file (read-only)

$CONTAINER_CMD exec \
    --bind "${REPO_ROOT}:/repo:ro" \
    --bind "${OUTPUT_DIR}:/output" \
    --bind "$(dirname "$CONFIG_FILE"):/config:ro" \
    --bind "${WORKSPACE}:/workspace" \
    --pwd /workspace \
    "$CONTAINER_SIF" \
    julia --project=/repo/export /repo/export/scripts/build_portable.jl \
        "/config/$(basename "$CONFIG_FILE")" \
        /output

BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    log_info "Portable build completed successfully!"
    log_info "Output files are in: $OUTPUT_DIR"
else
    log_error "Build failed with status $BUILD_STATUS"
    exit $BUILD_STATUS
fi

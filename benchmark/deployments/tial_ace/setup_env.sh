#!/bin/bash
# Source this file to set up the environment for using the ACE potential
# Usage: source setup_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
echo "ACE potential environment configured."
echo "Library path: $SCRIPT_DIR/lib"

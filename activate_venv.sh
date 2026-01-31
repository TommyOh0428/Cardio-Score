#!/usr/bin/env bash

# Check if the script is being sourced (required to affect parent shell)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced to activate the environment."
    echo "Usage: source ./activate_venv.sh"
    exit 1
fi

VENV_NAME="venv"

if [ ! -d "$VENV_NAME" ]; then
    echo "Error: Directory '$VENV_NAME' not found."
    return 1
fi

# Determine OS and set activation path
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ACTIVATE_PATH="./$VENV_NAME/bin/activate"
elif [[ "$OSTYPE" == "msys" ]]; then
    ACTIVATE_PATH="./$VENV_NAME/Scripts/activate"
else
    echo "Unsupported OS: $OSTYPE"
    return 1
fi

# Source the activation script
if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
    echo "Virtual environment activated."
else
    echo "Error: Activation script not found at $ACTIVATE_PATH"
    return 1
fi
#!/bin/zsh

# Check if the Python script exists
if [ -f "models/hello.py" ]; then
    echo "Running .py file..."
    # Run the Python script
    python3 models/hello.py
else
    echo "py not found!"
fi

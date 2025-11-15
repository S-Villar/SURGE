#!/bin/bash
# Development setup script for SURGE

echo "Setting up SURGE development environment..."

# Install in development mode with all dependencies
pip install -e ".[dev]"

echo "Running tests to verify installation..."
python -m pytest tests/ -v

echo "Setup complete! You can now:"
echo "1. Import surge from any Python script"
echo "2. Run tests with: python -m pytest tests/"
echo "3. Format code with: black surge/ tests/"
echo "4. Build package with: python -m build"

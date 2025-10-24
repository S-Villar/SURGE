#!/bin/bash
# SURGE Environment Setup Script
# This script sets up the SURGE environment variable for easy access to data and resources

echo "🚀 SURGE Environment Variable Setup"
echo "===================================="
echo ""

# Detect the current script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SURGE_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Detected SURGE location: $SURGE_ROOT"
echo ""

# Verify this is the SURGE directory
if [ ! -d "$SURGE_ROOT/surge" ] || [ ! -f "$SURGE_ROOT/setup.py" ]; then
    echo "❌ Error: This doesn't appear to be the SURGE root directory."
    echo "   Expected to find 'surge/' folder and 'setup.py'"
    echo "   Current location: $SURGE_ROOT"
    exit 1
fi

echo "✅ Verified SURGE directory structure"
echo ""

# Determine which shell config file to use
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    echo "⚠️  Unknown shell. Please manually add to your shell config:"
    echo "   export SURGE=\"$SURGE_ROOT\""
    echo "   export PYTHONPATH=\"\$SURGE:\$PYTHONPATH\""
    exit 0
fi

echo "Detected shell config: $SHELL_CONFIG"
echo ""

# Check if SURGE is already in the config
if grep -q "export SURGE=" "$SHELL_CONFIG"; then
    echo "⚠️  SURGE environment variable already exists in $SHELL_CONFIG"
    echo ""
    echo "Current setting:"
    grep "export SURGE=" "$SHELL_CONFIG"
    echo ""
    read -p "Do you want to update it? [y/N]: " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping update."
        exit 0
    fi
    # Remove old SURGE entries
    sed -i.bak '/export SURGE=/d' "$SHELL_CONFIG"
    echo "✅ Removed old SURGE setting"
fi

# Add SURGE environment variable
echo "" >> "$SHELL_CONFIG"
echo "# SURGE - Surrogate Model Framework" >> "$SHELL_CONFIG"
echo "export SURGE=\"$SURGE_ROOT\"" >> "$SHELL_CONFIG"
echo "export PYTHONPATH=\"\$SURGE:\$PYTHONPATH\"" >> "$SHELL_CONFIG"

echo "✅ Added SURGE environment variable to $SHELL_CONFIG"
echo ""
echo "📝 Added the following lines:"
echo "   export SURGE=\"$SURGE_ROOT\""
echo "   export PYTHONPATH=\"\$SURGE:\$PYTHONPATH\""
echo ""
echo "🔄 To activate now, run:"
echo "   source $SHELL_CONFIG"
echo ""
echo "Or restart your terminal."
echo ""
echo "✅ Setup complete!"

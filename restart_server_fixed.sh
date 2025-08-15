#!/bin/bash

echo "Restarting server with fixed protocol..."

# Kill existing server
pkill -f "python.*brain.py"
sleep 1

# Start server with the fix
cd server
python3 brain.py --safe-mode

# The fixed protocol now properly handles:
# - NEW protocol clients (with magic bytes) consistently
# - OLD protocol clients (without magic bytes) consistently
# - No more misinterpretation after handshake
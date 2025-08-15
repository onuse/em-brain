#!/bin/bash

echo "========================================="
echo "Restarting Server with Chunked Protocol"
echo "========================================="
echo ""
echo "This protocol properly handles 1.2MB messages"
echo "even with limited TCP buffers (16KB default)."
echo ""

# Kill existing server
pkill -f "python.*brain.py"
sleep 1

# Start server
cd server
python3 brain.py --safe-mode

# The chunked protocol:
# - Reads large messages in 64KB chunks
# - Handles partial sends/receives properly
# - Works with system TCP buffer limits
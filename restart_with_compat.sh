#!/bin/bash

echo "========================================="
echo "Restarting Brain Server with Backward-Compatible Protocol"
echo "========================================="
echo ""
echo "This server will work with BOTH:"
echo "  • OLD robots (no magic bytes)"
echo "  • NEW robots (with magic bytes)"
echo ""

# Kill existing brain server
echo "Stopping existing brain server..."
pkill -f "python.*brain.py"
sleep 2

# Start new server
echo "Starting brain server with backward-compatible protocol..."
cd server
python3 brain.py --safe-mode 2>&1 | tee brain_compat.log &
cd ..

echo ""
echo "✅ Server restarted with backward compatibility!"
echo ""
echo "The server will automatically detect which protocol each client uses:"
echo "  • Old clients: [length][type][vector_length][data]"
echo "  • New clients: [magic][length][type][vector_length][data]"
echo ""
echo "Monitor logs with: tail -f server/brain_compat.log"
echo ""
echo "You should see messages like:"
echo "  'Client using OLD protocol (no magic bytes)' - for old robots"
echo "  'Client using NEW protocol (with magic bytes)' - for updated robots"
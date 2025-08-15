#!/bin/bash

echo "Applying protocol validation fix..."
echo "Changed MAX_REASONABLE_VECTOR_LENGTH from 100K to 10M"
echo ""

# Kill and restart brain server
echo "Restarting brain server..."
pkill -f "python.*brain.py"
sleep 1

cd server
python3 brain.py --safe-mode &
cd ..

echo ""
echo "✅ Fix applied! The server will now accept 307,212 sensor values."
echo ""
echo "The limit is now 10 million values, supporting:"
echo "  • 640×480 (VGA): 307,212 values ✓"
echo "  • 1280×720 (HD): 921,612 values ✓"  
echo "  • 1920×1080 (Full HD): 2,073,612 values ✓"
echo "  • 3840×2160 (4K): 8,294,412 values ✓"
echo ""
echo "Monitor logs with: tail -f server/brain.log"
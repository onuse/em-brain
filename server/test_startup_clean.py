#!/usr/bin/env python3
"""Test clean startup output"""

import sys
import threading
import time
sys.path.append('.')

from brain import DynamicBrainServer

# Create server
server = DynamicBrainServer()

# Create a thread to stop after showing startup
def stop_after_delay():
    time.sleep(0.5)
    print("\n\n✅ Startup output test complete")
    sys.exit(0)

threading.Thread(target=stop_after_delay, daemon=True).start()

# Start server (will be interrupted)
try:
    server.start()
except SystemExit:
    pass
#!/usr/bin/env python3
"""Test clean startup output"""

import sys
sys.path.append('.')

from brain import DynamicBrainServer

# Create server
server = DynamicBrainServer()
print("\n✅ Server initialized successfully")
#!/usr/bin/env python3
"""
Simple robot test without brain.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.brainstem.brainstem import Brainstem

print("Starting simple robot test...")
print("This runs the brainstem without brain connection.\n")

try:
    # Create brainstem without brain
    brainstem = Brainstem(enable_brain=False)
    
    print("Running 5 cycles...\n")
    for i in range(5):
        print(f"Cycle {i+1}")
        brainstem.run_cycle()
        time.sleep(0.1)
    
    print("\nShutting down...")
    brainstem.shutdown()
    print("✅ Test successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
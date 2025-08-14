#!/usr/bin/env python3
"""
Verify persistence is working in the actual brain server.
"""

import time
import subprocess
import sys
from pathlib import Path

print("Starting brain server to verify persistence...\n")

# Start brain server
process = subprocess.Popen(
    [sys.executable, "brain.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("Monitoring brain server output for 30 seconds...\n")
print("=" * 60)

start_time = time.time()
save_count = 0

try:
    while time.time() - start_time < 30:
        line = process.stdout.readline()
        if line:
            line = line.strip()
            # Look for save messages
            if "save" in line.lower() or "💾" in line:
                print(f"SAVE: {line}")
                save_count += 1
            elif "persistence" in line.lower():
                print(f"PERSISTENCE: {line}")
            elif "session" in line.lower():
                print(f"SESSION: {line}")
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    process.terminate()
    process.wait()

print("=" * 60)
print(f"\nSave events detected: {save_count}")

# Check brain_memory folder
memory_path = Path("./brain_memory")
if memory_path.exists():
    files = list(memory_path.glob("*.pt.gz"))
    print(f"\nBrain memory files: {len(files)}")
    for f in sorted(files)[-5:]:  # Show last 5
        size = f.stat().st_size / 1024 / 1024  # MB
        print(f"  - {f.name} ({size:.2f} MB)")
else:
    print("\n⚠️ Brain memory folder doesn't exist")

print("\n✅ Verification complete")
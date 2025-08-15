#!/usr/bin/env python3
"""
Run this ON THE ROBOT to check which protocol version it has.
"""

import sys
import os

print("Robot Protocol Version Check")
print("=" * 60)

# Check if brain_client.py exists and has magic bytes
brain_client_paths = [
    "client_picarx/src/brainstem/brain_client.py",
    "src/brainstem/brain_client.py",
    "brainstem/brain_client.py",
    "brain_client.py"
]

found = False
for path in brain_client_paths:
    if os.path.exists(path):
        print(f"\nFound brain_client.py at: {path}")
        found = True
        
        # Read the file and check for magic bytes
        with open(path, 'r') as f:
            content = f.read()
            
        # Check for key indicators
        if "MAGIC_BYTES = 0xDEADBEEF" in content:
            print("✅ HAS magic bytes support (NEW protocol)")
        else:
            print("❌ NO magic bytes support (OLD protocol)")
            
        if "MAX_REASONABLE_VECTOR_LENGTH = 10_000_000" in content:
            print("✅ Supports large vision data (10M limit)")
        elif "MAX_REASONABLE_VECTOR_LENGTH = 100_000" in content:
            print("❌ Limited to 100K values (needs update)")
        elif "max_vector_size = 10_000_000" in content:
            print("✅ Has 10M vector size limit")
        elif "max_vector_size: int = 1024" in content:
            print("❌ Limited to 1024 values (VERY OLD)")
            
        # Check the encoding method
        if "struct.pack('!IIBI', self.MAGIC_BYTES" in content:
            print("✅ Encodes messages with magic bytes")
        elif "struct.pack('!IIBI', MAGIC_BYTES" in content:
            print("✅ Encodes messages with magic bytes")
        elif "struct.pack('!IBI'" in content:
            print("❌ Encodes WITHOUT magic bytes (old format)")
        
        # Show when file was last modified
        import os.path
        import datetime
        mtime = os.path.getmtime(path)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nFile last modified: {mtime_str}")
        
        break

if not found:
    print("❌ Could not find brain_client.py")
    print("   Current directory:", os.getcwd())
    print("   Files here:", os.listdir('.'))

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("If you see ❌ marks above, the robot needs updating:")
print("  cd ~/em-brain")
print("  git pull")
print("  # Then restart the robot client")
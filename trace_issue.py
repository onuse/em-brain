#!/usr/bin/env python3
"""Trace exactly where the slowdown is."""

import torch
import time
import sys

print("Tracing performance issue...")

# Test imports
print("\n1. Testing imports...")
try:
    from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain
    print("   ✓ Brain imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test initialization
print("\n2. Testing initialization...")
try:
    brain = TrulyMinimalBrain(
        sensory_dim=12,
        motor_dim=6,
        spatial_size=16,  # Start small
        channels=32,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        quiet_mode=True
    )
    print("   ✓ Small brain created")
except Exception as e:
    print(f"   ✗ Init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test processing
print("\n3. Testing process cycle...")
sensors = [0.5] * 12

try:
    start = time.perf_counter()
    print("   Starting process...")
    motors, telemetry = brain.process(sensors)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   ✓ Process completed in {elapsed:.1f}ms")
except Exception as e:
    print(f"   ✗ Process failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now test with larger size
print("\n4. Testing with medium size (32³×64)...")
try:
    brain2 = TrulyMinimalBrain(
        sensory_dim=12,
        motor_dim=6,
        spatial_size=32,
        channels=64,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        quiet_mode=True
    )
    
    start = time.perf_counter()
    motors, telemetry = brain2.process(sensors)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   ✓ Medium brain: {elapsed:.1f}ms")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test largest size with timeout
print("\n5. Testing with full size (96³×192)...")
print("   Creating brain...")

try:
    brain3 = TrulyMinimalBrain(
        sensory_dim=12,
        motor_dim=6,
        spatial_size=96,
        channels=192,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        quiet_mode=True
    )
    print("   ✓ Large brain created")
    
    print("   Processing (this may take a moment)...")
    start = time.perf_counter()
    
    # Add timeout using alarm if on Linux
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Process timed out")
    
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
    
    try:
        motors, telemetry = brain3.process(sensors)
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
        elapsed = (time.perf_counter() - start) * 1000
        print(f"   ✓ Large brain processed in {elapsed:.1f}ms")
        
        # Try a few more cycles
        print("\n   Testing multiple cycles:")
        for i in range(3):
            start = time.perf_counter()
            motors, telemetry = brain3.process(sensors)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"     Cycle {i+1}: {elapsed:.1f}ms")
            
    except TimeoutError:
        print("   ✗ Process timed out after 30 seconds")
        print("   This indicates a serious performance issue")
        
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
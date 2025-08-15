#!/usr/bin/env python3
"""
Test complete system with brain server.

This starts a brain server and robot client to test the full pipeline.
"""

import subprocess
import time
import sys
import os
import signal

print("=" * 60)
print("TESTING COMPLETE SYSTEM WITH BRAIN SERVER")
print("=" * 60)

# Start brain server
print("\n1. Starting brain server...")
brain_process = subprocess.Popen(
    [sys.executable, "server/brain.py", "--safe-mode"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Wait for brain to start
print("   Waiting for brain server to initialize...")
time.sleep(5)

# Check if brain started
brain_started = False
for i in range(10):
    line = brain_process.stdout.readline()
    if line:
        print(f"   Brain: {line.strip()}")
        if "Brain server ready" in line or "listening" in line.lower():
            brain_started = True
            break

if not brain_started:
    print("   ⚠️ Brain server may not have started properly")

# Start robot client pointing to localhost
print("\n2. Starting robot client...")
robot_process = subprocess.Popen(
    [sys.executable, "client_picarx/picarx_robot.py", "--brain-host", "localhost", "--rate", "5"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Monitor both processes for 10 seconds
print("\n3. Running system for 10 seconds...")
print("-" * 40)

start_time = time.time()
error_count = 0
success_count = 0

try:
    while time.time() - start_time < 10:
        # Read from brain
        brain_line = brain_process.stdout.readline()
        if brain_line:
            brain_line = brain_line.strip()
            if "error" in brain_line.lower():
                print(f"❌ Brain: {brain_line}")
                error_count += 1
            elif "session" in brain_line.lower() and "processing" in brain_line.lower():
                print(f"✅ Brain: {brain_line}")
                success_count += 1
            elif brain_line:
                print(f"   Brain: {brain_line}")
        
        # Read from robot
        robot_line = robot_process.stdout.readline()
        if robot_line:
            robot_line = robot_line.strip()
            if robot_line:
                print(f"   Robot: {robot_line}")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nInterrupted by user")

# Stop processes
print("\n4. Stopping processes...")
robot_process.terminate()
brain_process.terminate()

# Wait for cleanup
time.sleep(2)
robot_process.wait()
brain_process.wait()

# Summary
print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print(f"\nResults:")
print(f"  Successful brain cycles: {success_count}")
print(f"  Errors encountered: {error_count}")

if success_count > 0 and error_count == 0:
    print("\n✅ SYSTEM WORKING PERFECTLY!")
elif success_count > 0 and error_count > 0:
    print("\n⚠️ System working but with some errors")
else:
    print("\n❌ System not working properly")

print("\nNext steps:")
print("  1. Deploy to robot with: ./deploy.sh")
print("  2. Start brain server on remote machine")
print("  3. Run on robot: sudo python3 picarx_robot.py")
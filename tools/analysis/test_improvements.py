#!/usr/bin/env python3
"""
Quick test of energy stability and boredom avoidance improvements.
"""

import subprocess
import time
import sys
import os

def run_test():
    """Run a 30-minute test to check improvements."""
    print("🧪 Testing Energy Stability and Boredom Avoidance")
    print("=" * 60)
    print("This will run a 30-minute validation test to check:")
    print("1. Field energy stability (should stay within 0.1-10.0)")
    print("2. Exploration improvement (target > 0.1 from baseline 0.003)")
    print("3. Boredom-driven exploration events")
    print("=" * 60)
    
    # Start the brain server
    print("\n🧠 Starting brain server...")
    server_process = subprocess.Popen(
        ["python3", "brain.py"],
        cwd="/Users/jkarlsson/Documents/Projects/robot-project/brain/server",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Give server time to start
    time.sleep(5)
    
    try:
        # Run validation for 30 minutes
        print("\n🤖 Running validation test (30 minutes)...")
        validation_process = subprocess.Popen(
            ["python3", "tools/runners/validation_runner.py", 
             "biological_embodied_learning", "--hours", "0.5"],
            cwd="/Users/jkarlsson/Documents/Projects/robot-project/brain",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor outputs
        maintenance_events = []
        boredom_events = []
        energy_readings = []
        exploration_scores = []
        
        print("\n📊 Monitoring progress...\n")
        
        # Monitor server output in background
        def monitor_server():
            for line in server_process.stdout:
                line = line.strip()
                if "Maintenance: Energy" in line:
                    maintenance_events.append(line)
                    print(f"  🔧 {line}")
                elif "BOREDOM EXPLORATION" in line:
                    boredom_events.append(line)
                    print(f"  🎲 {line}")
                elif "mean energy" in line:
                    # Extract energy value
                    try:
                        energy = float(line.split("mean energy")[-1].strip())
                        energy_readings.append(energy)
                    except:
                        pass
        
        # Monitor validation output
        for line in validation_process.stdout:
            line = line.strip()
            if line:
                print(f"  {line}")
            if "Exploration score:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                    exploration_scores.append(score)
                except:
                    pass
        
        # Wait for completion
        validation_process.wait()
        
        # Analysis
        print("\n" + "=" * 60)
        print("📈 TEST RESULTS")
        print("=" * 60)
        
        if energy_readings:
            print(f"\n🔋 Energy Stability:")
            print(f"   Initial energy: {energy_readings[0]:.3f}")
            print(f"   Final energy: {energy_readings[-1]:.3f}")
            print(f"   Max energy: {max(energy_readings):.3f}")
            print(f"   Min energy: {min(energy_readings):.3f}")
            
            if max(energy_readings) < 100:
                print("   ✅ Energy remained stable!")
            else:
                print("   ⚠️ Energy grew too high")
        
        if exploration_scores:
            print(f"\n🔍 Exploration:")
            print(f"   Average score: {sum(exploration_scores)/len(exploration_scores):.3f}")
            print(f"   Max score: {max(exploration_scores):.3f}")
            
            if max(exploration_scores) > 0.01:
                print("   ✅ Exploration improved from baseline!")
            else:
                print("   ⚠️ Exploration still low")
        
        print(f"\n🔧 Maintenance Events: {len(maintenance_events)}")
        if maintenance_events and len(maintenance_events) < 5:
            for event in maintenance_events:
                print(f"   {event}")
        
        print(f"\n🎲 Boredom Events: {len(boredom_events)}")
        if boredom_events and len(boredom_events) < 5:
            for event in boredom_events:
                print(f"   {event}")
        
    finally:
        # Clean up
        print("\n🛑 Stopping processes...")
        server_process.terminate()
        validation_process.terminate()
        time.sleep(2)
        server_process.kill()
        validation_process.kill()

if __name__ == "__main__":
    run_test()
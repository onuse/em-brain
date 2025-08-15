#!/usr/bin/env python3
"""
Test battery streaming - Day 1 of multi-stream implementation.

Run on brain server machine:
    python3 test_battery_stream.py

Then on robot:
    python3 client_picarx/src/streams/battery_stream.py BRAIN_IP

You should see battery voltage arriving!
"""

import sys
import os

# Add server src to path
sys.path.insert(0, 'server/src')

from streams.sensor_listeners import BatteryListener
import time


def main():
    print("🔋 Battery Stream Test")
    print("=" * 50)
    
    # Try UDP first (preferred)
    print("\nStarting UDP battery listener on port 10004...")
    listener = BatteryListener(port=10004, use_tcp=False)
    listener.start()
    
    print("\nWaiting for battery data...")
    print("On the robot, run:")
    print("  python3 client_picarx/src/streams/battery_stream.py BRAIN_IP")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        last_report = 0
        while True:
            voltage, age = listener.get_latest()
            
            # Report every 2 seconds if we have data
            if age < 10 and time.time() - last_report > 2:
                print(f"🔋 Battery: {voltage:.2f}V (updated {age:.1f}s ago)")
                
                # Check battery status
                if voltage < 6.5:
                    print("  ⚠️  LOW BATTERY WARNING!")
                elif voltage > 8.0:
                    print("  ✓ Fully charged")
                else:
                    percent = (voltage - 6.5) / (8.4 - 6.5) * 100
                    print(f"  ≈ {percent:.0f}% charge")
                
                last_report = time.time()
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    listener.stop()
    
    print("\n" + "=" * 50)
    print(f"📊 Statistics:")
    print(f"  Packets received: {listener.packet_count}")
    print(f"  Latest voltage: {listener.latest_voltage:.2f}V")
    
    if listener.packet_count > 0:
        print("\n✅ SUCCESS! Battery streaming works!")
        print("\nNext steps:")
        print("1. Add ultrasonic stream (Day 2)")
        print("2. Add video stream (Day 3)")
        print("3. Remove from TCP (Day 4)")
    else:
        print("\n❌ No packets received")
        print("\nTroubleshooting:")
        print("1. Check firewall (UDP port 10004)")
        print("2. Verify brain IP on robot")
        print("3. Try TCP mode instead")


if __name__ == "__main__":
    main()
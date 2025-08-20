#!/usr/bin/env python3
"""
Quick diagnostic to check if the brain is actually processing
"""

import socket
import json
import time

print("Brain Status Check")
print("=" * 60)

# Try to connect to telemetry
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    sock.connect(('localhost', 9998))
    print("✓ Connected to telemetry port 9998")
    
    # Read for a few seconds
    print("\nListening for telemetry data...")
    print("-" * 40)
    
    sock.settimeout(0.5)  # Short timeout for reading
    start_time = time.time()
    message_count = 0
    last_cycle = 0
    
    buffer = ""
    while time.time() - start_time < 5:  # Listen for 5 seconds
        try:
            data = sock.recv(4096).decode('utf-8')
            if data:
                buffer += data
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            telemetry = json.loads(line)
                            message_count += 1
                            
                            # Show key metrics
                            cycle = telemetry.get('cycles', 0)
                            if cycle != last_cycle:
                                print(f"Cycle {cycle}:")
                                print(f"  Learning score: {telemetry.get('learning_score', 0):.1%}")
                                print(f"  Causal chains: {telemetry.get('causal_chains', 0)}")
                                print(f"  Field energy: {telemetry.get('field_energy', 0):.3f}")
                                last_cycle = cycle
                                
                        except json.JSONDecodeError:
                            pass
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Read error: {e}")
            break
    
    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"\nReceived {message_count} messages in {elapsed:.1f} seconds")
    
    if message_count > 0:
        rate = message_count / elapsed
        print(f"Rate: {rate:.1f} messages/second")
        
        if rate < 1:
            print("⚠️ Brain is processing very slowly")
            print("   This is normal for initial cycles with large field size")
        elif rate < 10:
            print("✓ Brain is processing at moderate speed")
        else:
            print("✓ Brain is processing quickly")
    else:
        print("❌ No telemetry received!")
        print("\nPossible issues:")
        print("1. Brain is stuck in long computation")
        print("2. Telemetry broadcasting issue")
        print("3. Brain hasn't started processing yet")
        
    sock.close()
    
except ConnectionRefusedError:
    print("❌ Cannot connect to telemetry port 9998")
    print("   Make sure server is running")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Diagnostics:")
print("- GPU at 100% = Brain IS computing")
print("- No telemetry = Likely still in first processing cycle")
print("- Enhanced brain with 48³×96 field can take 30-60 seconds per cycle initially")
print("- This should speed up after first few cycles")
print("\nRecommendations:")
print("1. Wait 2-3 minutes for first cycle to complete")
print("2. Consider using --target speed for faster initial processing")
print("3. Or reduce field size in EmergenceConfig")
print("=" * 60)
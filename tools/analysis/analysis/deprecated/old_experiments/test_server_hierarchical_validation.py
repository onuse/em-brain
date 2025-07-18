#!/usr/bin/env python3
"""
Server Hierarchical Validation Test

Connects to the brain server and runs a 5-minute validation test to verify:
1. Hierarchical clustering performance under real conditions
2. No performance regression from optimizations  
3. Scaling behavior as experiences accumulate
4. TCP protocol stability with hierarchical indexing
"""

import socket
import struct
import time
import math
import sys
import os

# Protocol constants (matching server MessageProtocol)
MSG_SENSORY_INPUT = 0
MSG_ACTION_OUTPUT = 1


def pack_sensory_message(sensory_vector):
    """Pack sensory input into server's expected binary format."""
    # Convert to float32 for consistent precision
    vector_data = struct.pack(f'{len(sensory_vector)}f', *sensory_vector)
    
    # Message length = type(1) + vector_length(4) + vector_data
    message_length = 1 + 4 + len(vector_data)
    
    # Pack message: length + type + vector_length + vector_data
    header = struct.pack('!IBI', message_length, MSG_SENSORY_INPUT, len(sensory_vector))
    
    return header + vector_data


def unpack_action_message(sock):
    """Unpack action message from server."""
    # Read message length first (4 bytes)
    length_data = _receive_exactly(sock, 4)
    message_length = struct.unpack('!I', length_data)[0]
    
    # Validate message length
    if message_length > 10000 or message_length <= 0:
        raise ValueError(f"Invalid message length: {message_length}")
    
    # Read the rest of the message
    message_data = _receive_exactly(sock, message_length)
    
    # Decode complete message (length + message_data)
    complete_data = length_data + message_data
    
    # Unpack header
    message_length, msg_type, vector_length = struct.unpack('!IBI', complete_data[:9])
    
    # Validate message
    if msg_type != MSG_ACTION_OUTPUT:
        raise ValueError(f"Expected action output, got message type {msg_type}")
    
    # Extract vector data
    vector_data = complete_data[9:9 + (vector_length * 4)]
    if len(vector_data) != vector_length * 4:
        raise ValueError("Incomplete vector data")
    
    # Unpack vector
    return list(struct.unpack(f'{vector_length}f', vector_data))


def _receive_exactly(sock, num_bytes):
    """Receive exactly num_bytes from socket."""
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly")
        data += chunk
    return data


def test_server_hierarchical_validation():
    """Run 5-minute hierarchical clustering validation with server."""
    print("🚀 SERVER HIERARCHICAL VALIDATION TEST")
    print("=" * 60)
    print("Duration: 5 minutes")
    print("Focus: Hierarchical clustering performance under real conditions")
    print("Goal: Verify no regression and scaling effectiveness")
    print()
    
    # Connect to server
    print("📡 Connecting to brain server...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 9999))
        print("✅ Connected to server successfully")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("💡 Make sure to start the server with: python3 brain_server.py")
        return
    
    # Test parameters
    test_duration = 5 * 60  # 5 minutes
    cycles_per_report = 50
    
    start_time = time.time()
    cycle_count = 0
    performance_history = []
    error_count = 0
    
    print(f"🎯 Starting {test_duration//60}-minute validation test...")
    print("📊 Will report performance every 50 cycles")
    print()
    
    try:
        while time.time() - start_time < test_duration:
            cycle_start_time = time.time()
            
            # Create diverse sensory input to encourage clustering
            angle = cycle_count * 0.02
            experience_type = cycle_count % 5  # Five types of experiences
            type_offset = experience_type * 1.2
            
            sensory_input = [
                0.5 + 0.3 * math.sin(angle + type_offset),
                0.5 + 0.3 * math.cos(angle * 1.1 + type_offset), 
                0.5 + 0.2 * math.sin(angle * 0.7 + type_offset),
                0.5 + 0.2 * math.cos(angle * 1.3 + type_offset)
            ]
            
            try:
                # Send sensory input using correct protocol
                message = pack_sensory_message(sensory_input)
                sock.sendall(message)
                
                # Receive action using correct protocol
                action = unpack_action_message(sock)
                
                cycle_time = (time.time() - cycle_start_time) * 1000
                cycle_count += 1
                
                # Track performance
                performance_history.append(cycle_time)
                
                # Report progress every N cycles
                if cycle_count % cycles_per_report == 0:
                    elapsed_time = time.time() - start_time
                    recent_times = performance_history[-cycles_per_report:]
                    avg_recent = sum(recent_times) / len(recent_times)
                    
                    # Calculate performance trend
                    if len(performance_history) >= cycles_per_report * 2:
                        early_times = performance_history[:cycles_per_report]
                        avg_early = sum(early_times) / len(early_times)
                        trend = ((avg_recent - avg_early) / avg_early) * 100
                        trend_str = f"{trend:+.1f}%"
                    else:
                        trend_str = "calculating..."
                    
                    print(f"📊 Cycle {cycle_count}: {avg_recent:.1f}ms avg (trend: {trend_str}) "
                          f"[{elapsed_time:.0f}s elapsed]")
                    
                    # Check for performance regression
                    if len(performance_history) >= cycles_per_report * 2:
                        if trend > 50:  # >50% degradation
                            print(f"⚠️  Performance regression detected: {trend_str}")
                        elif trend < -10:  # >10% improvement
                            print(f"🎉 Performance improvement: {trend_str}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ Cycle {cycle_count} error: {e}")
                if error_count > 10:
                    print("🛑 Too many errors, stopping test")
                    break
                time.sleep(0.1)  # Brief pause on error
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        sock.close()
    
    # Analysis
    total_time = time.time() - start_time
    print(f"\n📈 TEST RESULTS ANALYSIS:")
    print("-" * 50)
    print(f"   Duration: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Total cycles: {cycle_count}")
    print(f"   Errors: {error_count}")
    
    if performance_history:
        avg_performance = sum(performance_history) / len(performance_history)
        min_performance = min(performance_history)
        max_performance = max(performance_history)
        
        print(f"   Average cycle time: {avg_performance:.1f}ms")
        print(f"   Best cycle time: {min_performance:.1f}ms")
        print(f"   Worst cycle time: {max_performance:.1f}ms")
        print(f"   Cycles per second: {cycle_count / total_time:.1f}")
        
        # Performance stability analysis
        if len(performance_history) >= 100:
            first_quarter = performance_history[:len(performance_history)//4]
            last_quarter = performance_history[-len(performance_history)//4:]
            
            avg_first = sum(first_quarter) / len(first_quarter)
            avg_last = sum(last_quarter) / len(last_quarter)
            
            overall_trend = ((avg_last - avg_first) / avg_first) * 100
            
            print(f"\n🔍 PERFORMANCE STABILITY:")
            print(f"   First quarter average: {avg_first:.1f}ms")
            print(f"   Last quarter average: {avg_last:.1f}ms")
            print(f"   Overall trend: {overall_trend:+.1f}%")
            
            if abs(overall_trend) < 5:
                print("   ✅ Stable performance (±5%)")
            elif overall_trend > 0:
                print(f"   ⚠️  Performance degradation: {overall_trend:.1f}%")
            else:
                print(f"   🚀 Performance improvement: {overall_trend:.1f}%")
        
        # Target analysis
        print(f"\n🎯 TARGET ANALYSIS:")
        target_time = 100  # 100ms target for biological timescale
        under_target = [t for t in performance_history if t < target_time]
        
        print(f"   Cycles under {target_time}ms: {len(under_target)}/{len(performance_history)} ({len(under_target)/len(performance_history)*100:.1f}%)")
        
        if avg_performance < target_time:
            print(f"   ✅ Average performance meets {target_time}ms target")
        else:
            print(f"   📊 Average {avg_performance:.1f}ms exceeds {target_time}ms target by {avg_performance - target_time:.1f}ms")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    if error_count == 0:
        print("   ✅ No connection errors - TCP protocol stable")
    else:
        print(f"   ⚠️  {error_count} errors occurred - check server logs")
    
    if performance_history and len(performance_history) >= 100:
        if overall_trend < 10:
            print("   ✅ Hierarchical clustering scaling effectively")
        else:
            print("   📊 Monitor hierarchical clustering performance")
            print("   💡 Consider checking server statistics for region count")
    
    print("\n🏁 Validation test completed!")


if __name__ == "__main__":
    test_server_hierarchical_validation()
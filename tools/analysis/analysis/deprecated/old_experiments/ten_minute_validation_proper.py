#!/usr/bin/env python3
"""
Ten Minute Validation with Proper TCP Protocol

Tests the complete brain system including TCP communication using the actual
protocol format used by the server.
"""

import sys
import os
import time
import math
import signal
import socket
import struct
from typing import Dict, List, Any, Tuple

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

class ProperTCPClient:
    """TCP client using the exact protocol format from the server."""
    
    # Message types from protocol.py
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.max_vector_size = 1024
        
    def connect(self):
        """Connect to brain server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Send handshake with robot capabilities
            capabilities = [4.0, 4.0, 1.0]  # 4D input, 4D output, version 1.0
            self._send_message(capabilities, self.MSG_HANDSHAKE)
            
            # Receive handshake response
            msg_type, response = self._receive_message()
            if msg_type == self.MSG_ERROR:
                print(f"Handshake failed: {response}")
                return False
            
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from brain server."""
        if self.socket:
            self.socket.close()
        self.connected = False
    
    def _encode_message(self, vector: List[float], msg_type: int) -> bytes:
        """Encode a vector with message header using exact protocol format."""
        if len(vector) > self.max_vector_size:
            raise ValueError(f"Vector too large: {len(vector)} > {self.max_vector_size}")
        
        # Convert to float32 for consistent precision
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Message length = type(1) + vector_length(4) + vector_data
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: length + type + vector_length + vector_data
        header = struct.pack('!IBI', message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def _decode_message(self, data: bytes) -> Tuple[int, List[float]]:
        """Decode received message using exact protocol format."""
        if len(data) < 9:  # Minimum message size
            raise ValueError("Message too short")
        
        # Unpack header
        message_length, msg_type, vector_length = struct.unpack('!IBI', data[:9])
        
        # Validate message
        expected_length = 1 + 4 + (vector_length * 4)
        if message_length != expected_length:
            raise ValueError(f"Invalid message length: {message_length} != {expected_length}")
        
        if vector_length > self.max_vector_size:
            raise ValueError(f"Vector too large: {vector_length}")
        
        # Extract vector data
        vector_data = data[9:9 + (vector_length * 4)]
        if len(vector_data) != vector_length * 4:
            raise ValueError("Incomplete vector data")
        
        # Unpack vector
        vector = list(struct.unpack(f'{vector_length}f', vector_data))
        
        return msg_type, vector
    
    def _send_message(self, vector: List[float], msg_type: int):
        """Send a message using the proper protocol."""
        if not self.connected:
            raise RuntimeError("Not connected")
        
        message = self._encode_message(vector, msg_type)
        self.socket.sendall(message)
    
    def _receive_message(self) -> Tuple[int, List[float]]:
        """Receive a complete message using the proper protocol."""
        if not self.connected:
            raise RuntimeError("Not connected")
        
        # Receive header (9 bytes)
        header_data = b''
        while len(header_data) < 9:
            chunk = self.socket.recv(9 - len(header_data))
            if not chunk:
                raise RuntimeError("Connection closed")
            header_data += chunk
        
        # Unpack header to get message length
        message_length, msg_type, vector_length = struct.unpack('!IBI', header_data)
        
        # Receive remaining data
        remaining_length = message_length - 5  # Already received type(1) + vector_length(4)
        remaining_data = b''
        while len(remaining_data) < remaining_length:
            chunk = self.socket.recv(remaining_length - len(remaining_data))
            if not chunk:
                raise RuntimeError("Connection closed")
            remaining_data += chunk
        
        # Decode the complete message
        complete_data = header_data + remaining_data
        return self._decode_message(complete_data)
    
    def process_sensory_input(self, sensory_input: List[float]) -> List[float]:
        """Send sensory input and receive predicted action."""
        if not self.connected:
            return None
        
        try:
            # Send sensory input
            self._send_message(sensory_input, self.MSG_SENSORY_INPUT)
            
            # Receive action response
            msg_type, action_vector = self._receive_message()
            
            if msg_type == self.MSG_ACTION_OUTPUT:
                return action_vector
            elif msg_type == self.MSG_ERROR:
                print(f"Error response: {action_vector}")
                return None
            else:
                print(f"Unexpected message type: {msg_type}")
                return None
                
        except Exception as e:
            print(f"Communication error: {e}")
            return None

class TenMinuteProperValidator:
    """
    10-minute validation using the proper TCP protocol.
    """
    
    def __init__(self):
        self.client = ProperTCPClient()
        self.test_duration = 600  # 10 minutes
        self.cycle_interval = 1.0  # 1 second per cycle (biological speed)
        self.running = True
        self.results = {
            'cycle_times': [],
            'tcp_times': [],
            'errors': [],
            'cycles_completed': 0
        }
        
    def run_validation(self):
        """Run 10-minute validation test."""
        print("🧪 TEN MINUTE PROPER TCP VALIDATION")
        print("=" * 60)
        print("Testing complete brain system with proper TCP protocol")
        print("Target: <100ms per cycle including TCP communication")
        print("\nConnecting to brain server...")
        
        # Connect to brain server
        if not self.client.connect():
            print("❌ Failed to connect to brain server")
            print("Please ensure server is running on port 9999")
            return
        
        print("✅ Connected to brain server with proper protocol")
        
        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        start_time = time.time()
        last_report_time = start_time
        cycle_count = 0
        
        try:
            while self.running and (time.time() - start_time) < self.test_duration:
                cycle_start = time.time()
                
                # Generate biological-like sensory input
                sensory_input = self._generate_sensory_input(cycle_count)
                
                # Process sensory input through TCP and time it
                tcp_start = time.time()
                predicted_action = self.client.process_sensory_input(sensory_input)
                tcp_time = (time.time() - tcp_start) * 1000
                
                if predicted_action:
                    # Record cycle performance
                    total_cycle_time = tcp_time
                    self.results['cycle_times'].append(total_cycle_time)
                    self.results['tcp_times'].append(tcp_time)
                    
                    # Report progress every 30 seconds
                    if time.time() - last_report_time >= 30:
                        self._report_progress(cycle_count, start_time)
                        last_report_time = time.time()
                    
                    cycle_count += 1
                    self.results['cycles_completed'] = cycle_count
                else:
                    self.results['errors'].append(f"Cycle {cycle_count}: No prediction returned")
                
                # Wait for next biological cycle
                elapsed = time.time() - cycle_start
                if elapsed < self.cycle_interval:
                    time.sleep(self.cycle_interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            self.results['errors'].append(str(e))
        finally:
            self._generate_final_report(time.time() - start_time)
            self.client.disconnect()
    
    def _generate_sensory_input(self, cycle: int) -> List[float]:
        """Generate biological-like sensory patterns."""
        # Create naturalistic sensory patterns with multiple frequencies
        time_component = cycle * 0.1
        return [
            0.5 + 0.3 * math.sin(time_component),
            0.5 + 0.3 * math.cos(time_component * 1.1),
            0.5 + 0.2 * math.sin(time_component * 0.7),
            0.5 + 0.2 * math.cos(time_component * 1.3)
        ]
    
    def _report_progress(self, cycles: int, start_time: float):
        """Report test progress."""
        elapsed = time.time() - start_time
        remaining = self.test_duration - elapsed
        
        # Calculate performance metrics
        recent_cycles = self.results['cycle_times'][-30:] if len(self.results['cycle_times']) >= 30 else self.results['cycle_times']
        avg_cycle_time = sum(recent_cycles) / len(recent_cycles) if recent_cycles else 0
        
        recent_tcp = self.results['tcp_times'][-30:] if len(self.results['tcp_times']) >= 30 else self.results['tcp_times']
        avg_tcp_time = sum(recent_tcp) / len(recent_tcp) if recent_tcp else 0
        
        print(f"\n📊 Progress Report - {elapsed/60:.1f} minutes elapsed")
        print(f"   Cycles completed: {cycles}")
        print(f"   Recent avg cycle time: {avg_cycle_time:.1f}ms")
        print(f"   Recent avg TCP time: {avg_tcp_time:.1f}ms")
        print(f"   Real-time capable: {'✅ YES' if avg_cycle_time < 100 else '❌ NO'}")
        print(f"   Time remaining: {remaining/60:.1f} minutes")
    
    def _generate_final_report(self, total_elapsed: float):
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("🏁 TEN MINUTE PROPER TCP VALIDATION COMPLETE")
        print("=" * 60)
        
        # Overall performance
        all_cycle_times = self.results['cycle_times']
        all_tcp_times = self.results['tcp_times']
        
        if all_cycle_times:
            avg_cycle = sum(all_cycle_times) / len(all_cycle_times)
            avg_tcp = sum(all_tcp_times) / len(all_tcp_times)
            min_cycle = min(all_cycle_times)
            max_cycle = max(all_cycle_times)
            
            # Performance over time
            first_minute = all_cycle_times[:60] if len(all_cycle_times) >= 60 else all_cycle_times
            last_minute = all_cycle_times[-60:] if len(all_cycle_times) >= 60 else all_cycle_times
            
            first_avg = sum(first_minute) / len(first_minute) if first_minute else 0
            last_avg = sum(last_minute) / len(last_minute) if last_minute else 0
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Total cycles: {self.results['cycles_completed']}")
            print(f"   Total time: {total_elapsed/60:.1f} minutes")
            print(f"   Average cycle time: {avg_cycle:.1f}ms")
            print(f"   Average TCP time: {avg_tcp:.1f}ms")
            print(f"   Min cycle time: {min_cycle:.1f}ms")
            print(f"   Max cycle time: {max_cycle:.1f}ms")
            
            print(f"\n📈 PERFORMANCE STABILITY:")
            print(f"   First minute avg: {first_avg:.1f}ms")
            print(f"   Last minute avg: {last_avg:.1f}ms")
            if first_avg > 0:
                degradation = ((last_avg - first_avg) / first_avg) * 100
                print(f"   Performance change: {degradation:+.1f}%")
            
            # Check real-time capability
            cycles_over_100ms = sum(1 for t in all_cycle_times if t > 100)
            percentage_over = (cycles_over_100ms / len(all_cycle_times)) * 100
            
            print(f"\n🎯 REAL-TIME PERFORMANCE:")
            print(f"   Cycles under 100ms: {len(all_cycle_times) - cycles_over_100ms}/{len(all_cycle_times)} ({100-percentage_over:.1f}%)")
            print(f"   Real-time capable: {'✅ YES' if percentage_over < 5 else '⚠️ MARGINAL' if percentage_over < 10 else '❌ NO'}")
            
            # Error summary
            if self.results['errors']:
                print(f"\n⚠️  ERRORS ENCOUNTERED: {len(self.results['errors'])}")
                for error in self.results['errors'][:5]:
                    print(f"   - {error}")
                if len(self.results['errors']) > 5:
                    print(f"   ... and {len(self.results['errors']) - 5} more")
            else:
                print(f"\n✅ NO ERRORS ENCOUNTERED")
            
            # Final verdict
            print(f"\n🏆 FINAL VERDICT:")
            if avg_cycle < 100 and percentage_over < 5:
                print("   ✅ EXCELLENT: Complete system maintains real-time performance!")
                print("   🎉 TCP communication + brain optimizations working perfectly")
            elif avg_cycle < 100 and percentage_over < 10:
                print("   ✅ GOOD: System mostly maintains real-time performance")
                print("   Minor tuning may help")
            else:
                print("   ⚠️  NEEDS WORK: Performance issues detected")
                print("   May be TCP overhead or brain processing bottlenecks")
        else:
            print("❌ No performance data collected")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n⚠️  Stopping test...")
        self.running = False

def main():
    """Run 10-minute validation."""
    print("🚀 Starting 10-minute proper TCP validation")
    print("This test uses the exact protocol the server expects")
    print("Press Ctrl+C to stop early\n")
    
    validator = TenMinuteProperValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
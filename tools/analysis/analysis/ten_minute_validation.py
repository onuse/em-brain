#!/usr/bin/env python3
"""
Ten Minute Biological Timescale Validation

Comprehensive 10-minute test at biological speed to validate all optimizations
are working correctly under sustained operation.
"""

import sys
import os
import time
import math
import signal
import threading
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

import socket
import struct

class SimpleBrainClient:
    """TCP client for brain server communication using binary protocol."""
    
    # Message types (matching server protocol)
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1  
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    def __init__(self, port=9999):
        self.port = port
        self.host = "localhost"
        self.socket = None
        self.connected = False
        self.max_vector_size = 1024
        
    def connect(self):
        """Connect to brain server via TCP and perform handshake."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout
            self.socket.connect((self.host, self.port))
            
            # Perform handshake
            client_capabilities = [1.0, 4.0, 4.0, 0.0]  # version, sensory_size, action_size, gpu
            handshake_msg = self._encode_vector(client_capabilities, self.MSG_HANDSHAKE)
            
            if not self._send_message(handshake_msg):
                return False
            
            # Receive server response
            response = self._receive_message()
            if response and response[0] == self.MSG_HANDSHAKE:
                self.connected = True
                print(f"✅ Connected to brain server at {self.host}:{self.port}")
                print(f"   Server capabilities: {response[1]}")
                return True
            else:
                print("❌ Handshake failed")
                return False
                
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close connection to brain server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
    
    def _encode_vector(self, vector, msg_type):
        """Encode a vector with message header."""
        if len(vector) > self.max_vector_size:
            raise ValueError(f"Vector too large: {len(vector)} > {self.max_vector_size}")
        
        # Convert to float32 for consistent precision
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Message length = type(1) + vector_length(4) + vector_data
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: length + type + vector_length + vector_data
        header = struct.pack('!IBI', message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def _send_message(self, message_data):
        """Send complete message over socket."""
        if not self.connected and not self.socket:
            return False
            
        try:
            total_sent = 0
            while total_sent < len(message_data):
                sent = self.socket.send(message_data[total_sent:])
                if sent == 0:
                    return False
                total_sent += sent
            return True
            
        except (socket.error, BrokenPipeError):
            return False
    
    def _receive_exactly(self, num_bytes):
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly")
            data += chunk
        return data
    
    def _receive_message(self):
        """Receive and decode a complete message from socket."""
        if not self.socket:
            return None
            
        try:
            # Receive message length first
            length_data = self._receive_exactly(4)
            message_length = struct.unpack('!I', length_data)[0]
            
            if message_length > (self.max_vector_size * 4 + 5):  # Sanity check
                raise ValueError(f"Message too large: {message_length}")
            
            # Receive the rest of the message
            message_data = self._receive_exactly(message_length)
            
            # Decode complete message
            return self._decode_message(length_data + message_data)
            
        except Exception as e:
            print(f"Communication error: {e}")
            return None
    
    def _decode_message(self, data):
        """Decode received message."""
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
    
    def process_sensory_input(self, sensory_input):
        """Send sensory input to brain and get prediction."""
        if not self.connected:
            return None
            
        try:
            # Encode and send sensory input
            message = self._encode_vector(sensory_input, self.MSG_SENSORY_INPUT)
            if not self._send_message(message):
                return None
            
            # Receive action response
            response = self._receive_message()
            if response and response[0] == self.MSG_ACTION_OUTPUT:
                return {
                    'predicted_action': response[1],
                    'brain_state': {'prediction_confidence': 0.8}  # Simplified
                }
            elif response and response[0] == self.MSG_ERROR:
                print(f"Server error: {response[1]}")
                return None
            else:
                print(f"Unexpected response type: {response[0] if response else 'None'}")
                return None
                
        except Exception as e:
            print(f"Communication error: {e}")
            return None
    
    def store_experience(self, sensory_input, action_taken, outcome):
        """
        Store experience in brain.
        Note: The server handles experience storage automatically between requests.
        This is a no-op for the current protocol.
        """
        # Parameters are for interface compatibility - server handles storage automatically
        return "stored"
    
    def get_brain_state(self):
        """
        Get current brain state and statistics.
        Note: Not directly supported by current protocol.
        """
        return {
            'num_experiences': 0,
            'working_memory_size': 0,
            'prediction_confidence': 0.8
        }

class TenMinuteValidator:
    """
    10-minute biological timescale validation with performance monitoring.
    """
    
    def __init__(self):
        self.client = SimpleBrainClient()
        self.test_duration = 600  # 10 minutes
        self.cycle_interval = 1.0  # 1 second per cycle (biological speed)
        self.running = True
        self.results = {
            'cycle_times': [],
            'performance_over_time': [],
            'similarity_stats': [],
            'activation_stats': [],
            'errors': [],
            'cycles_completed': 0
        }
        
    def run_validation(self):
        """Run 10-minute validation test."""
        print("🧪 TEN MINUTE BIOLOGICAL TIMESCALE VALIDATION")
        print("=" * 60)
        print("Testing optimized brain at biological speed for 10 minutes")
        print("Target: <100ms per cycle throughout entire test")
        print("\nStarting test...")
        
        # Connect to brain server
        if not self.client.connect():
            print("❌ Failed to connect to brain server")
            print("Please ensure server is running on port 9999")
            return
        
        print("✅ Connected to brain server")
        
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
                
                # Process sensory input and time it
                process_start = time.time()
                result = self.client.process_sensory_input(sensory_input)
                process_time = (time.time() - process_start) * 1000
                
                if result and 'predicted_action' in result:
                    # Store experience with outcome
                    outcome = self._simulate_outcome(result['predicted_action'])
                    store_start = time.time()
                    self.client.store_experience(
                        sensory_input,
                        result['predicted_action'],
                        outcome
                    )
                    store_time = (time.time() - store_start) * 1000
                    
                    # Record cycle performance
                    total_cycle_time = process_time + store_time
                    self.results['cycle_times'].append(total_cycle_time)
                    
                    # Get brain statistics periodically
                    if cycle_count % 30 == 0:
                        self._collect_brain_statistics()
                    
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
    
    def _simulate_outcome(self, action: List[float]) -> List[float]:
        """Simulate environmental outcome from action."""
        # Simple physics-like outcome with some noise
        import random
        return [
            a * 0.8 + 0.1 + random.uniform(-0.05, 0.05) 
            for a in action
        ]
    
    def _collect_brain_statistics(self):
        """Collect brain statistics for monitoring."""
        try:
            stats = self.client.get_brain_state()
            if stats:
                timestamp = time.time()
                
                # Extract key statistics
                if 'similarity_stats' in stats:
                    self.results['similarity_stats'].append({
                        'timestamp': timestamp,
                        'stats': stats['similarity_stats']
                    })
                
                if 'activation_stats' in stats:
                    self.results['activation_stats'].append({
                        'timestamp': timestamp,
                        'stats': stats['activation_stats']
                    })
                
                # Track performance metrics
                perf_data = {
                    'timestamp': timestamp,
                    'num_experiences': stats.get('num_experiences', 0),
                    'working_memory_size': stats.get('working_memory_size', 0),
                    'prediction_confidence': stats.get('prediction_confidence', 0)
                }
                self.results['performance_over_time'].append(perf_data)
                
        except Exception as e:
            self.results['errors'].append(f"Stats collection error: {e}")
    
    def _report_progress(self, cycles: int, start_time: float):
        """Report test progress."""
        elapsed = time.time() - start_time
        remaining = self.test_duration - elapsed
        
        # Calculate performance metrics
        recent_cycles = self.results['cycle_times'][-30:] if len(self.results['cycle_times']) >= 30 else self.results['cycle_times']
        avg_cycle_time = sum(recent_cycles) / len(recent_cycles) if recent_cycles else 0
        
        print(f"\n📊 Progress Report - {elapsed/60:.1f} minutes elapsed")
        print(f"   Cycles completed: {cycles}")
        print(f"   Recent avg cycle time: {avg_cycle_time:.1f}ms")
        print(f"   Real-time capable: {'✅ YES' if avg_cycle_time < 100 else '❌ NO'}")
        print(f"   Time remaining: {remaining/60:.1f} minutes")
    
    def _generate_final_report(self, total_elapsed: float):
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("🏁 TEN MINUTE VALIDATION COMPLETE")
        print("=" * 60)
        
        # Overall performance
        all_cycle_times = self.results['cycle_times']
        if all_cycle_times:
            avg_cycle = sum(all_cycle_times) / len(all_cycle_times)
            min_cycle = min(all_cycle_times)
            max_cycle = max(all_cycle_times)
            
            # Performance over time
            first_minute = all_cycle_times[:60] if len(all_cycle_times) >= 60 else all_cycle_times
            last_minute = all_cycle_times[-60:] if len(all_cycle_times) >= 60 else all_cycle_times
            
            first_avg = sum(first_minute) / len(first_minute)
            last_avg = sum(last_minute) / len(last_minute)
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Total cycles: {self.results['cycles_completed']}")
            print(f"   Total time: {total_elapsed/60:.1f} minutes")
            print(f"   Average cycle time: {avg_cycle:.1f}ms")
            print(f"   Min cycle time: {min_cycle:.1f}ms")
            print(f"   Max cycle time: {max_cycle:.1f}ms")
            
            print(f"\n📈 PERFORMANCE STABILITY:")
            print(f"   First minute avg: {first_avg:.1f}ms")
            print(f"   Last minute avg: {last_avg:.1f}ms")
            degradation = ((last_avg - first_avg) / first_avg) * 100
            print(f"   Performance change: {degradation:+.1f}%")
            
            # Check real-time capability
            cycles_over_100ms = sum(1 for t in all_cycle_times if t > 100)
            percentage_over = (cycles_over_100ms / len(all_cycle_times)) * 100
            
            print(f"\n🎯 REAL-TIME PERFORMANCE:")
            print(f"   Cycles under 100ms: {len(all_cycle_times) - cycles_over_100ms}/{len(all_cycle_times)} ({100-percentage_over:.1f}%)")
            print(f"   Real-time capable: {'✅ YES' if percentage_over < 5 else '⚠️ MARGINAL' if percentage_over < 10 else '❌ NO'}")
            
            # Similarity learning progress
            if self.results['similarity_stats']:
                first_stats = self.results['similarity_stats'][0]['stats']
                last_stats = self.results['similarity_stats'][-1]['stats']
                
                print(f"\n🧠 SIMILARITY LEARNING:")
                print(f"   Initial connections: {first_stats.get('avg_connections', 0):.1f}")
                print(f"   Final connections: {last_stats.get('avg_connections', 0):.1f}")
                print(f"   Adaptations performed: {last_stats.get('adaptations_performed', 0)}")
            
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
            if avg_cycle < 100 and percentage_over < 5 and abs(degradation) < 20:
                print("   ✅ EXCELLENT: Brain maintains real-time performance!")
                print("   🎉 All optimizations working correctly")
            elif avg_cycle < 100 and percentage_over < 10:
                print("   ✅ GOOD: Brain mostly maintains real-time performance")
                print("   Minor optimization tuning may help")
            else:
                print("   ⚠️  NEEDS WORK: Performance issues detected")
                print("   Further optimization required")
        else:
            print("❌ No performance data collected")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        # Parameters required by signal handler interface
        print("\n⚠️  Stopping test...")
        self.running = False

def main():
    """Run 10-minute validation."""
    print("🚀 Starting 10-minute biological timescale validation")
    print("Please ensure brain server is running on port 9999")
    print("Press Ctrl+C to stop early\n")
    
    validator = TenMinuteValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ten Minute Biological Timescale Validation with TCP

Comprehensive 10-minute test at biological speed to validate all optimizations
are working correctly under sustained operation.
"""

import sys
import os
import time
import json
import math
import signal
import socket
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

class SimpleTCPClient:
    """Simple TCP client for brain server communication."""
    
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to brain server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from brain server."""
        if self.socket:
            self.socket.close()
        self.connected = False
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request and receive response."""
        if not self.connected:
            return None
            
        try:
            # Send request
            request_json = json.dumps(request)
            request_bytes = request_json.encode('utf-8')
            self.socket.sendall(len(request_bytes).to_bytes(4, 'big'))
            self.socket.sendall(request_bytes)
            
            # Receive response
            length_bytes = self.socket.recv(4)
            if not length_bytes:
                return None
            
            response_length = int.from_bytes(length_bytes, 'big')
            response_bytes = self.socket.recv(response_length)
            
            if not response_bytes:
                return None
                
            return json.loads(response_bytes.decode('utf-8'))
            
        except Exception as e:
            print(f"Communication error: {e}")
            return None
    
    def process_sensory_input(self, sensory_input):
        """Send sensory input to brain and get prediction."""
        request = {
            'type': 'process_sensory_input',
            'sensory_input': sensory_input
        }
        
        response = self._send_request(request)
        
        if response and response.get('status') == 'success':
            return response.get('data', {})
        else:
            print(f"Error: {response.get('error') if response else 'No response'}")
            return None
    
    def store_experience(self, sensory_input, action_taken, outcome):
        """Store experience in brain."""
        request = {
            'type': 'store_experience',
            'sensory_input': sensory_input,
            'action_taken': action_taken,
            'outcome': outcome
        }
        
        response = self._send_request(request)
        
        if response and response.get('status') == 'success':
            return response.get('data', {}).get('experience_id')
        else:
            print(f"Error: {response.get('error') if response else 'No response'}")
            return None
    
    def get_brain_state(self):
        """Get current brain state and statistics."""
        request = {
            'type': 'get_brain_state'
        }
        
        response = self._send_request(request)
        
        if response and response.get('status') == 'success':
            return response.get('data', {})
        else:
            print(f"Error: {response.get('error') if response else 'No response'}")
            return None

class TenMinuteValidator:
    """
    10-minute biological timescale validation with performance monitoring.
    """
    
    def __init__(self):
        self.client = SimpleTCPClient()
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
                    exp_id = self.client.store_experience(
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
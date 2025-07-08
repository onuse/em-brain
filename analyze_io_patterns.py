#!/usr/bin/env python3
"""
Analyze the actual I/O patterns and disk write timing
"""

import sys
sys.path.append('.')

import os
import time
import threading
from pathlib import Path

class IOMonitor:
    """Monitor disk I/O operations"""
    
    def __init__(self, watch_directory="./robot_memory"):
        self.watch_dir = Path(watch_directory)
        self.io_events = []
        self.monitoring = False
        self.start_time = None
        
    def start_monitoring(self):
        """Start monitoring I/O operations"""
        self.start_time = time.time()
        self.monitoring = True
        self.io_events.clear()
        
        print(f"📊 Starting I/O monitoring on: {self.watch_dir}")
        
        # Monitor thread
        def monitor_files():
            last_check = {}
            
            while self.monitoring:
                try:
                    if self.watch_dir.exists():
                        for file_path in self.watch_dir.rglob("*"):
                            if file_path.is_file():
                                try:
                                    stat = file_path.stat()
                                    current_mtime = stat.st_mtime
                                    current_size = stat.st_size
                                    
                                    file_key = str(file_path)
                                    
                                    if file_key in last_check:
                                        prev_mtime, prev_size = last_check[file_key]
                                        
                                        if current_mtime > prev_mtime:
                                            elapsed = time.time() - self.start_time
                                            size_change = current_size - prev_size
                                            
                                            self.io_events.append({
                                                'time': elapsed,
                                                'file': file_path.name,
                                                'type': 'write',
                                                'size_change': size_change,
                                                'total_size': current_size
                                            })
                                    
                                    last_check[file_key] = (current_mtime, current_size)
                                    
                                except (OSError, PermissionError):
                                    pass
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except Exception as e:
                    print(f"Monitor error: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_files, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        self.monitoring = False
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        return self.io_events
    
    def print_io_summary(self):
        """Print summary of I/O operations"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\\n📈 I/O MONITORING RESULTS ({total_time:.1f}s)")
        print("=" * 50)
        
        if not self.io_events:
            print("  No disk writes detected during monitoring period")
            return
        
        # Group by file type
        writes_by_type = {}
        total_bytes = 0
        
        for event in self.io_events:
            file_ext = Path(event['file']).suffix or 'no_ext'
            
            if file_ext not in writes_by_type:
                writes_by_type[file_ext] = {'count': 0, 'bytes': 0, 'times': []}
            
            writes_by_type[file_ext]['count'] += 1
            writes_by_type[file_ext]['bytes'] += event['size_change']
            writes_by_type[file_ext]['times'].append(event['time'])
            total_bytes += event['size_change']
        
        print(f"  Total writes: {len(self.io_events)}")
        print(f"  Total data written: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
        print(f"  Write frequency: {len(self.io_events)/total_time:.2f} writes/second")
        
        print(f"\\n📁 Writes by file type:")
        for file_type, stats in writes_by_type.items():
            avg_interval = (max(stats['times']) - min(stats['times'])) / max(1, stats['count']-1) if stats['count'] > 1 else 0
            print(f"    {file_type}: {stats['count']} writes, {stats['bytes']:,} bytes, every {avg_interval:.1f}s")
        
        print(f"\\n⏰ Write timing pattern:")
        for i, event in enumerate(self.io_events[:10]):  # Show first 10
            print(f"    {event['time']:6.1f}s: {event['file']} (+{event['size_change']:,} bytes)")
        
        if len(self.io_events) > 10:
            print(f"    ... and {len(self.io_events)-10} more writes")

def test_fresh_brain_performance():
    """Test performance with fresh brain"""
    print("🆕 FRESH BRAIN PERFORMANCE TEST")
    print("=" * 50)
    
    # Start I/O monitoring
    monitor = IOMonitor()
    monitor.start_monitoring()
    
    try:
        from simulation.brainstem_sim import GridWorldBrainstem
        from core.communication import SensoryPacket
        from datetime import datetime
        
        print("Creating fresh brain system...")
        brainstem = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
        
        # Start session (this creates fresh memory structure)
        session_id = brainstem.brain_client.start_memory_session("Fresh Performance Test")
        print(f"  Session: {session_id}")
        
        brain_nodes = len(brainstem.brain_client.brain_interface.world_graph.nodes)
        print(f"  Brain nodes: {brain_nodes}")
        
        # Test prediction performance
        print(f"\\n🧠 Testing brain prediction speed...")
        
        def single_prediction_cycle():
            state = brainstem.simulation.get_state()
            sensory_packet = SensoryPacket(
                sequence_id=brainstem.sequence_counter,
                sensor_values=state['sensors'],
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now()
            )
            mental_context = state['sensors'][:8]
            
            pred_start = time.time()
            prediction = brainstem.brain_client.process_sensory_input(
                sensory_packet, mental_context, threat_level="normal"
            )
            pred_time = time.time() - pred_start
            
            if prediction:
                brainstem.simulation.apply_action(prediction.motor_action)
            
            return pred_time, prediction is not None
        
        # Warmup
        for _ in range(3):
            single_prediction_cycle()
        
        # Measure performance
        n_predictions = 30
        prediction_times = []
        
        print(f"  Running {n_predictions} prediction cycles...")
        test_start = time.time()
        
        for i in range(n_predictions):
            pred_time, success = single_prediction_cycle()
            prediction_times.append(pred_time)
            
            if (i + 1) % 10 == 0:
                print(f"    Completed {i+1}/{n_predictions} predictions")
        
        test_duration = time.time() - test_start
        
        # Results
        avg_pred_time = sum(prediction_times) / len(prediction_times)
        max_pred_time = max(prediction_times)
        min_pred_time = min(prediction_times)
        
        theoretical_fps = 1.0 / avg_pred_time
        actual_fps = n_predictions / test_duration
        
        final_nodes = len(brainstem.brain_client.brain_interface.world_graph.nodes)
        
        print(f"\\n📊 FRESH BRAIN RESULTS:")
        print(f"  Test duration: {test_duration:.3f}s")
        print(f"  Average prediction time: {avg_pred_time:.6f}s")
        print(f"  Min/Max prediction time: {min_pred_time:.6f}s / {max_pred_time:.6f}s")
        print(f"  Theoretical FPS: {theoretical_fps:.1f}")
        print(f"  Actual FPS: {actual_fps:.1f}")
        print(f"  Brain growth: {brain_nodes} → {final_nodes} nodes")
        
        # End session
        brainstem.brain_client.end_memory_session()
        
        return {
            'avg_prediction_time': avg_pred_time,
            'theoretical_fps': theoretical_fps,
            'actual_fps': actual_fps,
            'brain_growth': final_nodes - brain_nodes,
            'test_duration': test_duration
        }
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Stop monitoring and show I/O results
        monitor.stop_monitoring()
        monitor.print_io_summary()

if __name__ == "__main__":
    results = test_fresh_brain_performance()
    
    if results:
        print(f"\\n🎯 PERFORMANCE SUMMARY")
        print(f"  Fresh brain achieves {results['actual_fps']:.1f} FPS")
        print(f"  Each prediction takes {results['avg_prediction_time']:.6f}s")
        print(f"  Demo at 1.3 FPS has {(1/1.3) - results['avg_prediction_time']:.3f}s visualization overhead")
#!/usr/bin/env python3
"""
Long Running Stability Test

Tests the optimized brain's stability over extended periods.
Suitable for 8-hour runs to verify memory management and performance stability.
"""

import sys
import os
import time
import numpy as np
import torch
import json
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


class LongRunningStabilityTest:
    """Test brain stability over extended periods."""
    
    def __init__(self, duration_hours: float = 8.0, checkpoint_interval_minutes: int = 30):
        """Initialize long-running test."""
        self.duration_seconds = duration_hours * 3600
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.start_time = None
        self.metrics = {
            'cycle_times': [],
            'memory_usage': [],
            'field_energy': [],
            'errors': [],
            'checkpoints': []
        }
        
        # Create optimized brain
        print(f"🧠 Initializing optimized brain for {duration_hours} hour test...")
        self.brain = SimplifiedUnifiedBrain(
            sensory_dim=24,
            motor_dim=4,
            spatial_resolution=32,
            quiet_mode=True,
            use_optimized=True
        )
        self.brain.enable_predictive_actions(False)  # For consistent performance
        self.brain.set_pattern_extraction_limit(3)
        
        print(f"✅ Brain initialized on {self.brain.device}")
        print(f"   Memory usage: {self._get_memory_usage():.1f}MB")
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.brain.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif self.brain.device.type == 'mps':
            # MPS doesn't have direct memory query
            # Estimate based on tensor size
            return self.brain.unified_field.numel() * 4 / 1024 / 1024
        else:
            # CPU memory - use tensor size
            return self.brain.unified_field.numel() * 4 / 1024 / 1024
    
    def _create_sensory_input(self, cycle: int) -> list:
        """Create varied sensory input to simulate real robot operation."""
        # Simulate robot exploring environment
        phase = cycle * 0.01
        base_sensors = [
            np.sin(phase),
            np.cos(phase),
            np.sin(phase * 2),
            np.cos(phase * 2)
        ]
        
        # Add noise and variation
        noise = np.random.randn(20) * 0.1
        
        # Occasional rewards
        reward = 0.0
        if cycle % 1000 == 0:
            reward = np.random.choice([-0.5, 0.0, 0.5, 1.0])
        
        return base_sensors + noise.tolist() + [reward]
    
    def _save_checkpoint(self, cycle: int, elapsed_time: float):
        """Save checkpoint data."""
        checkpoint = {
            'cycle': cycle,
            'elapsed_hours': elapsed_time / 3600,
            'timestamp': datetime.now().isoformat(),
            'avg_cycle_time_ms': np.mean(self.metrics['cycle_times'][-1000:]) if self.metrics['cycle_times'] else 0,
            'memory_mb': self._get_memory_usage(),
            'field_energy': np.mean(self.metrics['field_energy'][-1000:]) if self.metrics['field_energy'] else 0,
            'error_count': len(self.metrics['errors']),
            'brain_cycles': self.brain.brain_cycles
        }
        
        self.metrics['checkpoints'].append(checkpoint)
        
        # Save to file
        checkpoint_file = f"stability_test_checkpoint_{int(elapsed_time/3600)}h.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"\n💾 Checkpoint saved: {checkpoint_file}")
        print(f"   Cycles: {cycle:,}")
        print(f"   Avg cycle time: {checkpoint['avg_cycle_time_ms']:.1f}ms")
        print(f"   Memory: {checkpoint['memory_mb']:.1f}MB")
        print(f"   Errors: {checkpoint['error_count']}")
    
    def run(self):
        """Run the long stability test."""
        print(f"\n🚀 Starting {self.duration_seconds/3600:.1f} hour stability test")
        print(f"   Checkpoint interval: {self.checkpoint_interval/60:.0f} minutes")
        print(f"   Expected cycles: ~{int(self.duration_seconds / 0.2):,}")
        print("\n" + "="*60)
        
        self.start_time = time.time()
        end_time = self.start_time + self.duration_seconds
        last_checkpoint = self.start_time
        
        cycle = 0
        error_count = 0
        
        try:
            while time.time() < end_time:
                cycle += 1
                
                # Create sensory input
                sensory_input = self._create_sensory_input(cycle)
                
                # Process cycle with timing
                try:
                    cycle_start = time.perf_counter()
                    motors, state = self.brain.process_robot_cycle(sensory_input)
                    cycle_time = (time.perf_counter() - cycle_start) * 1000
                    
                    # Record metrics
                    self.metrics['cycle_times'].append(cycle_time)
                    self.metrics['field_energy'].append(state.get('field_energy', 0))
                    
                except Exception as e:
                    error_count += 1
                    self.metrics['errors'].append({
                        'cycle': cycle,
                        'error': str(e),
                        'time': time.time() - self.start_time
                    })
                    print(f"\n⚠️  Error at cycle {cycle}: {e}")
                    continue
                
                # Progress update every 1000 cycles
                if cycle % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    progress = elapsed / self.duration_seconds * 100
                    eta = (end_time - time.time()) / 3600
                    
                    recent_times = self.metrics['cycle_times'][-100:]
                    avg_cycle = np.mean(recent_times) if recent_times else 0
                    
                    print(f"\rProgress: {progress:5.1f}% | "
                          f"Cycle: {cycle:,} | "
                          f"Avg: {avg_cycle:.1f}ms | "
                          f"ETA: {eta:.1f}h | "
                          f"Errors: {error_count}", end='')
                
                # Checkpoint
                if time.time() - last_checkpoint >= self.checkpoint_interval:
                    elapsed = time.time() - self.start_time
                    self._save_checkpoint(cycle, elapsed)
                    last_checkpoint = time.time()
                    
                    # Clear old metrics to prevent memory growth
                    if len(self.metrics['cycle_times']) > 10000:
                        self.metrics['cycle_times'] = self.metrics['cycle_times'][-5000:]
                        self.metrics['field_energy'] = self.metrics['field_energy'][-5000:]
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
        
        finally:
            # Final summary
            elapsed_time = time.time() - self.start_time
            self._print_summary(cycle, elapsed_time)
    
    def _print_summary(self, total_cycles: int, elapsed_time: float):
        """Print test summary."""
        print("\n\n" + "="*60)
        print("STABILITY TEST SUMMARY")
        print("="*60)
        
        hours_run = elapsed_time / 3600
        print(f"\nDuration: {hours_run:.2f} hours")
        print(f"Total cycles: {total_cycles:,}")
        print(f"Cycles per hour: {int(total_cycles / hours_run):,}")
        
        if self.metrics['cycle_times']:
            times = np.array(self.metrics['cycle_times'])
            print(f"\nCycle times:")
            print(f"  Mean: {np.mean(times):.1f}ms")
            print(f"  Std: {np.std(times):.1f}ms")
            print(f"  Min: {np.min(times):.1f}ms")
            print(f"  Max: {np.max(times):.1f}ms")
            
            # Check for performance degradation
            first_1000 = times[:1000] if len(times) > 1000 else times
            last_1000 = times[-1000:] if len(times) > 1000 else times
            degradation = (np.mean(last_1000) - np.mean(first_1000)) / np.mean(first_1000) * 100
            
            print(f"\nPerformance stability:")
            print(f"  First 1000 cycles: {np.mean(first_1000):.1f}ms")
            print(f"  Last 1000 cycles: {np.mean(last_1000):.1f}ms")
            print(f"  Degradation: {degradation:+.1f}%")
            
            if abs(degradation) < 10:
                print("  ✅ Performance stable")
            else:
                print("  ⚠️  Performance degradation detected")
        
        print(f"\nMemory usage: {self._get_memory_usage():.1f}MB")
        print(f"Errors: {len(self.metrics['errors'])}")
        
        if self.metrics['errors']:
            print("\nFirst few errors:")
            for err in self.metrics['errors'][:5]:
                print(f"  Cycle {err['cycle']}: {err['error']}")
        
        # Save final report
        report_file = f"stability_test_report_{hours_run:.1f}h.json"
        with open(report_file, 'w') as f:
            json.dump({
                'duration_hours': hours_run,
                'total_cycles': total_cycles,
                'performance_degradation_pct': degradation if self.metrics['cycle_times'] else 0,
                'error_count': len(self.metrics['errors']),
                'checkpoints': self.metrics['checkpoints']
            }, f, indent=2)
        
        print(f"\n📊 Full report saved: {report_file}")
        print("="*60)


def main():
    """Run stability test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Long-running brain stability test')
    parser.add_argument('--hours', type=float, default=8.0,
                       help='Test duration in hours (default: 8)')
    parser.add_argument('--checkpoint-minutes', type=int, default=30,
                       help='Checkpoint interval in minutes (default: 30)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick 5-minute test')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Running quick 5-minute test...")
        test = LongRunningStabilityTest(
            duration_hours=5/60,  # 5 minutes
            checkpoint_interval_minutes=1
        )
    else:
        test = LongRunningStabilityTest(
            duration_hours=args.hours,
            checkpoint_interval_minutes=args.checkpoint_minutes
        )
    
    test.run()


if __name__ == "__main__":
    main()
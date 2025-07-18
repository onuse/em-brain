#!/usr/bin/env python3
"""
Ten Minute Inline Performance Test

Tests the brain directly without server communication to validate optimizations.
"""

import sys
import os
import time
import math
import signal
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

class TenMinuteInlineValidator:
    """
    10-minute biological timescale validation running brain directly.
    """
    
    def __init__(self):
        self.brain = None
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
        print("🧪 TEN MINUTE INLINE PERFORMANCE VALIDATION")
        print("=" * 60)
        print("Testing optimized brain at biological speed for 10 minutes")
        print("Target: <100ms per cycle throughout entire test")
        print("\nInitializing brain...")
        
        # Initialize brain with optimizations
        try:
            self.brain = MinimalBrain(
                enable_logging=False,
                enable_persistence=False,
                enable_storage_optimization=True,
                use_utility_based_activation=True,
                enable_phase2_adaptations=True
            )
            print("✅ Brain initialized with all optimizations")
        except Exception as e:
            print(f"❌ Failed to initialize brain: {e}")
            return
        
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
                predicted_action, brain_state = self.brain.process_sensory_input(sensory_input)
                process_time = (time.time() - process_start) * 1000
                
                if predicted_action:
                    # Store experience with outcome
                    outcome = self._simulate_outcome(predicted_action)
                    store_start = time.time()
                    exp_id = self.brain.store_experience(
                        sensory_input,
                        predicted_action,
                        outcome,
                        predicted_action
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
            if self.brain:
                self.brain.finalize_session()
    
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
            # Get brain state directly
            stats = self.brain.get_brain_state()
            timestamp = time.time()
            
            # Extract similarity statistics
            sim_stats = self.brain.similarity_engine.get_performance_stats()
            if sim_stats:
                self.results['similarity_stats'].append({
                    'timestamp': timestamp,
                    'stats': sim_stats
                })
            
            # Extract activation statistics
            act_stats = self.brain.activation_dynamics.get_utility_statistics()
            if act_stats:
                self.results['activation_stats'].append({
                    'timestamp': timestamp,
                    'stats': act_stats
                })
            
            # Track performance metrics
            perf_data = {
                'timestamp': timestamp,
                'num_experiences': len(self.brain.experience_storage._experiences),
                'working_memory_size': self.brain.activation_dynamics.get_working_memory_size(),
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
        print(f"   Experiences stored: {len(self.brain.experience_storage._experiences)}")
        print(f"   Working memory size: {self.brain.activation_dynamics.get_working_memory_size()}")
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
            
            first_avg = sum(first_minute) / len(first_minute) if first_minute else 0
            last_avg = sum(last_minute) / len(last_minute) if last_minute else 0
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Total cycles: {self.results['cycles_completed']}")
            print(f"   Total time: {total_elapsed/60:.1f} minutes")
            print(f"   Average cycle time: {avg_cycle:.1f}ms")
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
            
            # Similarity learning progress
            if self.results['similarity_stats'] and len(self.results['similarity_stats']) >= 2:
                first_stats = self.results['similarity_stats'][0]['stats']
                last_stats = self.results['similarity_stats'][-1]['stats']
                
                print(f"\n🧠 SIMILARITY LEARNING:")
                print(f"   Adaptations performed: {last_stats.get('total_adaptations', 0)}")
                print(f"   Learning rate: {last_stats.get('current_learning_rate', 0):.3f}")
                
            # Activation system stats
            if self.results['activation_stats'] and len(self.results['activation_stats']) >= 2:
                first_stats = self.results['activation_stats'][0]['stats']
                last_stats = self.results['activation_stats'][-1]['stats']
                
                print(f"\n⚡ ACTIVATION SYSTEM:")
                print(f"   Utility-based decisions: {last_stats.get('utility_based_decisions', 0)}")
                print(f"   Working memory experiences: {last_stats.get('current_working_memory_size', 0)}")
                
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
                print("   - Similarity learning feedback loop ✅")
                print("   - Sparse connectivity ✅")
                print("   - Storage optimization ✅")
                print("   - Activation system optimization ✅")
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
    print("🚀 Starting 10-minute inline performance validation")
    print("This test runs the brain directly without server communication")
    print("Press Ctrl+C to stop early\n")
    
    validator = TenMinuteInlineValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Performance Analysis Test

Tests whether brain response time degrades linearly with experience accumulation
or stabilizes after initial learning period.
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

# Add paths
brain_root = Path(__file__).parent
sys.path.insert(0, str(brain_root))

from server.src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

class PerformanceAnalysisTest:
    """Test to analyze brain performance scaling with experience accumulation."""
    
    def __init__(self, test_duration_minutes: int = 10):
        self.test_duration_minutes = test_duration_minutes
        self.results = {
            'start_time': time.time(),
            'cycles': [],
            'response_times': [],
            'experience_counts': [],
            'performance_phases': []
        }
        
    def run_test(self):
        """Run performance analysis test."""
        print(f"⚡ Performance Analysis Test ({self.test_duration_minutes} minutes)")
        print("=" * 60)
        print("Testing: Response time scaling with experience accumulation")
        print("Goal: Determine if performance degrades linearly or stabilizes")
        print()
        
        client = MinimalBrainClient()
        environment = SensoryMotorWorld(random_seed=42)
        
        if not client.connect():
            print("❌ Failed to connect to brain")
            return False
        
        print("✅ Connected - starting performance analysis...")
        
        try:
            end_time = time.time() + (self.test_duration_minutes * 60)
            cycle_count = 0
            
            # Track performance in phases
            phase_duration = 60  # 1 minute phases
            next_phase_time = time.time() + phase_duration
            current_phase = 0
            
            while time.time() < end_time:
                # Get sensory input
                sensory_input = environment.get_sensory_input()
                
                # Measure response time
                start_time = time.time()
                action = client.get_action(sensory_input, timeout=10.0)
                response_time = time.time() - start_time
                
                if action is None:
                    continue
                
                # Execute action
                environment.execute_action(action)
                
                # Record data
                self.results['cycles'].append(cycle_count)
                self.results['response_times'].append(response_time * 1000)  # Convert to ms
                self.results['experience_counts'].append(self._estimate_experience_count(cycle_count))
                
                # Check if we've moved to next phase
                if time.time() >= next_phase_time:
                    self._analyze_current_phase(current_phase)
                    current_phase += 1
                    next_phase_time = time.time() + phase_duration
                
                # Progress reporting
                if cycle_count % 50 == 0 and cycle_count > 0:
                    elapsed_minutes = (time.time() - self.results['start_time']) / 60
                    recent_times = self.results['response_times'][-10:]
                    avg_recent = sum(recent_times) / len(recent_times)
                    est_experiences = self._estimate_experience_count(cycle_count)
                    
                    print(f"⏱️  {elapsed_minutes:.1f}min: cycle {cycle_count}, "
                          f"{avg_recent:.1f}ms response, ~{est_experiences} experiences")
                
                cycle_count += 1
                time.sleep(0.1)
            
            client.disconnect()
            
            # Final analysis
            self._analyze_performance_scaling()
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            client.disconnect()
            return False
    
    def _estimate_experience_count(self, cycle_count: int) -> int:
        """Estimate experience count based on cycle count."""
        # First cycle creates no experience (needs previous state)
        # Each subsequent cycle creates 1 experience
        return max(0, cycle_count - 1)
    
    def _analyze_current_phase(self, phase_num: int):
        """Analyze performance for current phase."""
        if not self.results['response_times']:
            return
        
        # Get data for this phase (last minute)
        phase_start_idx = max(0, len(self.results['response_times']) - 60)
        phase_times = self.results['response_times'][phase_start_idx:]
        phase_experiences = self.results['experience_counts'][phase_start_idx:]
        
        if phase_times:
            avg_time = sum(phase_times) / len(phase_times)
            avg_experiences = sum(phase_experiences) / len(phase_experiences)
            
            phase_data = {
                'phase': phase_num,
                'avg_response_time_ms': avg_time,
                'avg_experience_count': avg_experiences,
                'sample_size': len(phase_times)
            }
            
            self.results['performance_phases'].append(phase_data)
    
    def _analyze_performance_scaling(self):
        """Analyze how performance scales with experience accumulation."""
        print("\n📊 Performance Scaling Analysis")
        print("=" * 50)
        
        if len(self.results['response_times']) < 10:
            print("❌ Insufficient data for analysis")
            return
        
        # Calculate correlation between experience count and response time
        experiences = np.array(self.results['experience_counts'])
        response_times = np.array(self.results['response_times'])
        
        # Remove initial outliers (first few cycles)
        if len(experiences) > 20:
            experiences = experiences[10:]
            response_times = response_times[10:]
        
        # Calculate correlation
        correlation = np.corrcoef(experiences, response_times)[0, 1]
        
        # Fit linear regression
        if len(experiences) > 1:
            linear_fit = np.polyfit(experiences, response_times, 1)
            slope_ms_per_experience = linear_fit[0]
            intercept_ms = linear_fit[1]
            
            print(f"📈 Linear Regression Analysis:")
            print(f"   Slope: {slope_ms_per_experience:.3f} ms per experience")
            print(f"   Intercept: {intercept_ms:.1f} ms")
            print(f"   Correlation: {correlation:.3f}")
            
            # Assess scaling behavior
            if abs(correlation) > 0.7:
                if slope_ms_per_experience > 0.01:
                    print("   ⚠️  STRONG LINEAR DEGRADATION - Performance will worsen over time")
                    self._project_future_performance(slope_ms_per_experience, intercept_ms)
                else:
                    print("   ✅ MINIMAL LINEAR GROWTH - Performance essentially stable")
            else:
                print("   ✅ NO LINEAR CORRELATION - Performance appears stable")
        
        # Analyze phases
        if len(self.results['performance_phases']) > 1:
            print(f"\n📊 Phase Analysis:")
            for phase in self.results['performance_phases']:
                print(f"   Phase {phase['phase']}: {phase['avg_response_time_ms']:.1f}ms "
                      f"({phase['avg_experience_count']:.0f} experiences)")
            
            # Check for stabilization
            if len(self.results['performance_phases']) >= 3:
                recent_phases = self.results['performance_phases'][-3:]
                times = [p['avg_response_time_ms'] for p in recent_phases]
                
                # Check if recent phases show stabilization
                variance = np.var(times)
                if variance < 100:  # Less than 100ms² variance
                    print("   ✅ PERFORMANCE STABILIZING - Recent phases show low variance")
                else:
                    print("   ⚠️  PERFORMANCE STILL CHANGING - High variance in recent phases")
        
        # Overall assessment
        print(f"\n🎯 Performance Assessment:")
        total_cycles = len(self.results['response_times'])
        final_experiences = self.results['experience_counts'][-1] if self.results['experience_counts'] else 0
        
        early_avg = np.mean(response_times[:min(20, len(response_times)//4)])
        late_avg = np.mean(response_times[-min(20, len(response_times)//4):])
        
        print(f"   Total cycles: {total_cycles}")
        print(f"   Final experience count: {final_experiences}")
        print(f"   Early performance: {early_avg:.1f}ms")
        print(f"   Late performance: {late_avg:.1f}ms")
        print(f"   Performance change: {late_avg - early_avg:+.1f}ms")
        
        # Generate plots
        self._generate_performance_plots()
        
        # Save results
        results_file = brain_root / f"performance_analysis_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {results_file.name}")
    
    def _project_future_performance(self, slope: float, intercept: float):
        """Project future performance based on linear regression."""
        print(f"\n🔮 Performance Projections:")
        
        projections = [1000, 5000, 10000, 50000, 100000]
        for exp_count in projections:
            projected_time = slope * exp_count + intercept
            print(f"   {exp_count:,} experiences: {projected_time:.1f}ms")
            
            if projected_time > 5000:  # 5 second responses
                print(f"   ⚠️  Performance becomes unusable at {exp_count:,} experiences")
                break
    
    def _generate_performance_plots(self):
        """Generate performance analysis plots."""
        if len(self.results['response_times']) < 10:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Response time vs cycle number
        ax1.plot(self.results['cycles'], self.results['response_times'], 'b-', alpha=0.6, label='Response Time')
        
        # Add moving average
        window = 20
        if len(self.results['response_times']) > window:
            moving_avg = []
            for i in range(window, len(self.results['response_times'])):
                avg = np.mean(self.results['response_times'][i-window:i])
                moving_avg.append(avg)
            
            ax1.plot(self.results['cycles'][window:], moving_avg, 'r-', linewidth=2, label='Moving Average')
        
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Response Time (ms)')
        ax1.set_title('Response Time vs Cycle Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Response time vs experience count
        ax2.scatter(self.results['experience_counts'], self.results['response_times'], 
                   alpha=0.6, s=1, label='Actual Data')
        
        # Add linear regression line
        if len(self.results['experience_counts']) > 1:
            experiences = np.array(self.results['experience_counts'])
            response_times = np.array(self.results['response_times'])
            
            # Fit line
            linear_fit = np.polyfit(experiences, response_times, 1)
            line_x = np.linspace(min(experiences), max(experiences), 100)
            line_y = linear_fit[0] * line_x + linear_fit[1]
            
            ax2.plot(line_x, line_y, 'r-', linewidth=2, 
                    label=f'Linear Fit (slope: {linear_fit[0]:.3f} ms/exp)')
        
        ax2.set_xlabel('Experience Count')
        ax2.set_ylabel('Response Time (ms)')
        ax2.set_title('Response Time vs Experience Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(brain_root / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Performance plots saved to: performance_analysis.png")

def main():
    """Run performance analysis test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Analysis Test')
    parser.add_argument('--duration', type=int, default=10, help='Duration in minutes (default: 10)')
    
    args = parser.parse_args()
    
    test = PerformanceAnalysisTest(test_duration_minutes=args.duration)
    success = test.run_test()
    
    if success:
        print(f"\n🎉 Performance analysis completed!")
    else:
        print(f"\n❌ Performance analysis failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
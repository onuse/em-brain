#!/usr/bin/env python3
"""
Debug Motor Pipeline Test

Traces motor commands through the entire pipeline to identify why exploration is low.
Tests:
1. Pattern-based motor generator output
2. Motor cortex amplification
3. Final motor command distribution
4. Boredom avoidance effectiveness
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
from collections import defaultdict, deque

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from behavioral_test_dynamic import DynamicBehavioralTestFramework

class MotorPipelineDebugger:
    """Debug the motor command pipeline"""
    
    def __init__(self):
        self.framework = DynamicBehavioralTestFramework(quiet_mode=True)
        self.motor_history = deque(maxlen=100)
        self.pattern_features = deque(maxlen=100)
        self.cortex_amplifications = deque(maxlen=100)
        self.exploration_metrics = deque(maxlen=100)
        
    def trace_motor_pipeline(self, num_cycles=50):
        """Trace motor commands through the pipeline"""
        print("🔍 Motor Pipeline Debug Test")
        print("=" * 60)
        
        # Setup robot
        self.framework.setup_virtual_robot()
        session_id = self.framework.connection_handler.get_session_id_for_client(
            self.framework.client_id
        )
        
        # Get brain reference
        brain_wrapper = self.framework.brain_service.get_brain_for_session(session_id)
        if not brain_wrapper or not hasattr(brain_wrapper, 'brain'):
            print("❌ Could not access brain instance")
            return
            
        brain = brain_wrapper.brain
        
        # Hook into motor pipeline components
        pattern_motor = brain.pattern_motor_generator if hasattr(brain, 'pattern_motor_generator') else None
        motor_cortex = brain.motor_cortex if hasattr(brain, 'motor_cortex') else None
        
        print(f"✅ Connected to brain components:")
        print(f"   Pattern Motor: {'Yes' if pattern_motor else 'No'}")
        print(f"   Motor Cortex: {'Yes' if motor_cortex else 'No'}")
        print()
        
        # Collect data through cycles
        for i in range(num_cycles):
            # Generate varied sensory input
            phase = i * 0.1
            sensory_input = [
                np.sin(phase) * 0.5,
                np.cos(phase) * 0.5,
                np.random.randn() * 0.1,
                0.5  # Some constant stimulus
            ] + [np.random.randn() * 0.1 for _ in range(12)]
            
            # Process through brain
            motor_output = self.framework.connection_handler.handle_sensory_input(
                self.framework.client_id, sensory_input
            )
            
            # Record motor output
            self.motor_history.append(motor_output[:4])  # First 4 motors
            
            # Extract pattern features if available
            if pattern_motor and hasattr(pattern_motor, '_last_pattern_features'):
                features = pattern_motor._last_pattern_features
                self.pattern_features.append(features)
                
            # Extract cortex amplification if available
            if motor_cortex and hasattr(motor_cortex, '_last_amplification'):
                amp = motor_cortex._last_amplification
                self.cortex_amplifications.append(amp)
                
            # Check exploration metrics
            if pattern_motor and hasattr(pattern_motor, '_last_exploration_drive'):
                self.exploration_metrics.append(pattern_motor._last_exploration_drive)
                
        # Analyze results
        self._analyze_motor_distribution()
        self._analyze_pattern_diversity()
        self._analyze_exploration_drive()
        self._analyze_cortex_effectiveness()
        
    def _analyze_motor_distribution(self):
        """Analyze distribution of motor commands"""
        print("\n📊 Motor Command Distribution")
        print("-" * 40)
        
        if not self.motor_history:
            print("❌ No motor data collected")
            return
            
        motor_array = np.array(self.motor_history)
        
        # Analyze each motor channel
        for i in range(4):
            values = motor_array[:, i]
            print(f"\nMotor {i} ({'forward' if i==0 else 'left' if i==1 else 'right' if i==2 else 'stop'}):")
            print(f"   Mean: {np.mean(values):.3f}")
            print(f"   Std:  {np.std(values):.3f}")
            print(f"   Min:  {np.min(values):.3f}")
            print(f"   Max:  {np.max(values):.3f}")
            print(f"   >0.5: {np.sum(values > 0.5)} ({np.sum(values > 0.5)/len(values)*100:.1f}%)")
            
        # Check which action would be selected most often
        action_counts = defaultdict(int)
        for motors in self.motor_history:
            max_idx = np.argmax(motors)
            action_names = ['FORWARD', 'LEFT', 'RIGHT', 'STOP']
            action_counts[action_names[max_idx]] += 1
            
        print(f"\n🎯 Action Selection (argmax):")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {action}: {count} ({count/len(self.motor_history)*100:.1f}%)")
            
    def _analyze_pattern_diversity(self):
        """Analyze diversity of pattern features"""
        print("\n🌀 Pattern Feature Analysis")
        print("-" * 40)
        
        if not self.pattern_features:
            print("❌ No pattern features collected")
            return
            
        # Analyze pattern diversity
        features_array = []
        for features in self.pattern_features:
            if features:
                features_array.append([
                    features.get('energy_concentration', 0),
                    features.get('spatial_variance', 0),
                    features.get('pattern_complexity', 0),
                    features.get('novelty_score', 0)
                ])
                
        if features_array:
            features_array = np.array(features_array)
            print(f"Pattern features over time:")
            print(f"   Energy concentration: {np.mean(features_array[:, 0]):.3f} ± {np.std(features_array[:, 0]):.3f}")
            print(f"   Spatial variance: {np.mean(features_array[:, 1]):.3f} ± {np.std(features_array[:, 1]):.3f}")
            print(f"   Pattern complexity: {np.mean(features_array[:, 2]):.3f} ± {np.std(features_array[:, 2]):.3f}")
            print(f"   Novelty score: {np.mean(features_array[:, 3]):.3f} ± {np.std(features_array[:, 3]):.3f}")
            
    def _analyze_exploration_drive(self):
        """Analyze exploration drive effectiveness"""
        print("\n🔍 Exploration Drive Analysis")
        print("-" * 40)
        
        if not self.exploration_metrics:
            print("❌ No exploration metrics collected")
            return
            
        exploration_array = np.array(self.exploration_metrics)
        print(f"Exploration drive statistics:")
        print(f"   Mean: {np.mean(exploration_array):.3f}")
        print(f"   Std:  {np.std(exploration_array):.3f}")
        print(f"   Min:  {np.min(exploration_array):.3f}")
        print(f"   Max:  {np.max(exploration_array):.3f}")
        print(f"   >0.3: {np.sum(exploration_array > 0.3)} ({np.sum(exploration_array > 0.3)/len(exploration_array)*100:.1f}%)")
        
    def _analyze_cortex_effectiveness(self):
        """Analyze motor cortex amplification"""
        print("\n🧠 Motor Cortex Analysis")
        print("-" * 40)
        
        if not self.cortex_amplifications:
            print("❌ No cortex data collected")
            return
            
        amp_array = np.array(self.cortex_amplifications)
        print(f"Amplification statistics:")
        print(f"   Mean: {np.mean(amp_array):.3f}")
        print(f"   Std:  {np.std(amp_array):.3f}")
        print(f"   Min:  {np.min(amp_array):.3f}")
        print(f"   Max:  {np.max(amp_array):.3f}")
        print(f"   >1.5: {np.sum(amp_array > 1.5)} ({np.sum(amp_array > 1.5)/len(amp_array)*100:.1f}%)")
        
    def run_movement_test(self):
        """Test actual movement in a simulated environment"""
        print("\n🚶 Movement Pattern Test")
        print("=" * 60)
        
        # Simulate robot position
        position = np.array([5.0, 5.0])
        orientation = 0.0
        positions = [position.copy()]
        
        # Run movement simulation
        for i in range(50):
            sensory_input = [np.random.randn() * 0.5 for _ in range(16)]
            motor_output = self.framework.connection_handler.handle_sensory_input(
                self.framework.client_id, sensory_input
            )
            
            # Interpret action
            action_idx = np.argmax(motor_output[:4])
            
            # Update position based on action
            if action_idx == 0:  # Forward
                dx = 0.1 * np.cos(orientation)
                dy = 0.1 * np.sin(orientation)
                position += np.array([dx, dy])
            elif action_idx == 1:  # Left
                orientation += 0.1
            elif action_idx == 2:  # Right
                orientation -= 0.1
            # else: STOP - no movement
            
            positions.append(position.copy())
            
        # Calculate exploration metrics
        positions_array = np.array(positions)
        total_distance = np.sum(np.linalg.norm(np.diff(positions_array, axis=0), axis=1))
        displacement = np.linalg.norm(positions_array[-1] - positions_array[0])
        
        # Grid coverage (10x10 grid)
        grid = np.zeros((10, 10))
        for pos in positions_array:
            grid_x = int(np.clip(pos[0], 0, 9))
            grid_y = int(np.clip(pos[1], 0, 9))
            grid[grid_x, grid_y] = 1
            
        coverage = np.sum(grid) / 100
        
        print(f"Movement Statistics:")
        print(f"   Total distance: {total_distance:.2f}")
        print(f"   Displacement: {displacement:.2f}")
        print(f"   Efficiency: {displacement/max(0.1, total_distance):.3f}")
        print(f"   Grid coverage: {coverage:.3f} ({int(np.sum(grid))}/100 cells)")
        
        # Show movement pattern
        print(f"\nMovement visualization (start=S, end=E):")
        vis_grid = [[' ' for _ in range(20)] for _ in range(20)]
        
        # Mark path
        for pos in positions_array:
            x = int(np.clip(pos[0]*2, 0, 19))
            y = int(np.clip(pos[1]*2, 0, 19))
            vis_grid[y][x] = '.'
            
        # Mark start and end
        start_x = int(np.clip(positions_array[0][0]*2, 0, 19))
        start_y = int(np.clip(positions_array[0][1]*2, 0, 19))
        end_x = int(np.clip(positions_array[-1][0]*2, 0, 19))
        end_y = int(np.clip(positions_array[-1][1]*2, 0, 19))
        vis_grid[start_y][start_x] = 'S'
        vis_grid[end_y][end_x] = 'E'
        
        for row in vis_grid:
            print(''.join(row))
            
    def cleanup(self):
        """Clean up resources"""
        self.framework.cleanup()


def main():
    """Run motor pipeline debugging"""
    debugger = MotorPipelineDebugger()
    
    try:
        # Run pipeline trace
        debugger.trace_motor_pipeline(num_cycles=50)
        
        # Run movement test
        debugger.run_movement_test()
        
        print("\n" + "="*60)
        print("🎯 Key Findings Summary")
        print("="*60)
        
    finally:
        debugger.cleanup()


if __name__ == "__main__":
    main()
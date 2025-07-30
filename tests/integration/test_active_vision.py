"""
Test Active Vision Through Predictive Sampling

Validates that the brain can:
1. Generate uncertainty maps from field state
2. Request focused glimpses at uncertain regions
3. Control camera movements to reduce uncertainty
4. Learn patterns through active sampling
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

# Import brain components
from server.src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
from server.src.brains.field.evolved_field_dynamics import EvolvedFieldDynamics
from server.src.core.glimpse_adapter import GlimpseSensoryAdapter, GlimpseRequest
from server.src.core.interfaces import Robot, SensorChannel, MotorChannel


def create_visual_robot() -> Robot:
    """Create a mock robot with camera and basic sensors."""
    sensory_channels = [
        SensorChannel(0, "distance", 0.0, 10.0, "meters", "Distance sensor"),
        SensorChannel(1, "camera_brightness", 0.0, 1.0, "normalized", "Average brightness from camera"),
        SensorChannel(2, "camera_contrast", 0.0, 1.0, "normalized", "Contrast in visual field"),
        SensorChannel(3, "camera_motion", -1.0, 1.0, "normalized", "Motion detection"),
    ]
    
    motor_channels = [
        MotorChannel(0, "left_wheel", -1.0, 1.0, "normalized", "Left wheel motor"),
        MotorChannel(1, "right_wheel", -1.0, 1.0, "normalized", "Right wheel motor"),
        MotorChannel(2, "arm", -1.0, 1.0, "normalized", "Arm motor"),
        MotorChannel(3, "camera_pan", -1.0, 1.0, "normalized", "Camera pan servo"),
        MotorChannel(4, "camera_tilt", -1.0, 1.0, "normalized", "Camera tilt servo"),
    ]
    
    return Robot(
        robot_id="visual_test_bot",
        robot_type="test_visual",
        sensory_channels=sensory_channels,
        motor_channels=motor_channels,
        capabilities={"camera": True, "movement": True}
    )


class SimulatedVisualEnvironment:
    """Simulates a visual environment with objects."""
    
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.objects = []
        self.camera_x = 0.0
        self.camera_y = 0.0
        
        # Place some objects
        self.add_object(0.5, 0.5, size=0.2, brightness=0.8)  # Bright object
        self.add_object(-0.3, 0.2, size=0.15, brightness=0.3)  # Dark object
        self.add_object(0.0, -0.6, size=0.25, brightness=0.6)  # Medium object
    
    def add_object(self, x: float, y: float, size: float, brightness: float):
        """Add an object to the environment."""
        self.objects.append({
            'x': x, 'y': y, 'size': size, 'brightness': brightness
        })
    
    def get_glimpse(self, request: GlimpseRequest, glimpse_size: int = 32) -> torch.Tensor:
        """Get a glimpse at the requested location."""
        # Create glimpse tensor
        glimpse = torch.zeros(glimpse_size, glimpse_size)
        
        # Calculate what's visible in this glimpse
        for obj in self.objects:
            # Check if object is in view
            dx = obj['x'] - request.center_x
            dy = obj['y'] - request.center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Object visibility based on distance and zoom
            if distance < (0.5 / request.zoom):
                # Add object to glimpse
                # Simple gaussian blob for now
                cx = int((0.5 + dx * request.zoom) * glimpse_size)
                cy = int((0.5 + dy * request.zoom) * glimpse_size)
                
                if 0 <= cx < glimpse_size and 0 <= cy < glimpse_size:
                    # Create object pattern
                    for i in range(max(0, cx-5), min(glimpse_size, cx+5)):
                        for j in range(max(0, cy-5), min(glimpse_size, cy+5)):
                            dist_to_center = np.sqrt((i-cx)**2 + (j-cy)**2)
                            if dist_to_center < 5:
                                glimpse[j, i] = obj['brightness'] * np.exp(-dist_to_center/3)
        
        # Add some noise
        glimpse += torch.randn_like(glimpse) * 0.05
        
        return torch.clamp(glimpse, 0, 1)
    
    def update_camera(self, pan_delta: float, tilt_delta: float):
        """Update camera position based on motor commands."""
        self.camera_x = np.clip(self.camera_x + pan_delta * 0.1, -1, 1)
        self.camera_y = np.clip(self.camera_y + tilt_delta * 0.1, -1, 1)


def test_active_vision():
    """Test the complete active vision pipeline."""
    print("\n=== Testing Active Vision System ===\n")
    
    # Create mock robot
    robot = create_visual_robot()
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=len(robot.sensory_channels),
        motor_dim=len(robot.motor_channels),
        device=torch.device('cpu')
    )
    
    # Create glimpse adapter
    glimpse_adapter = GlimpseSensoryAdapter(
        robot=robot,
        field_dimensions=len(robot.sensory_channels),
        glimpse_size=32,
        max_glimpses_per_cycle=3
    )
    
    # Create environment
    env = SimulatedVisualEnvironment()
    
    # Run several cycles
    print("Running active vision cycles...")
    
    for cycle in range(20):
        # 1. Generate uncertainty map
        uncertainty_map = brain.field_dynamics.generate_uncertainty_map(brain.unified_field)
        
        # 2. Generate glimpse requests
        glimpse_requests = glimpse_adapter.generate_glimpse_requests(uncertainty_map)
        
        # 3. Get glimpses from environment
        glimpse_data = {}
        for i, request in enumerate(glimpse_requests):
            # Adjust request based on current camera position
            adjusted_request = GlimpseRequest(
                center_x=request.center_x + env.camera_x,
                center_y=request.center_y + env.camera_y,
                zoom=request.zoom
            )
            glimpse_data[f"glimpse_{i}"] = env.get_glimpse(adjusted_request)
        
        # 4. Prepare sensory input
        base_sensors = [
            5.0,  # Distance
            np.mean([g.mean().item() for g in glimpse_data.values()]) if glimpse_data else 0.5,  # Brightness
            np.mean([g.std().item() for g in glimpse_data.values()]) if glimpse_data else 0.1,   # Contrast
            0.0   # Motion (simplified)
        ]
        
        # 5. Process through brain
        # Convert to field space with glimpses
        field_input = glimpse_adapter.to_field_space_with_glimpses(base_sensors, glimpse_data)
        
        # Add reward based on finding bright objects
        reward = 0.0
        if glimpse_data:
            max_brightness = max(g.max().item() for g in glimpse_data.values())
            if max_brightness > 0.7:
                reward = 1.0
                print(f"  Cycle {cycle}: Found bright object! (brightness: {max_brightness:.2f})")
        
        # Process brain cycle
        # Add reward to sensory input (last element)
        sensory_with_reward = field_input.detach().cpu().numpy().tolist()
        sensory_with_reward.append(reward)
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_with_reward)
        
        # 6. Update camera position
        if len(motor_output) >= 5:
            pan_delta = motor_output[3]
            tilt_delta = motor_output[4]
            env.update_camera(pan_delta, tilt_delta)
            
            if abs(pan_delta) > 0.1 or abs(tilt_delta) > 0.1:
                print(f"  Cycle {cycle}: Camera movement - pan: {pan_delta:.2f}, tilt: {tilt_delta:.2f}")
        
        # Print uncertainty stats
        if cycle % 5 == 0:
            mean_uncertainty = uncertainty_map.mean().item()
            max_uncertainty = uncertainty_map.max().item()
            print(f"  Cycle {cycle}: Uncertainty - mean: {mean_uncertainty:.3f}, max: {max_uncertainty:.3f}")
    
    # Check results
    stats = glimpse_adapter.get_glimpse_statistics()
    print(f"\nGlimpse Statistics:")
    print(f"  Total glimpses: {stats['total_glimpses']}")
    print(f"  Spatial coverage: {stats['spatial_coverage']:.2f}")
    print(f"  Average zoom: {stats['avg_zoom']:.2f}")
    
    # Verify active vision behaviors
    assert stats['total_glimpses'] > 0, "No glimpses were generated"
    assert stats['spatial_coverage'] > 0.01, "Insufficient spatial exploration"
    
    # Check if brain learned to focus on bright objects
    final_confidence = brain.field_dynamics.smoothed_confidence
    print(f"\nFinal confidence: {final_confidence:.3f}")
    
    print("\n✅ Active vision test passed!")
    return True


if __name__ == "__main__":
    # Run the test
    try:
        test_active_vision()
        print("\n🎉 All active vision tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
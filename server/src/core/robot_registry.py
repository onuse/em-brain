"""
Robot Registry implementation.

Manages robot profiles and parses capabilities from handshake messages.
"""

import json
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from server.src.core.interfaces import IRobotRegistry, Robot, SensorChannel, MotorChannel


class RobotRegistry(IRobotRegistry):
    """
    Manages robot profiles and types.
    
    This is the ONLY place that understands capability encoding from
    the communication protocol.
    """
    
    def __init__(self, profiles_dir: Optional[str] = None):
        self.robots: Dict[str, Robot] = {}
        self.profiles_dir = Path(profiles_dir) if profiles_dir else None
        
        # Known robot types and their default configurations
        self.known_types = {
            1.0: "picarx",      # PiCar-X
            2.0: "generic",     # Generic robot
            # Add more as needed
        }
    
    def register_robot(self, capabilities: List[float]) -> Robot:
        """Parse capabilities vector and create/retrieve robot profile."""
        
        # Parse handshake capabilities
        # [version, sensory_dim, motor_dim, hardware_type, capabilities_mask]
        version = capabilities[0] if len(capabilities) > 0 else 1.0
        sensory_dim = int(capabilities[1]) if len(capabilities) > 1 else 16
        motor_dim = int(capabilities[2]) if len(capabilities) > 2 else 4
        hardware_type = capabilities[3] if len(capabilities) > 3 else 0.0
        capabilities_mask = int(capabilities[4]) if len(capabilities) > 4 else 0
        
        # Determine robot type
        robot_type = self.known_types.get(hardware_type, "unknown")
        
        # Try to load robot profile if available
        robot = self._load_profile(robot_type, sensory_dim, motor_dim)
        
        if robot is None:
            # Create generic robot profile
            robot = self._create_generic_robot(
                robot_type, sensory_dim, motor_dim, capabilities_mask
            )
        
        # Register robot
        self.robots[robot.robot_id] = robot
        
        return robot
    
    def get_robot(self, robot_id: str) -> Optional[Robot]:
        """Retrieve robot by ID."""
        return self.robots.get(robot_id)
    
    def _load_profile(self, robot_type: str, sensory_dim: int, motor_dim: int) -> Optional[Robot]:
        """Load robot profile from JSON file if available."""
        
        if not self.profiles_dir:
            return None
        
        # Look for profile file
        profile_path = self.profiles_dir / f"{robot_type}_profile.json"
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            # Validate dimensions match
            if (profile_data.get('sensory_mapping', {}).get('dimensions') != sensory_dim or
                profile_data.get('action_mapping', {}).get('dimensions') != motor_dim):
                print(f"⚠️  Profile dimensions mismatch for {robot_type}")
                return None
            
            # Create robot from profile
            robot_id = f"{robot_type}_{uuid.uuid4().hex[:8]}"
            
            # Parse sensor channels
            sensory_channels = []
            for channel_data in profile_data.get('sensory_mapping', {}).get('channels', []):
                sensory_channels.append(SensorChannel(
                    index=channel_data['index'],
                    name=channel_data['name'],
                    range_min=channel_data['range'][0],
                    range_max=channel_data['range'][1],
                    unit=channel_data['unit'],
                    description=channel_data['description']
                ))
            
            # Parse motor channels
            motor_channels = []
            for channel_data in profile_data.get('action_mapping', {}).get('channels', []):
                motor_channels.append(MotorChannel(
                    index=channel_data['index'],
                    name=channel_data['name'],
                    range_min=channel_data['range'][0],
                    range_max=channel_data['range'][1],
                    unit=channel_data['unit'],
                    description=channel_data['description']
                ))
            
            # Create robot
            return Robot(
                robot_id=robot_id,
                robot_type=robot_type,
                sensory_channels=sensory_channels,
                motor_channels=motor_channels,
                capabilities=profile_data.get('capabilities', {})
            )
            
        except Exception as e:
            print(f"❌ Failed to load profile for {robot_type}: {e}")
            return None
    
    def _create_generic_robot(self, robot_type: str, sensory_dim: int, 
                             motor_dim: int, capabilities_mask: int) -> Robot:
        """Create a generic robot profile when no specific profile is available."""
        
        robot_id = f"{robot_type}_{uuid.uuid4().hex[:8]}"
        
        # Create generic sensor channels
        sensory_channels = []
        for i in range(sensory_dim):
            sensory_channels.append(SensorChannel(
                index=i,
                name=f"sensor_{i}",
                range_min=0.0,
                range_max=1.0,
                unit="normalized",
                description=f"Generic sensor channel {i}"
            ))
        
        # Create generic motor channels
        motor_channels = []
        for i in range(motor_dim):
            motor_channels.append(MotorChannel(
                index=i,
                name=f"motor_{i}",
                range_min=-1.0,
                range_max=1.0,
                unit="normalized",
                description=f"Generic motor channel {i}"
            ))
        
        # Decode capabilities
        capabilities = {
            'visual_processing': bool(capabilities_mask & 1),
            'audio_processing': bool(capabilities_mask & 2),
            'manipulation': bool(capabilities_mask & 4),
            'multi_agent': bool(capabilities_mask & 8),
        }
        
        return Robot(
            robot_id=robot_id,
            robot_type=robot_type,
            sensory_channels=sensory_channels,
            motor_channels=motor_channels,
            capabilities=capabilities
        )
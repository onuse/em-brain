# PiCar-X Simulation Architecture

## Overview

This simulation environment replicates a SunFounder PiCar-X robot with Raspberry Pi Zero WH in a virtual 3D world. The purpose is to test and iterate on the brainstem intelligence before deploying to real hardware.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Brain Server  │◄──►│ Brainstem Client │◄──►│ 3D Simulation   │
│                 │    │                  │    │                 │
│ - Experience    │    │ - Sensor Fusion  │    │ - Physics World │
│ - Similarity    │    │ - Motor Control  │    │ - Vehicle Model │
│ - Activation    │    │ - TCP/IP Comms   │    │ - Sensors       │
│ - Prediction    │    │ - Real-time Loop │    │ - Visualization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. Brain Server
- **Location**: `brain_server.py` (root)
- **Purpose**: Hosts the intelligent brain system
- **Interface**: TCP/IP socket server
- **Processing**: 
  - Sensory input → experience storage
  - Experience similarity → working memory
  - Prediction → motor commands

### 2. Brainstem Client
- **Location**: `demos/picar_x_simulation/brainstem.py`
- **Purpose**: Robot's autonomous nervous system
- **Responsibilities**:
  - Connect to brain server via TCP/IP
  - Collect sensor data from simulation
  - Send fused sensory input to brain
  - Receive motor commands from brain
  - Execute actions in simulation
- **Real-time Requirements**: 20-60 Hz operation

### 3. 3D Simulation Environment
- **Location**: `demos/picar_x_simulation/world/`
- **Purpose**: Virtual testing environment
- **Components**:
  - **World**: Room with walls, floor, obstacles
  - **Physics**: Collision detection, movement dynamics
  - **Vehicle**: PiCar-X model with accurate dimensions
  - **Sensors**: Camera, ultrasonic, IMU simulation
  - **Rendering**: 3D visualization (optional)

## Hardware Specifications (Target: SunFounder PiCar-X)

### Vehicle Dimensions
- **Length**: 165mm
- **Width**: 95mm  
- **Height**: 85mm
- **Wheelbase**: 90mm
- **Track Width**: 65mm
- **Weight**: ~200g

### Sensors
- **Camera**: 
  - 640x480 resolution
  - 60° field of view
  - Pan servo: ±90°
  - Tilt servo: ±30°
- **Ultrasonic Sensor**: 
  - Range: 2cm - 300cm
  - Accuracy: ±3mm
  - Beam angle: 30°
- **IMU (if added)**:
  - 3-axis accelerometer
  - 3-axis gyroscope
  - Heading estimation

### Actuators
- **Drive Motors**: 
  - Differential drive (left/right wheels)
  - Speed range: -100% to +100%
- **Steering Servo**:
  - Range: ±30° from center
  - Resolution: 0.5°
- **Camera Servos**:
  - Pan: ±90° horizontal
  - Tilt: ±30° vertical

## World Environment Design

### Room Layout
- **Shape**: Rectangular room (3m x 4m)
- **Walls**: 0.3m high (car height = 0.085m)
- **Floor**: Flat, textured surface
- **Ceiling**: Open (for lighting/camera view)

### Obstacles
- **Static Obstacles**:
  - Boxes: Various sizes (0.1m - 0.5m cubes)
  - Cylinders: Poles, trash cans
  - Ramps: Slight elevation changes
- **Dynamic Obstacles** (future):
  - Moving objects
  - Other robots

### Lighting
- **Ambient**: Uniform room lighting
- **Shadows**: Basic shadow casting for depth perception
- **Camera Simulation**: Realistic lighting affects on camera feed

## Sensor Simulation Details

### Camera Simulation
```python
class CameraSimulator:
    def __init__(self, resolution=(640, 480), fov=60):
        self.resolution = resolution
        self.field_of_view = fov
        self.pan_angle = 0  # ±90°
        self.tilt_angle = 0  # ±30°
    
    def capture_frame(self, world_state, vehicle_pose):
        # Render 3D scene from camera perspective
        # Apply lighting, shadows, textures
        # Return RGB image array
        pass
```

### Ultrasonic Simulation
```python
class UltrasonicSimulator:
    def __init__(self, max_range=3.0, beam_angle=30):
        self.max_range = max_range
        self.beam_angle = beam_angle
    
    def measure_distance(self, world_state, vehicle_pose):
        # Cast ray from sensor position
        # Find closest obstacle intersection
        # Add realistic noise (±3mm)
        # Return distance in meters
        pass
```

### IMU Simulation
```python
class IMUSimulator:
    def __init__(self):
        self.acceleration = [0, 0, 9.81]  # m/s² (gravity)
        self.angular_velocity = [0, 0, 0]  # rad/s
        self.heading = 0  # degrees
    
    def update(self, vehicle_dynamics, dt):
        # Calculate acceleration from vehicle motion
        # Add gyroscope data from turning
        # Integrate heading from angular velocity
        # Add realistic sensor noise
        pass
```

## Physics Simulation

### Vehicle Dynamics
- **Drive System**: Differential drive kinematics
- **Steering**: Ackermann steering geometry (simplified)
- **Friction**: Tire-ground interaction model
- **Mass**: 0.2kg vehicle mass
- **Inertia**: Rotational inertia for turning dynamics

### Collision Detection
- **Vehicle Hull**: Rectangular bounding box
- **Obstacle Detection**: AABB collision detection
- **Response**: Elastic collision with damping
- **Sensor Mounting**: Accurate sensor positions on vehicle

## Communication Protocol

### Brainstem → Brain (Sensory Input)
```json
{
  "timestamp": 1640995200.123,
  "camera": {
    "image_data": "base64_encoded_640x480_rgb",
    "pan_angle": 15.0,
    "tilt_angle": -5.0
  },
  "ultrasonic": {
    "distance": 0.847  // meters
  },
  "imu": {
    "acceleration": [0.1, -0.05, 9.81],
    "angular_velocity": [0, 0, 0.15],
    "heading": 142.5
  },
  "vehicle_state": {
    "position": [1.23, 0.85],
    "orientation": 142.5,
    "velocity": [0.15, 0.12]
  }
}
```

### Brain → Brainstem (Motor Commands)
```json
{
  "timestamp": 1640995200.125,
  "drive_motors": {
    "left_speed": 0.3,   // -1.0 to 1.0
    "right_speed": 0.35
  },
  "steering": {
    "angle": -5.0        // degrees ±30°
  },
  "camera": {
    "pan_target": 20.0,  // degrees ±90°
    "tilt_target": -10.0 // degrees ±30°
  }
}
```

## File Structure

```
demos/picar_x_simulation/
├── SIMULATION_ARCHITECTURE.md     # This document
├── __init__.py
├── brainstem.py                   # Main brainstem client
├── vehicle/
│   ├── __init__.py
│   ├── picar_x_model.py          # Vehicle physics model
│   ├── sensors.py                # Camera, ultrasonic, IMU
│   └── actuators.py              # Motors, servos
├── world/
│   ├── __init__.py
│   ├── environment.py            # 3D world definition
│   ├── physics.py                # Collision detection, dynamics
│   └── obstacles.py              # Static/dynamic obstacles
├── visualization/
│   ├── __init__.py
│   ├── renderer.py               # 3D rendering (pygame/OpenGL)
│   └── ui.py                     # Debug UI, controls
└── utils/
    ├── __init__.py
    ├── math_utils.py             # Vector math, transformations
    └── network.py                # TCP/IP communication helpers
```

## Testing Strategy

### Unit Tests
- Vehicle dynamics accuracy
- Sensor simulation precision
- Communication protocol validation
- Physics collision detection

### Integration Tests
- Brainstem → Brain → Brainstem loop
- Real-time performance (20+ Hz)
- Sensor fusion accuracy
- Motor command execution

### Scenario Tests
- **Navigation**: Simple point-to-point movement
- **Obstacle Avoidance**: Static obstacle navigation
- **Exploration**: Unknown environment mapping
- **Learning**: Repeated environment adaptation

## Development Phases

### Phase 1: Core Infrastructure ✅
- [x] Architecture documentation
- [ ] Basic brainstem client
- [ ] Simple 3D world
- [ ] Vehicle model
- [ ] Communication protocol

### Phase 2: Sensor Integration
- [ ] Camera simulation
- [ ] Ultrasonic sensor
- [ ] IMU integration
- [ ] Sensor fusion

### Phase 3: Physics & Dynamics
- [ ] Vehicle physics
- [ ] Collision detection
- [ ] Realistic movement
- [ ] Environmental interactions

### Phase 4: Intelligence Integration
- [ ] Brain server connection
- [ ] Learning behaviors
- [ ] Adaptive navigation
- [ ] Performance optimization

### Phase 5: Visualization & Debug
- [ ] 3D rendering
- [ ] Debug UI
- [ ] Performance monitoring
- [ ] Scenario recording

## Real Hardware Transition

### Validation Metrics
- **Sensor Accuracy**: Simulation vs real sensor comparison
- **Movement Precision**: Simulated vs actual vehicle motion
- **Timing**: Real-time performance matching
- **Learning Transfer**: Brain knowledge portability

### Hardware-Specific Adaptations
- **Raspberry Pi Zero Performance**: CPU/memory constraints
- **Real Sensor Noise**: Calibration and filtering
- **Physical Limitations**: Motor backlash, mechanical tolerances
- **Environmental Factors**: Lighting, surface variations

This simulation foundation will enable rapid iteration and robust testing before deploying to the physical PiCar-X robot.
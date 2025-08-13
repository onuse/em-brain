# PiCar-X Protocol Mapping

This document describes how the SunFounder PiCar-X robot implements the generic brain communication protocol.

## Overview

The PiCar-X brainstem adapts between:
- **Hardware**: 16 physical sensor channels from PiCar-X
- **Brain Interface**: 24 sensory inputs + 1 reward signal
- **Motor Control**: 4 brain outputs → 5 hardware actuators

## Handshake Parameters

During connection, PiCar-X sends:
```python
[1.0, 24.0, 4.0, 1.0, 3.0]
 │    │     │    │    └─ Capabilities: 1(visual) + 2(audio) = 3
 │    │     │    └─ Hardware type: 1.0 (PiCar-X)
 │    │     └─ Expected action vector size: 4
 │    └─ Sensory vector size: 24
 └─ Robot software version: 1.0
```

## Sensor Mapping (16 Hardware → 24 Brain Inputs)

### Hardware Sensors (16 channels)
| Index | Sensor | Range | Hardware
|-------|--------|-------|----------
| 0 | Ultrasonic distance | 0-4m | HC-SR04
| 1 | Grayscale right | 0-1 | ADC pin A0
| 2 | Grayscale center | 0-1 | ADC pin A1
| 3 | Grayscale left | 0-1 | ADC pin A2
| 4 | Left motor speed | -1 to 1 | Encoder feedback
| 5 | Right motor speed | -1 to 1 | Encoder feedback
| 6 | Camera pan angle | -90 to 90° | Servo position
| 7 | Camera tilt angle | -35 to 65° | Servo position
| 8 | Steering angle | -30 to 30° | Servo position
| 9 | Battery voltage | 0-8.4V | ADC measurement
| 10 | Line detected | 0/1 | Computed from grayscale
| 11 | Cliff detected | 0/1 | Threshold detection
| 12 | CPU temperature | 0-100°C | System sensor
| 13 | Memory usage | 0-1 | System monitor
| 14 | Timestamp | ms | System clock
| 15 | Reserved | 0 | Future use

### Brain Input Mapping (24 channels + reward)
| Index | Description | Mapping from Hardware
|-------|-------------|----------------------
| 0 | Distance proximity | 1 - (sensor[0] / 2.0)
| 1 | Lateral position | 0.5 (no sensor)
| 2 | Vertical position | 0.5 (no sensor)
| 3 | Grayscale right | sensor[1]
| 4 | Grayscale center | sensor[2]
| 5 | Grayscale left | sensor[3]
| 6 | Left motor state | (sensor[4] + 1) / 2
| 7 | Right motor state | (sensor[5] + 1) / 2
| 8 | Steering state | (sensor[8] + 30) / 60
| 9 | Camera pan state | (sensor[6] + 90) / 180
| 10 | Camera tilt state | (sensor[7] + 35) / 100
| 11 | Battery level | sensor[9] / 8.4
| 12 | Line detected | sensor[10]
| 13 | Cliff detected | sensor[11]
| 14 | CPU temperature | sensor[12] / 100
| 15 | Memory usage | sensor[13]
| 16 | Angular velocity | Computed from motor differential
| 17 | Forward velocity | Computed from motor average
| 18 | Distance gradient | Change in distance over time
| 19 | Line quality | Center vs side sensors
| 20 | Steering effort | abs(steering) / 30
| 21 | Speed magnitude | abs(forward velocity)
| 22 | Exploration score | Sensor variance metric
| 23 | System health | Combined battery/temp/safety
| 24 | **Reward signal** | Computed behavioral reward

## Action Mapping (4 Brain Outputs → 5 Hardware Motors)

### Brain Outputs
| Index | Control Signal | Range
|-------|---------------|-------
| 0 | Forward/backward | -1 to 1
| 1 | Left/right steering | -1 to 1
| 2 | Camera pan | -1 to 1
| 3 | Camera tilt | -1 to 1

### Hardware Motor Mapping
| Motor | Computation | Range | Hardware
|-------|------------|-------|----------
| Left wheel | brain[0] + differential(brain[1]) | -50 to 50% | DC motor
| Right wheel | brain[0] - differential(brain[1]) | -50 to 50% | DC motor
| Steering servo | brain[1] × 25 | -25 to 25° | Servo
| Camera pan | brain[2] × 45 | -90 to 90° | Servo
| Camera tilt | brain[3] × 30 | -35 to 65° | Servo

## Reward Signal Computation

The reward signal (channel 24) encourages:

**Positive rewards for:**
- Safe forward movement (distance > 0.2m)
- Successful line following (center sensor > 0.6)
- Exploration (steering changes)
- Good system health

**Negative rewards for:**
- Near collisions (distance < 0.1m)
- Cliff detection
- Low battery (< 6.5V)
- High temperature (> 60°C)
- Getting stuck

## Safety Features

### Hardware Reflexes
Immediate responses that bypass the brain:
- **Collision avoidance**: Stop if distance < 5cm
- **Cliff detection**: Reverse if edge detected
- **Thermal protection**: Throttle if CPU > 70°C

### Fallback Behavior
When brain is disconnected:
- Simple obstacle avoidance
- Basic line following
- Safe speed limits

## Implementation Files

- `sensor_motor_adapter.py`: Handles all mapping logic
- `integrated_brainstem.py`: Manages brain communication
- `brain_client.py`: Implements TCP binary protocol
- `picarx_robot.py`: Main robot controller

## Testing

Run the integration test:
```bash
python3 test_full_integration.py
```

This verifies:
- Sensor expansion (16 → 24)
- Motor mapping (4 → 5)
- Protocol compliance
- Safety reflexes

## Performance

- **Control rate**: 20Hz (50ms cycles)
- **Sensor latency**: <5ms
- **Network latency**: <10ms typical
- **Total loop time**: ~45ms average

## Customization

To modify mappings, edit `sensor_motor_adapter.py`:
- `sensors_to_brain_input()`: Change sensor mappings
- `brain_output_to_motors()`: Adjust motor control
- `_calculate_reward()`: Modify reward function

The adapter pattern allows easy customization while maintaining protocol compliance.
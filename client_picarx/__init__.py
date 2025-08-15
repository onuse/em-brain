"""
PiCar-X Brain Client

The PiCar-X client implements the brainstem and hardware abstraction layer
for the robot, connecting to the brain server for intelligence while handling
local sensorimotor control.

This is the "peripheral nervous system" of the unified brain system,
handling real-time robot control while the server provides higher-order
intelligence and learning.

Components:
- brainstem/: Core robot control and brain server integration
- hardware/: Hardware Abstraction Layer with drivers and interfaces
- vocal/: Digital vocal cord system (complex actuator)
- deployment/: Update and management systems
- utils/: Client-specific utilities

Architecture:
- Brain Server: Intelligence, learning, prediction (runs on development machine)
- Brain Client: Sensorimotor control, safety, hardware (runs on robot)
- Communication: REST API for commands and sensor data exchange
"""

__version__ = "0.1.0"
__author__ = "Unified Brain System Project"

# Core client components will be imported here as they're implemented
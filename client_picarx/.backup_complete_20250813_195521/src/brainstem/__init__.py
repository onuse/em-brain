"""
PiCar-X Brainstem Implementation

The brainstem handles real-time robot control, sensor processing, and
communication with the brain server. It acts as the "spinal cord" of
the unified brain system.

Components:
- brain_client: Communication with brain server
- control_loop: Main robot control cycle
- safety_monitor: Hardware safety monitoring
- sensor_processor: Sensor data collection and preprocessing
"""
#!/usr/bin/env python3
"""
Biological-Inspired Nucleus Architecture

Modular brainstem components that operate independently and communicate
via the event bus. Each nucleus handles a specific aspect of robot control,
similar to biological nervous system organization.
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import logging

from .event_bus import AsyncEventBus, Event, EventType, ThreadSafeState, CommandQueue
from ..config.brainstem_config import BrainstemConfig, get_config

# Configure logging
logger = logging.getLogger(__name__)


class Nucleus(ABC):
    """
    Base class for all brainstem nuclei.
    
    Each nucleus:
    - Operates independently in its own thread/async context
    - Communicates via event bus
    - Has its own internal state
    - Can operate even if other nuclei fail
    """
    
    def __init__(self, name: str, event_bus: AsyncEventBus, config: BrainstemConfig):
        """Initialize nucleus."""
        self.name = name
        self.event_bus = event_bus
        self.config = config
        
        # Thread-safe internal state
        self.state = ThreadSafeState()
        self.state.set('active', False)
        self.state.set('cycle_count', 0)
        self.state.set('errors', 0)
        
        # Performance monitoring
        self.last_cycle_time = 0.0
        
        logger.info(f"{name} nucleus initialized")
    
    @abstractmethod
    async def start(self):
        """Start the nucleus operations."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the nucleus operations."""
        pass
    
    @abstractmethod
    async def process(self):
        """Main processing loop for the nucleus."""
        pass
    
    def is_active(self) -> bool:
        """Check if nucleus is active."""
        return self.state.get('active', False)
    
    def get_status(self) -> Dict[str, Any]:
        """Get nucleus status."""
        return {
            'name': self.name,
            'active': self.is_active(),
            'cycles': self.state.get('cycle_count', 0),
            'errors': self.state.get('errors', 0),
            'last_cycle_ms': self.last_cycle_time * 1000
        }


class SensoryNucleus(Nucleus):
    """
    Handles all sensor input processing.
    
    Responsibilities:
    - Poll hardware sensors
    - Normalize sensor values
    - Detect sensor failures
    - Publish sensor events
    """
    
    def __init__(self, event_bus: AsyncEventBus, config: BrainstemConfig):
        super().__init__("Sensory", event_bus, config)
        
        # Sensor-specific state
        self.state.set('last_sensor_data', None)
        self.state.set('sensor_failures', 0)
        
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.STATUS_REQUEST, self._on_status_request)
    
    async def start(self):
        """Start sensor processing."""
        self.state.set('active', True)
        logger.info(f"{self.name} nucleus started")
        
        # Start processing loop
        asyncio.create_task(self.process())
    
    async def stop(self):
        """Stop sensor processing."""
        self.state.set('active', False)
        logger.info(f"{self.name} nucleus stopped")
    
    async def process(self):
        """Main sensor processing loop."""
        poll_rate = self.config.threading.sensor_poll_rate
        
        while self.is_active():
            try:
                start_time = time.time()
                
                # Read sensors (placeholder - integrate with actual hardware)
                sensor_data = await self._read_sensors()
                
                # Validate and normalize
                if self._validate_sensors(sensor_data):
                    normalized = self._normalize_sensors(sensor_data)
                    
                    # Update state
                    self.state.set('last_sensor_data', normalized)
                    self.state.increment('cycle_count')
                    
                    # Publish sensor event
                    event = Event(
                        EventType.SENSOR_DATA,
                        data=normalized,
                        source=self.name,
                        priority=5
                    )
                    self.event_bus.publish(event)
                    
                    # Check for critical conditions
                    self._check_safety_conditions(normalized)
                else:
                    self.state.increment('sensor_failures')
                    self.event_bus.publish(Event(
                        EventType.SENSOR_ERROR,
                        data="Sensor validation failed",
                        source=self.name,
                        priority=8
                    ))
                
                # Track performance
                self.last_cycle_time = time.time() - start_time
                
                # Sleep to maintain poll rate
                sleep_time = poll_rate - self.last_cycle_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"{self.name} processing error: {e}")
                self.state.increment('errors')
                await asyncio.sleep(poll_rate)
    
    async def _read_sensors(self) -> List[float]:
        """Read raw sensor values from hardware."""
        # TODO: Integrate with actual PiCar-X hardware
        # For now, return mock data
        return [
            0.5,   # Distance
            0.3, 0.3, 0.3,  # Grayscale sensors
            0.0, 0.0,  # Motor feedback
            0.0, 0.0,  # Camera position
            0.0,   # Steering angle
            7.4,   # Battery voltage
            0.0,   # Line detected
            0.0,   # Cliff detected
            45.0,  # CPU temperature
            0.3,   # Memory usage
            0.0, 0.0  # Reserved
        ]
    
    def _validate_sensors(self, sensor_data: List[float]) -> bool:
        """Validate sensor data for sanity."""
        if len(sensor_data) != self.config.sensors.picarx_sensor_count:
            return False
        
        # Check for NaN or inf
        if any(np.isnan(v) or np.isinf(v) for v in sensor_data):
            return False
        
        # Check reasonable ranges
        distance = sensor_data[0]
        if distance < 0 or distance > 10:  # 10 meters max
            return False
        
        battery = sensor_data[9]
        if battery < 0 or battery > 12:  # 12V max
            return False
        
        return True
    
    def _normalize_sensors(self, sensor_data: List[float]) -> Dict[str, Any]:
        """Normalize sensor values to standard ranges."""
        return {
            'distance_m': sensor_data[0],
            'grayscale': sensor_data[1:4],
            'motor_feedback': sensor_data[4:6],
            'camera_position': sensor_data[6:8],
            'steering_angle': sensor_data[8],
            'battery_voltage': sensor_data[9],
            'line_detected': bool(sensor_data[10]),
            'cliff_detected': bool(sensor_data[11]),
            'cpu_temperature': sensor_data[12],
            'memory_usage': sensor_data[13],
            'timestamp': time.time()
        }
    
    def _check_safety_conditions(self, sensor_data: Dict[str, Any]):
        """Check for safety-critical conditions and publish events."""
        # Collision detection
        if sensor_data['distance_m'] < self.config.safety.emergency_stop_distance:
            self.event_bus.publish_urgent(Event(
                EventType.COLLISION_DETECTED,
                data={'distance': sensor_data['distance_m']},
                source=self.name,
                priority=10
            ))
        
        # Cliff detection
        if sensor_data['cliff_detected']:
            self.event_bus.publish_urgent(Event(
                EventType.CLIFF_DETECTED,
                data=True,
                source=self.name,
                priority=10
            ))
        
        # Battery low
        if sensor_data['battery_voltage'] < self.config.sensors.battery_min:
            self.event_bus.publish(Event(
                EventType.BATTERY_LOW,
                data={'voltage': sensor_data['battery_voltage']},
                source=self.name,
                priority=7
            ))
        
        # Temperature high
        if sensor_data['cpu_temperature'] > self.config.sensors.cpu_temp_critical:
            self.event_bus.publish(Event(
                EventType.TEMPERATURE_HIGH,
                data={'temperature': sensor_data['cpu_temperature']},
                source=self.name,
                priority=7
            ))
    
    def _on_status_request(self, event: Event):
        """Handle status request."""
        self.event_bus.publish(Event(
            EventType.STATUS_RESPONSE,
            data=self.get_status(),
            source=self.name
        ))


class MotorNucleus(Nucleus):
    """
    Handles all motor control and actuation.
    
    Responsibilities:
    - Execute motor commands
    - Apply safety limits
    - Smooth motor transitions
    - Monitor motor health
    """
    
    def __init__(self, event_bus: AsyncEventBus, config: BrainstemConfig):
        super().__init__("Motor", event_bus, config)
        
        # Motor-specific state
        self.command_queue = CommandQueue(max_age=self.config.safety.command_timeout)
        self.state.set('last_command', None)
        self.state.set('emergency_stop', False)
        
        # Smoothing buffers
        self.motor_smoothing = {
            'left': 0.0,
            'right': 0.0,
            'steering': 0.0,
            'pan': 0.0,
            'tilt': 0.0
        }
        
        # Subscribe to motor events
        self.event_bus.subscribe(EventType.MOTOR_COMMAND, self._on_motor_command)
        self.event_bus.subscribe(EventType.MOTOR_EMERGENCY_STOP, self._on_emergency_stop)
        self.event_bus.subscribe(EventType.COLLISION_DETECTED, self._on_collision)
        self.event_bus.subscribe(EventType.CLIFF_DETECTED, self._on_cliff)
    
    async def start(self):
        """Start motor control."""
        self.state.set('active', True)
        self.state.set('emergency_stop', False)
        logger.info(f"{self.name} nucleus started")
        
        # Start processing loop
        asyncio.create_task(self.process())
    
    async def stop(self):
        """Stop motor control."""
        # Stop all motors first
        await self._stop_all_motors()
        
        self.state.set('active', False)
        logger.info(f"{self.name} nucleus stopped")
    
    async def process(self):
        """Main motor control loop."""
        update_rate = self.config.threading.motor_update_rate
        
        while self.is_active():
            try:
                start_time = time.time()
                
                # Check emergency stop
                if self.state.get('emergency_stop'):
                    await self._stop_all_motors()
                    self.state.set('emergency_stop', False)
                else:
                    # Get latest motor command
                    command = self.command_queue.get_latest()
                    
                    if command:
                        # Apply safety limits
                        safe_command = self._apply_safety_limits(command)
                        
                        # Apply smoothing
                        smooth_command = self._apply_smoothing(safe_command)
                        
                        # Execute command
                        await self._execute_motor_command(smooth_command)
                        
                        # Update state
                        self.state.set('last_command', smooth_command)
                        self.state.increment('cycle_count')
                        
                        # Publish feedback
                        self.event_bus.publish(Event(
                            EventType.MOTOR_FEEDBACK,
                            data=smooth_command,
                            source=self.name
                        ))
                
                # Track performance
                self.last_cycle_time = time.time() - start_time
                
                # Sleep to maintain update rate
                sleep_time = update_rate - self.last_cycle_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"{self.name} processing error: {e}")
                self.state.increment('errors')
                await asyncio.sleep(update_rate)
    
    def _on_motor_command(self, event: Event):
        """Handle motor command event."""
        self.command_queue.put(event.data)
    
    def _on_emergency_stop(self, event: Event):
        """Handle emergency stop event."""
        logger.warning(f"EMERGENCY STOP triggered by {event.source}")
        self.state.set('emergency_stop', True)
        self.command_queue.clear()
    
    def _on_collision(self, event: Event):
        """Handle collision detection."""
        logger.warning(f"Collision detected at {event.data['distance']}m")
        self.state.set('emergency_stop', True)
    
    def _on_cliff(self, event: Event):
        """Handle cliff detection."""
        logger.warning("Cliff detected!")
        self.state.set('emergency_stop', True)
    
    def _apply_safety_limits(self, command: Dict[str, float]) -> Dict[str, float]:
        """Apply safety limits to motor commands."""
        safe = command.copy()
        
        # Limit motor speeds
        max_speed = self.config.motors.max_motor_speed
        for key in ['left_motor', 'right_motor']:
            if key in safe:
                safe[key] = np.clip(safe[key], -max_speed, max_speed)
        
        # Limit steering angle
        max_angle = self.config.motors.max_steering_angle
        if 'steering_servo' in safe:
            safe['steering_servo'] = np.clip(safe['steering_servo'], -max_angle, max_angle)
        
        # Limit camera servos
        if 'camera_pan_servo' in safe:
            safe['camera_pan_servo'] = np.clip(
                safe['camera_pan_servo'],
                self.config.motors.camera_pan_min,
                self.config.motors.camera_pan_max
            )
        
        if 'camera_tilt_servo' in safe:
            safe['camera_tilt_servo'] = np.clip(
                safe['camera_tilt_servo'],
                self.config.motors.camera_tilt_min,
                self.config.motors.camera_tilt_max
            )
        
        return safe
    
    def _apply_smoothing(self, command: Dict[str, float]) -> Dict[str, float]:
        """Apply exponential smoothing to motor commands."""
        smooth = {}
        alpha = self.config.motors.motor_smoothing_alpha
        
        # Map command keys to smoothing keys
        mapping = {
            'left_motor': 'left',
            'right_motor': 'right',
            'steering_servo': 'steering',
            'camera_pan_servo': 'pan',
            'camera_tilt_servo': 'tilt'
        }
        
        for cmd_key, smooth_key in mapping.items():
            if cmd_key in command:
                # Exponential smoothing
                self.motor_smoothing[smooth_key] = (
                    alpha * command[cmd_key] +
                    (1 - alpha) * self.motor_smoothing[smooth_key]
                )
                smooth[cmd_key] = self.motor_smoothing[smooth_key]
        
        return smooth
    
    async def _execute_motor_command(self, command: Dict[str, float]):
        """Execute motor command on hardware."""
        # TODO: Integrate with actual PiCar-X hardware
        # For now, just log
        logger.debug(f"Executing motor command: {command}")
    
    async def _stop_all_motors(self):
        """Emergency stop all motors."""
        stop_command = {
            'left_motor': 0.0,
            'right_motor': 0.0,
            'steering_servo': 0.0,
            'camera_pan_servo': 0.0,
            'camera_tilt_servo': 0.0
        }
        await self._execute_motor_command(stop_command)
        
        # Reset smoothing
        for key in self.motor_smoothing:
            self.motor_smoothing[key] = 0.0


class SafetyNucleus(Nucleus):
    """
    Handles safety reflexes and emergency responses.
    
    Responsibilities:
    - Monitor safety conditions
    - Trigger reflexes (work without brain!)
    - Emergency stop coordination
    - Watchdog timers
    """
    
    def __init__(self, event_bus: AsyncEventBus, config: BrainstemConfig):
        super().__init__("Safety", event_bus, config)
        
        # Safety state
        self.state.set('safety_violations', 0)
        self.state.set('last_sensor_time', time.time())
        self.state.set('last_command_time', time.time())
        
        # Subscribe to safety-relevant events
        self.event_bus.subscribe(EventType.SENSOR_DATA, self._on_sensor_data)
        self.event_bus.subscribe(EventType.MOTOR_COMMAND, self._on_motor_command)
        self.event_bus.subscribe(EventType.COLLISION_DETECTED, self._on_safety_violation)
        self.event_bus.subscribe(EventType.CLIFF_DETECTED, self._on_safety_violation)
    
    async def start(self):
        """Start safety monitoring."""
        self.state.set('active', True)
        logger.info(f"{self.name} nucleus started - Safety systems online")
        
        # Start watchdog loop
        asyncio.create_task(self.process())
    
    async def stop(self):
        """Stop safety monitoring."""
        self.state.set('active', False)
        logger.info(f"{self.name} nucleus stopped")
    
    async def process(self):
        """Main safety monitoring loop."""
        while self.is_active():
            try:
                current_time = time.time()
                
                # Check sensor timeout
                last_sensor = self.state.get('last_sensor_time')
                if current_time - last_sensor > self.config.safety.sensor_timeout:
                    logger.warning("Sensor timeout detected!")
                    self.event_bus.publish_urgent(Event(
                        EventType.SENSOR_TIMEOUT,
                        source=self.name,
                        priority=9
                    ))
                
                # Check command timeout
                last_command = self.state.get('last_command_time')
                if current_time - last_command > self.config.safety.command_timeout:
                    # No recent commands - safe to stop
                    self.event_bus.publish(Event(
                        EventType.MOTOR_EMERGENCY_STOP,
                        data="Command timeout",
                        source=self.name,
                        priority=8
                    ))
                
                # Update cycle count
                self.state.increment('cycle_count')
                
                # Sleep for watchdog interval
                await asyncio.sleep(0.1)  # 10Hz watchdog
                
            except Exception as e:
                logger.error(f"{self.name} watchdog error: {e}")
                self.state.increment('errors')
                await asyncio.sleep(0.1)
    
    def _on_sensor_data(self, event: Event):
        """Update last sensor time."""
        self.state.set('last_sensor_time', time.time())
        
        # Check for immediate safety issues
        data = event.data
        if isinstance(data, dict):
            # Distance-based reflex
            if data.get('distance_m', float('inf')) < self.config.safety.min_safe_distance:
                # Trigger speed reduction
                self.event_bus.publish(Event(
                    EventType.MOTOR_COMMAND,
                    data={
                        'left_motor': 0.0,
                        'right_motor': 0.0
                    },
                    source=self.name,
                    priority=9
                ))
    
    def _on_motor_command(self, event: Event):
        """Update last command time."""
        self.state.set('last_command_time', time.time())
    
    def _on_safety_violation(self, event: Event):
        """Handle safety violations."""
        self.state.increment('safety_violations')
        logger.warning(f"Safety violation: {event.type.name} from {event.source}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_nuclei():
        """Test the nucleus system."""
        print("Testing Nucleus Architecture")
        print("=" * 50)
        
        # Create event bus
        bus = AsyncEventBus()
        bus.start()
        
        # Get configuration
        config = get_config(profile="testing")
        
        # Create nuclei
        sensory = SensoryNucleus(bus, config)
        motor = MotorNucleus(bus, config)
        safety = SafetyNucleus(bus, config)
        
        # Start nuclei
        await sensory.start()
        await motor.start()
        await safety.start()
        
        print("\nNuclei started. Running for 5 seconds...")
        
        # Simulate some commands
        await asyncio.sleep(1)
        
        # Send motor command
        bus.publish(Event(
            EventType.MOTOR_COMMAND,
            data={
                'left_motor': 30.0,
                'right_motor': 30.0,
                'steering_servo': 0.0
            },
            source="test"
        ))
        
        await asyncio.sleep(2)
        
        # Simulate collision
        bus.publish_urgent(Event(
            EventType.COLLISION_DETECTED,
            data={'distance': 0.05},
            source="test"
        ))
        
        await asyncio.sleep(2)
        
        # Get status
        print("\nNucleus Status:")
        for nucleus in [sensory, motor, safety]:
            status = nucleus.get_status()
            print(f"  {status['name']}: {status['cycles']} cycles, {status['errors']} errors")
        
        print(f"\nEvent Bus Stats: {bus.get_stats()}")
        
        # Stop everything
        await sensory.stop()
        await motor.stop()
        await safety.stop()
        bus.stop()
        
        print("\nTest complete!")
    
    # Run test
    asyncio.run(test_nuclei())
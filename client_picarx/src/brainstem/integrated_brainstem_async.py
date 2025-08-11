#!/usr/bin/env python3
"""
Async Event-Driven Integrated Brainstem for PiCar-X

Complete implementation of the nucleus-based architecture with:
- Parallel processing via event bus
- Thread-safe operation
- Graceful degradation
- No reward signals (brain discovers through experience)
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from .event_bus import AsyncEventBus, Event, EventType
from .nuclei import (
    SensoryNucleus, MotorNucleus, SafetyNucleus, 
    CommunicationNucleus, BehavioralNucleus
)
from .brain_client import BrainServerClient, BrainServerConfig, MockBrainServerClient
from .sensor_motor_adapter_fixed import PiCarXBrainAdapter
from ..config.brainstem_config import BrainstemConfig, get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AsyncBrainstemConfig:
    """Configuration for async brainstem."""
    brain_server_config: BrainServerConfig
    use_mock_brain: bool = False
    brainstem_config: BrainstemConfig = None
    enable_telemetry: bool = True
    
    def __post_init__(self):
        if self.brainstem_config is None:
            self.brainstem_config = get_config()


class BrainstemConductor:
    """
    Orchestrates all nuclei via event-driven architecture.
    
    This is the main conductor that:
    - Initializes and manages all nuclei
    - Wires up event routing
    - Handles lifecycle management
    - Provides unified status interface
    """
    
    def __init__(self, config: AsyncBrainstemConfig):
        """Initialize the brainstem conductor."""
        self.config = config
        self.bsc = config.brainstem_config
        
        # Create event bus
        self.event_bus = AsyncEventBus(max_queue_size=1000)
        
        # Create brain client
        if config.use_mock_brain:
            self.brain_client = MockBrainServerClient(config.brain_server_config)
        else:
            self.brain_client = BrainServerClient(config.brain_server_config)
        
        # Create adapter
        self.adapter = PiCarXBrainAdapter(self.bsc)
        
        # Initialize nuclei
        self._init_nuclei()
        
        # Wire up event handlers
        self._wire_event_handlers()
        
        # State
        self.running = False
        self.start_time = None
        
        logger.info("🧠 Async Brainstem Conductor initialized")
        logger.info(f"   Event-driven: Yes")
        logger.info(f"   Mock brain: {config.use_mock_brain}")
        logger.info(f"   Safe mode: {self.bsc.safe_mode}")
        logger.info(f"   Update rate: {self.bsc.threading.motor_update_rate}Hz")
    
    def _init_nuclei(self):
        """Initialize all brainstem nuclei."""
        # Create nuclei with dependencies
        self.sensory_nucleus = SensoryNucleus(
            self.event_bus, self.bsc
        )
        
        self.motor_nucleus = MotorNucleus(
            self.event_bus, self.bsc
        )
        
        self.safety_nucleus = SafetyNucleus(
            self.event_bus, self.bsc
        )
        
        self.comm_nucleus = CommunicationNucleus(
            self.event_bus, self.bsc,
            self.brain_client, self.adapter
        )
        
        self.behavioral_nucleus = BehavioralNucleus(
            self.event_bus, self.bsc
        )
        
        # Store all nuclei for lifecycle management
        self.nuclei = [
            self.sensory_nucleus,
            self.motor_nucleus,
            self.safety_nucleus,
            self.comm_nucleus,
            self.behavioral_nucleus
        ]
        
        logger.info(f"✅ Initialized {len(self.nuclei)} nuclei")
    
    def _wire_event_handlers(self):
        """Wire up event routing between nuclei."""
        
        # Sensory → Communication (send to brain)
        self.event_bus.subscribe_async(
            EventType.SENSOR_DATA,
            self.comm_nucleus.handle_sensor_data
        )
        
        # Brain → Motor (execute commands)
        self.event_bus.subscribe_async(
            EventType.BRAIN_RESPONSE,
            self.motor_nucleus.handle_brain_commands
        )
        
        # Safety → Motor (emergency override)
        self.event_bus.subscribe_async(
            EventType.COLLISION_DETECTED,
            self.safety_nucleus.handle_collision
        )
        
        self.event_bus.subscribe_async(
            EventType.CLIFF_DETECTED,
            self.safety_nucleus.handle_cliff
        )
        
        # Safety → Motor (emergency stop has highest priority)
        self.event_bus.subscribe_async(
            EventType.MOTOR_EMERGENCY_STOP,
            self.motor_nucleus.emergency_stop
        )
        
        # System events
        self.event_bus.subscribe_async(
            EventType.BATTERY_LOW,
            self.behavioral_nucleus.handle_low_battery
        )
        
        self.event_bus.subscribe_async(
            EventType.TEMPERATURE_HIGH,
            self.behavioral_nucleus.handle_high_temperature
        )
        
        # Telemetry
        if self.config.enable_telemetry:
            self.event_bus.subscribe_async(
                EventType.TELEMETRY_UPDATE,
                self._handle_telemetry
            )
        
        logger.info("✅ Event handlers wired")
    
    async def connect(self) -> bool:
        """Connect to brain server."""
        success = self.brain_client.connect()
        
        if success:
            # Notify all nuclei of connection
            await self.event_bus.emit(Event(
                type=EventType.BRAIN_CONNECTED,
                data={'timestamp': time.time()},
                source='conductor'
            ))
            logger.info("✅ Connected to brain server")
        else:
            # Notify all nuclei to use fallback behaviors
            await self.event_bus.emit(Event(
                type=EventType.BRAIN_DISCONNECTED,
                data={'reason': 'connection_failed'},
                source='conductor'
            ))
            logger.warning("⚠️  Running in autonomous mode (no brain)")
        
        return success
    
    async def start(self):
        """Start all nuclei and begin processing."""
        logger.info("Starting brainstem conductor...")
        
        self.running = True
        self.start_time = time.time()
        
        # Start event bus
        await self.event_bus.start()
        
        # Start all nuclei concurrently
        await asyncio.gather(*[
            nucleus.start() for nucleus in self.nuclei
        ])
        
        logger.info("✅ All nuclei started")
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self):
        """Stop all nuclei and shutdown."""
        logger.info("Stopping brainstem conductor...")
        
        self.running = False
        
        # Send shutdown event
        await self.event_bus.emit(Event(
            type=EventType.SHUTDOWN,
            source='conductor',
            priority=100  # High priority
        ))
        
        # Stop all nuclei
        await asyncio.gather(*[
            nucleus.stop() for nucleus in self.nuclei
        ])
        
        # Stop event bus
        await self.event_bus.stop()
        
        logger.info("🛑 Brainstem conductor stopped")
    
    async def process_sensor_cycle(self, raw_sensors: List[float]):
        """
        Process one sensor cycle through the event system.
        
        This is the main entry point for sensor data.
        """
        # Emit sensor data event - nuclei will handle the rest
        await self.event_bus.emit(Event(
            type=EventType.SENSOR_DATA,
            data={'raw_sensors': raw_sensors},
            source='hardware',
            priority=10  # High priority for sensor data
        ))
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat events."""
        while self.running:
            await self.event_bus.emit(Event(
                type=EventType.HEARTBEAT,
                data={
                    'uptime': time.time() - self.start_time,
                    'cycles': sum(n.state.get('cycle_count', 0) for n in self.nuclei)
                },
                source='conductor'
            ))
            await asyncio.sleep(1.0)  # 1Hz heartbeat
    
    async def _handle_telemetry(self, event: Event):
        """Handle telemetry events for monitoring."""
        data = event.data
        logger.debug(f"📊 Telemetry: {data}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive brainstem status."""
        return {
            'running': self.running,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'connected': self.brain_client.is_connected(),
            'nuclei': {
                nucleus.name: nucleus.get_status() 
                for nucleus in self.nuclei
            },
            'event_bus': self.event_bus.get_stats(),
            'adapter': self.adapter.get_debug_info()
        }
    
    async def get_motor_commands(self) -> Optional[Dict[str, float]]:
        """Get latest motor commands from motor nucleus."""
        return self.motor_nucleus.get_latest_commands()


async def test_async_brainstem():
    """Test the async event-driven brainstem."""
    print("🧪 Testing Async Event-Driven Brainstem")
    print("=" * 50)
    
    # Configure for testing
    config = AsyncBrainstemConfig(
        brain_server_config=BrainServerConfig(),
        use_mock_brain=True,
        brainstem_config=get_config(profile="testing")
    )
    
    # Create conductor
    conductor = BrainstemConductor(config)
    
    # Connect and start
    await conductor.connect()
    await conductor.start()
    
    # Simulate some sensor cycles
    print("\nSimulating sensor cycles...")
    
    test_scenarios = [
        ("Normal", [0.8, 0.3, 0.3, 0.3, 0.2, 0.2, 0, 0, 0, 7.4, 0, 0, 45, 0.3, 1000, 0]),
        ("Obstacle", [0.15, 0.3, 0.3, 0.3, 0.1, 0.1, 0, 0, 0, 7.4, 0, 0, 45, 0.3, 1000, 0]),
        ("Cliff", [0.5, 0.3, 0.3, 0.3, 0.2, 0.2, 0, 0, 0, 7.4, 0, 1, 45, 0.3, 1000, 0]),
    ]
    
    for scenario, sensors in test_scenarios:
        print(f"\n{scenario} scenario:")
        await conductor.process_sensor_cycle(sensors)
        
        # Give system time to process
        await asyncio.sleep(0.1)
        
        # Get motor commands
        motors = await conductor.get_motor_commands()
        if motors:
            print(f"  Motors: L={motors.get('left_motor', 0):.1f}, "
                  f"R={motors.get('right_motor', 0):.1f}, "
                  f"Steer={motors.get('steering_servo', 0):.1f}°")
    
    # Show status
    print("\n📊 System Status:")
    status = conductor.get_status()
    print(f"  Uptime: {status['uptime']:.1f}s")
    print(f"  Connected: {status['connected']}")
    print(f"  Nuclei status:")
    for name, nucleus_status in status['nuclei'].items():
        print(f"    {name}: active={nucleus_status['active']}, "
              f"cycles={nucleus_status['cycles']}")
    
    # Stop
    await conductor.stop()
    print("\n✅ Async brainstem test complete!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_async_brainstem())
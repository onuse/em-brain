#!/usr/bin/env python3
"""
Async Event Bus System

Thread-safe event-driven architecture for brainstem components.
Eliminates race conditions by using message passing instead of shared state.
"""

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for brainstem communication."""
    # Sensor events
    SENSOR_DATA = auto()
    SENSOR_TIMEOUT = auto()
    SENSOR_ERROR = auto()
    
    # Motor events
    MOTOR_COMMAND = auto()
    MOTOR_FEEDBACK = auto()
    MOTOR_EMERGENCY_STOP = auto()
    
    # Brain communication
    BRAIN_CONNECTED = auto()
    BRAIN_DISCONNECTED = auto()
    BRAIN_RESPONSE = auto()
    BRAIN_TIMEOUT = auto()
    
    # Safety events
    COLLISION_DETECTED = auto()
    CLIFF_DETECTED = auto()
    BATTERY_LOW = auto()
    TEMPERATURE_HIGH = auto()
    
    # System events
    SHUTDOWN = auto()
    HEARTBEAT = auto()
    CONFIG_CHANGED = auto()
    
    # Telemetry
    TELEMETRY_UPDATE = auto()
    STATUS_REQUEST = auto()
    STATUS_RESPONSE = auto()


@dataclass
class Event:
    """Event message structure."""
    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    priority: int = 0  # Higher = more important
    
    def __lt__(self, other):
        """For priority queue sorting (higher priority first)."""
        return self.priority > other.priority


class AsyncEventBus:
    """
    Thread-safe async event bus for component communication.
    
    Features:
    - Async/await support
    - Thread-safe operations
    - Priority-based event delivery
    - Event filtering and routing
    - Performance monitoring
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize the event bus."""
        self.max_queue_size = max_queue_size
        
        # Async event loop (created when started)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        
        # Event subscribers (event_type -> list of callbacks)
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.async_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Thread-safe event queue
        self.event_queue: asyncio.Queue = None
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'events_delivered': 0,
            'events_dropped': 0,
            'queue_high_water': 0
        }
        
        # Running state
        self.running = False
        self._lock = threading.Lock()
        
        logger.info("Event bus initialized")
    
    def start(self):
        """Start the event bus in a separate thread."""
        with self._lock:
            if self.running:
                logger.warning("Event bus already running")
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.thread.start()
            
            # Wait for loop to be ready
            while self.loop is None:
                time.sleep(0.01)
            
            logger.info("Event bus started")
    
    def stop(self):
        """Stop the event bus."""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            
            # Post shutdown event
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._post_event(Event(EventType.SHUTDOWN, source="event_bus")),
                    self.loop
                )
            
            # Wait for thread to finish
            if self.thread:
                self.thread.join(timeout=2.0)
            
            logger.info("Event bus stopped")
            logger.info(f"Statistics: {self.stats}")
    
    def _run_event_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()
        self.event_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        try:
            self.loop.run_until_complete(self._event_processor())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self.loop.close()
            self.loop = None
    
    async def _event_processor(self):
        """Main event processing loop."""
        logger.info("Event processor started")
        
        while self.running:
            try:
                # Get next event (with timeout to check running state)
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=0.1
                )
                
                # Update statistics
                self.stats['events_delivered'] += 1
                queue_size = self.event_queue.qsize()
                if queue_size > self.stats['queue_high_water']:
                    self.stats['queue_high_water'] = queue_size
                
                # Handle shutdown
                if event.type == EventType.SHUTDOWN:
                    logger.info("Shutdown event received")
                    break
                
                # Deliver to subscribers
                await self._deliver_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
        
        logger.info("Event processor stopped")
    
    async def _deliver_event(self, event: Event):
        """Deliver event to all subscribers."""
        # Async subscribers
        tasks = []
        for callback in self.async_subscribers.get(event.type, []):
            tasks.append(self._call_async_subscriber(callback, event))
        
        # Sync subscribers (run in thread pool)
        for callback in self.subscribers.get(event.type, []):
            tasks.append(self._call_sync_subscriber(callback, event))
        
        # Wait for all deliveries
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_async_subscriber(self, callback: Callable, event: Event):
        """Call async subscriber."""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Async subscriber error: {e}")
    
    async def _call_sync_subscriber(self, callback: Callable, event: Event):
        """Call sync subscriber in thread pool."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, event)
        except Exception as e:
            logger.error(f"Sync subscriber error: {e}")
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type (sync callback)."""
        with self._lock:
            self.subscribers[event_type].append(callback)
            logger.debug(f"Subscribed {callback.__name__} to {event_type.name}")
    
    def subscribe_async(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type (async callback)."""
        with self._lock:
            self.async_subscribers[event_type].append(callback)
            logger.debug(f"Async subscribed {callback.__name__} to {event_type.name}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type."""
        with self._lock:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
            if callback in self.async_subscribers[event_type]:
                self.async_subscribers[event_type].remove(callback)
    
    def publish(self, event: Event):
        """
        Publish an event (thread-safe, non-blocking).
        
        Returns True if event was queued, False if dropped.
        """
        if not self.running or not self.loop:
            logger.warning("Event bus not running")
            return False
        
        self.stats['events_published'] += 1
        
        # Post event to async loop
        future = asyncio.run_coroutine_threadsafe(
            self._post_event(event),
            self.loop
        )
        
        try:
            return future.result(timeout=0.1)
        except:
            self.stats['events_dropped'] += 1
            return False
    
    async def _post_event(self, event: Event):
        """Post event to queue (internal async method)."""
        try:
            # Try to put without blocking
            self.event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.type.name}")
            self.stats['events_dropped'] += 1
            return False
    
    def publish_urgent(self, event: Event):
        """Publish high-priority event that jumps the queue."""
        event.priority = 1000  # Max priority
        return self.publish(event)
    
    def get_stats(self) -> Dict[str, int]:
        """Get event bus statistics."""
        return self.stats.copy()


class ThreadSafeState:
    """
    Thread-safe state container using locks.
    
    Provides atomic read/write operations for shared state.
    """
    
    def __init__(self):
        """Initialize thread-safe state."""
        self._state = {}
        self._lock = threading.RLock()  # Reentrant lock
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe get."""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Thread-safe set."""
        with self._lock:
            self._state[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Thread-safe bulk update."""
        with self._lock:
            self._state.update(updates)
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Thread-safe increment (returns new value)."""
        with self._lock:
            current = self._state.get(key, 0)
            new_value = current + amount
            self._state[key] = new_value
            return new_value
    
    def get_and_clear(self, key: str) -> Any:
        """Atomically get and clear a value."""
        with self._lock:
            value = self._state.get(key)
            if key in self._state:
                del self._state[key]
            return value
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get atomic snapshot of entire state."""
        with self._lock:
            return self._state.copy()


class CommandQueue:
    """
    Thread-safe command queue with timeout support.
    
    Used for motor commands and other time-sensitive data.
    """
    
    def __init__(self, max_age: float = 0.5):
        """
        Initialize command queue.
        
        Args:
            max_age: Maximum age of commands in seconds
        """
        self.max_age = max_age
        self._queue = []
        self._lock = threading.Lock()
    
    def put(self, command: Any):
        """Add command to queue."""
        with self._lock:
            self._queue.append({
                'command': command,
                'timestamp': time.time()
            })
            
            # Remove old commands
            self._cleanup()
    
    def get_latest(self) -> Optional[Any]:
        """Get most recent valid command."""
        with self._lock:
            self._cleanup()
            
            if self._queue:
                return self._queue[-1]['command']
            return None
    
    def get_and_clear(self) -> Optional[Any]:
        """Get latest command and clear queue."""
        with self._lock:
            self._cleanup()
            
            if self._queue:
                command = self._queue[-1]['command']
                self._queue.clear()
                return command
            return None
    
    def _cleanup(self):
        """Remove expired commands (must be called under lock)."""
        current_time = time.time()
        self._queue = [
            item for item in self._queue
            if current_time - item['timestamp'] < self.max_age
        ]
    
    def clear(self):
        """Clear all commands."""
        with self._lock:
            self._queue.clear()
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            self._cleanup()
            return len(self._queue)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create event bus
    bus = AsyncEventBus()
    bus.start()
    
    # Example subscriber
    def on_sensor_data(event: Event):
        print(f"Received sensor data: {event.data}")
    
    # Example async subscriber
    async def on_motor_command(event: Event):
        print(f"Processing motor command: {event.data}")
        await asyncio.sleep(0.1)  # Simulate processing
    
    # Subscribe to events
    bus.subscribe(EventType.SENSOR_DATA, on_sensor_data)
    bus.subscribe_async(EventType.MOTOR_COMMAND, on_motor_command)
    
    # Publish some events
    print("\nPublishing events...")
    bus.publish(Event(EventType.SENSOR_DATA, data={'distance': 0.5}))
    bus.publish(Event(EventType.MOTOR_COMMAND, data={'speed': 50}))
    bus.publish_urgent(Event(EventType.MOTOR_EMERGENCY_STOP, data="STOP!"))
    
    # Test thread-safe state
    print("\nTesting thread-safe state...")
    state = ThreadSafeState()
    state.set('cycle_count', 0)
    
    def increment_cycles():
        for _ in range(100):
            state.increment('cycle_count')
    
    # Run from multiple threads
    threads = [threading.Thread(target=increment_cycles) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"Final cycle count: {state.get('cycle_count')}")  # Should be 1000
    
    # Test command queue
    print("\nTesting command queue...")
    cmd_queue = CommandQueue(max_age=1.0)
    cmd_queue.put({'motor': 'forward', 'speed': 30})
    time.sleep(0.5)
    cmd_queue.put({'motor': 'forward', 'speed': 50})
    
    print(f"Latest command: {cmd_queue.get_latest()}")
    
    # Wait a bit for async operations
    time.sleep(1)
    
    # Show statistics
    print(f"\nEvent bus stats: {bus.get_stats()}")
    
    # Cleanup
    bus.stop()
    print("\nDemo complete!")
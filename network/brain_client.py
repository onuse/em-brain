"""
Brain Socket Client - Brainstem side of distributed brain architecture.
Connects to Brain Server and provides local interface for brainstem operations.
"""

import asyncio
import websockets
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.communication import SensoryPacket, PredictionPacket


class BrainSocketClient:
    """
    WebSocket client that connects brainstem to remote brain server.
    
    This runs on the Pi Zero 2 WH (or simulation) and provides a clean
    interface that looks like local function calls but uses network communication.
    """
    
    def __init__(self, brain_host: str = "localhost", brain_port: int = 8080,
                 client_name: str = "brainstem", reconnect_delay: float = 5.0):
        """
        Initialize brain socket client.
        
        Args:
            brain_host: Brain server hostname/IP
            brain_port: Brain server port
            client_name: Name identifier for this client
            reconnect_delay: Delay between reconnection attempts
        """
        self.brain_host = brain_host
        self.brain_port = brain_port
        self.client_name = client_name
        self.reconnect_delay = reconnect_delay
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.connection_attempts = 0
        
        # Statistics
        self.predictions_requested = 0
        self.connection_errors = 0
        self.last_prediction_time = 0.0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"BrainClient-{client_name}")
        
    async def connect(self) -> bool:
        """Connect to brain server."""
        try:
            uri = f"ws://{self.brain_host}:{self.brain_port}"
            self.logger.info(f"Connecting to brain server at {uri}")
            
            self.websocket = await websockets.connect(uri)
            self.connected = True
            self.connection_attempts += 1
            
            self.logger.info(f"Connected to brain server (attempt #{self.connection_attempts})")
            return True
            
        except Exception as e:
            self.connection_errors += 1
            self.logger.error(f"Failed to connect to brain server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from brain server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.connected = False
        self.logger.info("Disconnected from brain server")
    
    async def ensure_connected(self) -> bool:
        """Ensure connection to brain server, reconnect if needed."""
        if self.connected and self.websocket:
            return True
            
        return await self.connect()
    
    async def process_sensory_input(self, sensory_packet: SensoryPacket, 
                                  mental_context: List[float], 
                                  threat_level: str = "normal") -> Optional[PredictionPacket]:
        """
        Send sensory input to brain and get prediction back.
        
        This is the main interface that brainstems use - it looks like a local
        function call but actually uses network communication.
        """
        if not await self.ensure_connected():
            self.logger.error("Cannot process sensory input: not connected to brain")
            return None
        
        try:
            # Create request message
            request = {
                "type": "sensory_input",
                "sensory_packet": sensory_packet.to_json(),
                "mental_context": mental_context,
                "threat_level": threat_level,
                "client_name": self.client_name
            }
            
            # Send request
            await self.websocket.send(json.dumps(request))
            
            # Wait for response
            response_str = await self.websocket.recv()
            response = json.loads(response_str)
            
            if response.get("success"):
                # Parse prediction from response
                prediction = PredictionPacket.from_json(response["prediction"])
                
                self.predictions_requested += 1
                self.last_prediction_time = asyncio.get_event_loop().time()
                
                return prediction
            else:
                self.logger.error(f"Brain server error: {response.get('error', 'Unknown error')}")
                return None
                
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            self.logger.warning("Connection to brain server lost")
            return None
        except Exception as e:
            self.logger.error(f"Error communicating with brain server: {e}")
            return None
    
    async def get_brain_statistics(self) -> Optional[Dict[str, Any]]:
        """Request brain statistics from server."""
        if not await self.ensure_connected():
            return None
        
        try:
            request = {"type": "brain_stats", "client_name": self.client_name}
            await self.websocket.send(json.dumps(request))
            
            response_str = await self.websocket.recv()
            response = json.loads(response_str)
            
            if response.get("success"):
                return response["stats"]
            else:
                self.logger.error(f"Failed to get brain stats: {response.get('error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting brain statistics: {e}")
            return None
    
    async def reset_brain(self) -> bool:
        """Request brain reset."""
        if not await self.ensure_connected():
            return False
        
        try:
            request = {"type": "brain_reset", "client_name": self.client_name}
            await self.websocket.send(json.dumps(request))
            
            response_str = await self.websocket.recv()
            response = json.loads(response_str)
            
            if response.get("success"):
                self.logger.info("Brain reset successful")
                return True
            else:
                self.logger.error(f"Brain reset failed: {response.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error resetting brain: {e}")
            return False
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get client-side statistics."""
        return {
            "client_name": self.client_name,
            "connected": self.connected,
            "brain_host": self.brain_host,
            "brain_port": self.brain_port,
            "connection_attempts": self.connection_attempts,
            "connection_errors": self.connection_errors,
            "predictions_requested": self.predictions_requested,
            "last_prediction_time": self.last_prediction_time
        }
    
    async def run_with_reconnect(self, brainstem_loop_func):
        """
        Run brainstem control loop with automatic reconnection.
        
        Args:
            brainstem_loop_func: Async function that implements brainstem control loop
        """
        while True:
            try:
                if await self.ensure_connected():
                    await brainstem_loop_func(self)
                else:
                    self.logger.warning(f"Reconnection failed, waiting {self.reconnect_delay}s")
                    await asyncio.sleep(self.reconnect_delay)
                    
            except KeyboardInterrupt:
                self.logger.info("Shutting down brainstem client")
                break
            except Exception as e:
                self.logger.error(f"Brainstem loop error: {e}")
                await asyncio.sleep(self.reconnect_delay)
        
        await self.disconnect()


class LocalBrainClient:
    """
    Brain client that wraps socket communication to look like synchronous function calls.
    
    This provides a synchronous interface for brainstems that don't want to deal
    with async/await syntax.
    """
    
    def __init__(self, brain_host: str = "localhost", brain_port: int = 8080,
                 client_name: str = "sync_brainstem"):
        """Initialize synchronous brain client."""
        self.async_client = BrainSocketClient(brain_host, brain_port, client_name)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
    def connect(self) -> bool:
        """Connect to brain server (synchronous)."""
        return self._run_async(self.async_client.connect())
    
    def process_sensory_input(self, sensory_packet: SensoryPacket, 
                            mental_context: List[float], 
                            threat_level: str = "normal") -> Optional[PredictionPacket]:
        """Process sensory input (synchronous)."""
        return self._run_async(
            self.async_client.process_sensory_input(sensory_packet, mental_context, threat_level)
        )
    
    def get_brain_statistics(self) -> Optional[Dict[str, Any]]:
        """Get brain statistics (synchronous)."""
        return self._run_async(self.async_client.get_brain_statistics())
    
    def reset_brain(self) -> bool:
        """Reset brain (synchronous)."""
        return self._run_async(self.async_client.reset_brain())
    
    def disconnect(self):
        """Disconnect from brain server (synchronous)."""
        self._run_async(self.async_client.disconnect())
    
    def _run_async(self, coro):
        """Run async function in sync context."""
        if self.loop is None or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        return self.loop.run_until_complete(coro)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


async def test_brain_client():
    """Test function for brain client."""
    client = BrainSocketClient(client_name="test_client")
    
    try:
        if await client.connect():
            # Test sensory input
            sensory = SensoryPacket(
                sensor_values=[1.0, 2.0, 3.0, 4.0],
                actuator_positions=[0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=1
            )
            
            prediction = await client.process_sensory_input(
                sensory, [0.0, 1.0], "normal"
            )
            
            if prediction:
                print(f"Received prediction: {prediction.motor_action}")
                print(f"Confidence: {prediction.confidence}")
            
            # Test statistics
            stats = await client.get_brain_statistics()
            if stats:
                print(f"Brain has {stats['interface_stats']['sensory_vector_length']} sensors")
                
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_brain_client())
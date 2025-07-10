"""
Brain Socket Server - Laptop side of distributed brain architecture.
Receives sensory input from brainstem clients and returns predictions.
"""

import asyncio
import websockets
import json
import logging
from typing import Optional, Dict, Any
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket, PredictionPacket
from prediction.action.triple_predictor import TriplePredictor


class BrainSocketServer:
    """
    WebSocket server that hosts the brain for remote brainstem connections.
    
    This runs on the laptop (RTX 3070) and provides brain services to
    any brainstem client (Pi Zero 2 WH, simulation, etc.).
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080, 
                 base_time_budget: float = 0.1):
        """
        Initialize brain socket server.
        
        Args:
            host: Host address to bind to ("0.0.0.0" for all interfaces)
            port: Port to listen on
            base_time_budget: Base thinking time budget for predictor
        """
        self.host = host
        self.port = port
        
        # Initialize brain components
        predictor = TriplePredictor(base_time_budget=base_time_budget)
        self.brain_interface = BrainInterface(predictor)
        
        # Connection tracking
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.prediction_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def start_server(self):
        """Start the brain server and listen for connections."""
        self.logger.info(f"Starting Brain Server on {self.host}:{self.port}")
        self.logger.info(f"Brain ready to accept brainstem connections...")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            self.logger.info("Brain Server running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket):
        """Handle a single brainstem client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        try:
            self.logger.info(f"Brainstem connected: {client_id}")
            
            # Register client
            self.connected_clients[client_id] = {
                "websocket": websocket,
                "connected_at": asyncio.get_event_loop().time(),
                "predictions_served": 0
            }
            
            # Handle messages from this client
            async for message in websocket:
                try:
                    response = await self.process_brainstem_message(message, client_id)
                    await websocket.send(response)
                    
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_id}: {e}")
                    error_response = self.create_error_response(str(e))
                    await websocket.send(error_response)
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Brainstem disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Client handler error for {client_id}: {e}")
        finally:
            # Cleanup client
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
    
    async def process_brainstem_message(self, message: str, client_id: str) -> str:
        """Process a message from a brainstem client."""
        try:
            # Parse the incoming message
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "sensory_input":
                return await self.handle_sensory_input(data, client_id)
            elif message_type == "brain_stats":
                return self.handle_stats_request(client_id)
            elif message_type == "brain_reset":
                return self.handle_reset_request(client_id)
            else:
                raise ValueError(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    async def handle_sensory_input(self, data: Dict[str, Any], client_id: str) -> str:
        """Handle sensory input from brainstem and return prediction."""
        try:
            # Deserialize sensory packet
            sensory_packet = SensoryPacket.from_json(data["sensory_packet"])
            mental_context = data.get("mental_context", [0.0])
            threat_level = data.get("threat_level", "normal")
            
            # Process through brain interface
            prediction = self.brain_interface.process_sensory_input(
                sensory_packet, mental_context, threat_level
            )
            
            # Update client stats
            self.connected_clients[client_id]["predictions_served"] += 1
            self.prediction_count += 1
            
            # Log every 10th prediction
            if self.prediction_count % 10 == 0:
                self.logger.info(f"Served {self.prediction_count} predictions to {len(self.connected_clients)} clients")
            
            # Return prediction
            return json.dumps({
                "type": "prediction_response",
                "prediction": prediction.to_json(),
                "success": True
            })
            
        except Exception as e:
            raise ValueError(f"Error processing sensory input: {e}")
    
    def handle_stats_request(self, client_id: str) -> str:
        """Handle request for brain statistics."""
        stats = self.brain_interface.get_brain_statistics()
        
        # Add server stats
        stats["server_stats"] = {
            "connected_clients": len(self.connected_clients),
            "total_predictions": self.prediction_count,
            "client_info": {
                cid: {
                    "predictions_served": info["predictions_served"],
                    "connected_duration": asyncio.get_event_loop().time() - info["connected_at"]
                }
                for cid, info in self.connected_clients.items()
            }
        }
        
        return json.dumps({
            "type": "stats_response",
            "stats": stats,
            "success": True
        })
    
    def handle_reset_request(self, client_id: str) -> str:
        """Handle request to reset brain state."""
        self.brain_interface.reset_brain()
        self.logger.info(f"Brain reset requested by {client_id}")
        
        return json.dumps({
            "type": "reset_response", 
            "success": True,
            "message": "Brain state reset successfully"
        })
    
    def create_error_response(self, error_message: str) -> str:
        """Create standardized error response."""
        return json.dumps({
            "type": "error",
            "success": False,
            "error": error_message
        })
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about server state."""
        return {
            "host": self.host,
            "port": self.port,
            "connected_clients": len(self.connected_clients),
            "total_predictions": self.prediction_count,
            "brain_stats": self.brain_interface.get_brain_statistics()
        }


async def main():
    """Run the brain server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain Socket Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--time-budget", type=float, default=0.1, help="Base time budget for thinking")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start server
    server = BrainSocketServer(args.host, args.port, args.time_budget)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\\nShutting down Brain Server...")
        print(f"Final stats: {server.get_server_info()}")


if __name__ == "__main__":
    asyncio.run(main())
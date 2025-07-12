"""
Minimal Brain TCP Server Entry Point

Simple server that hosts the minimal brain and accepts connections
from any client (Pi Zero brainstems, simulations, mobile apps, etc.)
"""

import sys
import signal
from typing import Optional

from src.brain import MinimalBrain
from src.communication import MinimalTCPServer


class MinimalBrainServer:
    """Main server process for the minimal brain."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9999):
        self.host = host
        self.port = port
        self.brain = MinimalBrain()
        self.tcp_server = MinimalTCPServer(self.brain, host, port)
        self.running = False
        
    def start(self):
        """Start the minimal brain server."""
        print("🚀 Starting Minimal Brain Server")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Brain: {self.brain}")
        
        # Setup graceful shutdown (only in main thread)
        try:
            signal.signal(signal.SIGINT, self._shutdown_handler)
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except ValueError:
            # Not in main thread - signals can't be set
            pass
        
        try:
            self.running = True
            self.tcp_server.start()
        except Exception as e:
            print(f"❌ Server error: {e}")
            self.shutdown()
    
    def stop(self):
        """Stop the server gracefully."""
        if self.running:
            print("🛑 Shutting down Minimal Brain Server")
            self.running = False
            self.tcp_server.stop()
            
            # Print final stats
            stats = self.brain.get_brain_stats()
            print(f"📊 Final Stats: {stats}")
    
    def shutdown(self):
        """Alias for stop() for backward compatibility."""
        self.stop()
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"🔔 Received signal {signum}")
        self.shutdown()
        sys.exit(0)


def main():
    """Entry point for minimal brain server."""
    server = MinimalBrainServer(host='0.0.0.0', port=9999)
    
    try:
        server.start()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
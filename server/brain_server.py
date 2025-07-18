"""
Minimal Brain TCP Server Entry Point

Simple server that hosts the minimal brain and accepts connections
from any client (Pi Zero brainstems, simulations, mobile apps, etc.)
"""

import sys
import signal
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, '..')
from src.brain import MinimalBrain
from src.communication import MinimalTCPServer


class MinimalBrainServer:
    """Main server process for the minimal brain."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9999):
        self.host = host
        self.port = port
        
        # Load settings and setup paths
        self.config = self._load_settings()
        self._setup_paths()
        
        # Initialize brain with config - use quiet mode to reduce startup spam
        self.brain = MinimalBrain(config=self.config, quiet_mode=True)
        self.tcp_server = MinimalTCPServer(self.brain, host, port)
        self.running = False
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from settings.json with proper path resolution."""
        # Get the server directory (where this file is located)
        server_dir = Path(__file__).parent
        settings_file = server_dir / "settings.json"
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load settings.json: {e}")
                return {}
        else:
            print(f"⚠️ Warning: settings.json not found at {settings_file}")
            return {}
    
    def _setup_paths(self):
        """Setup absolute paths for logs and robot_memory."""
        # Always use the brain root directory (parent of server directory)
        brain_root = Path(__file__).parent.parent
        
        # Convert relative paths to absolute paths
        if 'logging' in self.config:
            log_dir = self.config['logging'].get('log_directory', './logs')
            self.config['logging']['log_directory'] = str(brain_root / log_dir.lstrip('./'))
        else:
            self.config['logging'] = {'log_directory': str(brain_root / 'logs')}
        
        if 'memory' in self.config:
            memory_path = self.config['memory'].get('persistent_memory_path', './robot_memory')
            self.config['memory']['persistent_memory_path'] = str(brain_root / memory_path.lstrip('./'))
        else:
            self.config['memory'] = {'persistent_memory_path': str(brain_root / 'robot_memory')}
        
        # Create directories if they don't exist
        os.makedirs(self.config['logging']['log_directory'], exist_ok=True)
        os.makedirs(self.config['memory']['persistent_memory_path'], exist_ok=True)
        
        print(f"📁 Logs directory: {self.config['logging']['log_directory']}")
        print(f"📁 Memory directory: {self.config['memory']['persistent_memory_path']}")
        
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
            
            # Finalize brain session (saves final checkpoint)
            self.brain.finalize_session()
            
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
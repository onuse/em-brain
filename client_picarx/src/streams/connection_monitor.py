"""
Connection Monitor - Shared state for auxiliary streams.

When the main brain connection drops, all auxiliary streams should stop.
This provides a simple shared flag that streams can check.
"""

import threading
import time


class ConnectionMonitor:
    """
    Singleton that tracks main brain connection state.
    All auxiliary streams check this to know when to stop.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.main_connected = False
        self.connection_time = None
        self.disconnect_time = None
        self.callbacks = []  # Functions to call on state change
        
    def set_connected(self, connected: bool):
        """Update connection state."""
        if connected != self.main_connected:
            self.main_connected = connected
            
            if connected:
                self.connection_time = time.time()
                self.disconnect_time = None
                print("📡 ConnectionMonitor: Main brain connection established")
            else:
                self.disconnect_time = time.time()
                print("🔌 ConnectionMonitor: Main brain connection lost")
                
            # Notify all callbacks
            for callback in self.callbacks:
                try:
                    callback(connected)
                except Exception as e:
                    print(f"⚠️ Callback error: {e}")
    
    def is_connected(self) -> bool:
        """Check if main connection is active."""
        return self.main_connected
    
    def register_callback(self, callback):
        """Register a function to call when connection state changes."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_uptime(self) -> float:
        """Get seconds since connection established."""
        if self.main_connected and self.connection_time:
            return time.time() - self.connection_time
        return 0.0
    
    def get_downtime(self) -> float:
        """Get seconds since connection lost."""
        if not self.main_connected and self.disconnect_time:
            return time.time() - self.disconnect_time
        return 0.0


# Global singleton instance
connection_monitor = ConnectionMonitor()
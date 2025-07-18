"""
Simple Sensor Buffer System

Decouples brain processing from socket I/O by buffering the latest sensor input.
This is Step 1 toward a predictive brain architecture.

Phase 1 goals:
- Buffer latest sensor input from each client
- Brain reads from buffer on its own schedule  
- Discard old data - only keep most recent
- Minimal disruption to existing architecture
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SensorData:
    """Container for sensor input with metadata."""
    vector: List[float]
    timestamp: float
    client_id: str
    sequence_id: int = 0


class SensorBuffer:
    """
    Thread-safe buffer for sensor data.
    
    Keeps only the latest sensor input per client to avoid memory buildup.
    Brain can read at its own pace without blocking on socket I/O.
    """
    
    def __init__(self, max_age_seconds: float = 1.0):
        """
        Initialize sensor buffer.
        
        Args:
            max_age_seconds: Maximum age of sensor data before considering it stale
        """
        self.max_age_seconds = max_age_seconds
        self.client_data: Dict[str, SensorData] = {}
        self.lock = threading.Lock()
        self.total_inputs_received = 0
        self.total_inputs_discarded = 0
        
    def add_sensor_input(self, client_id: str, sensor_vector: List[float]) -> bool:
        """
        Add new sensor input from a client.
        
        Replaces any existing data for this client (keeping only latest).
        
        Args:
            client_id: ID of the client sending the data
            sensor_vector: Sensor readings as vector
            
        Returns:
            True if data was stored, False if rejected
        """
        if not sensor_vector or len(sensor_vector) == 0:
            return False
            
        with self.lock:
            current_time = time.time()
            
            # Track statistics
            self.total_inputs_received += 1
            if client_id in self.client_data:
                self.total_inputs_discarded += 1  # Previous data discarded
            
            # Store latest data (replace any existing)
            self.client_data[client_id] = SensorData(
                vector=sensor_vector.copy(),
                timestamp=current_time,
                client_id=client_id,
                sequence_id=self.total_inputs_received
            )
            
            return True
    
    def get_latest_sensor_data(self, client_id: str) -> Optional[SensorData]:
        """
        Get the latest sensor data for a specific client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Latest sensor data or None if no data available/too old
        """
        with self.lock:
            if client_id not in self.client_data:
                return None
                
            data = self.client_data[client_id]
            
            # Check if data is too old
            age = time.time() - data.timestamp
            if age > self.max_age_seconds:
                # Remove stale data
                del self.client_data[client_id]
                return None
                
            return data
    
    def get_all_latest_data(self) -> Dict[str, SensorData]:
        """
        Get latest sensor data for all clients.
        
        Returns:
            Dictionary of client_id -> SensorData for all active clients
        """
        with self.lock:
            current_time = time.time()
            active_data = {}
            stale_clients = []
            
            for client_id, data in self.client_data.items():
                age = current_time - data.timestamp
                if age <= self.max_age_seconds:
                    active_data[client_id] = data
                else:
                    stale_clients.append(client_id)
            
            # Remove stale data
            for client_id in stale_clients:
                del self.client_data[client_id]
                
            return active_data
    
    def has_data_for_client(self, client_id: str) -> bool:
        """Check if there's fresh data available for a client."""
        return self.get_latest_sensor_data(client_id) is not None
    
    def has_any_data(self) -> bool:
        """Check if there's any fresh sensor data available."""
        with self.lock:
            current_time = time.time()
            for data in self.client_data.values():
                if (current_time - data.timestamp) <= self.max_age_seconds:
                    return True
            return False
    
    def get_active_client_count(self) -> int:
        """Get number of clients with fresh data."""
        return len(self.get_all_latest_data())
    
    def clear_client_data(self, client_id: str):
        """Remove all data for a specific client (when they disconnect)."""
        with self.lock:
            if client_id in self.client_data:
                del self.client_data[client_id]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get buffer statistics for monitoring."""
        with self.lock:
            active_clients = self.get_active_client_count()
            
            return {
                'total_inputs_received': self.total_inputs_received,
                'total_inputs_discarded': self.total_inputs_discarded,
                'active_clients': active_clients,
                'discard_rate': self.total_inputs_discarded / max(1, self.total_inputs_received),
                'buffer_efficiency': 1.0 - (self.total_inputs_discarded / max(1, self.total_inputs_received))
            }
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"SensorBuffer(clients={stats['active_clients']}, received={stats['total_inputs_received']}, discard_rate={stats['discard_rate']:.2%})"


# Global sensor buffer instance
_global_sensor_buffer = None


def get_sensor_buffer() -> SensorBuffer:
    """Get the global sensor buffer instance."""
    global _global_sensor_buffer
    if _global_sensor_buffer is None:
        _global_sensor_buffer = SensorBuffer()
    return _global_sensor_buffer
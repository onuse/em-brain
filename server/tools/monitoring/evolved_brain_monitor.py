#!/usr/bin/env python3
"""
Evolved Brain Monitor

Real-time monitoring tool for the evolved brain architecture.
Displays comprehensive telemetry including evolution dynamics,
regional specialization, and emergent behaviors.
"""

import socket
import json
import time
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class EvolvedBrainMonitor:
    """Monitor client for evolved brain telemetry."""
    
    def __init__(self, host: str = 'localhost', port: int = 9998):
        """Initialize monitor client."""
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to monitoring server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(5.0)
            
            # Receive welcome message
            welcome = self._receive_json()
            if welcome and welcome.get('status') == 'connected':
                self.connected = True
                print(f"✅ Connected to {welcome.get('server', 'monitoring server')}")
                return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
        
        return False
    
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
    
    def _send_command(self, command: str) -> bool:
        """Send command to server."""
        try:
            self.socket.send(f"{command}\n".encode('utf-8'))
            return True
        except Exception as e:
            print(f"❌ Send error: {e}")
            return False
    
    def _receive_json(self) -> Optional[Dict[str, Any]]:
        """Receive JSON response from server."""
        try:
            data = self.socket.recv(4096).decode('utf-8')
            return json.loads(data)
        except Exception as e:
            print(f"❌ Receive error: {e}")
            return None
    
    def get_telemetry(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get telemetry data."""
        command = f"telemetry {session_id}" if session_id else "telemetry"
        
        if self._send_command(command):
            return self._receive_json()
        return None
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get session information."""
        if self._send_command("session_info"):
            return self._receive_json()
        return None
    
    def get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics."""
        if self._send_command("performance_metrics"):
            return self._receive_json()
        return None
    
    def display_evolved_telemetry(self, telemetry_data: Dict[str, Any]):
        """Display evolved brain telemetry in a formatted way."""
        
        print("\n" + "="*80)
        print("🧠 EVOLVED BRAIN TELEMETRY")
        print("="*80)
        
        data = telemetry_data.get('data', {})
        
        # If multiple sessions, show summary
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            print(f"\n📊 Active Sessions: {len(data)}")
            
            for session_id, session_data in data.items():
                print(f"\n🔹 Session: {session_id}")
                self._display_session_telemetry(session_data)
        else:
            # Single session telemetry
            self._display_session_telemetry(data)
    
    def _display_session_telemetry(self, data: Dict[str, Any]):
        """Display telemetry for a single session."""
        
        # Core metrics
        print(f"\n⚡ Core Metrics:")
        print(f"   Cycles: {data.get('cycle', 0):,}")
        print(f"   Energy: {data.get('field_energy', 0):.4f}")
        print(f"   Prediction Confidence: {data.get('prediction_confidence', 0):.1%}")
        print(f"   Cycle Time: {data.get('cycle_time_ms', 0):.1f}ms")
        
        # Evolution dynamics
        evolution = data.get('evolution_state', {})
        if evolution:
            print(f"\n🧬 Evolution Dynamics:")
            print(f"   Evolution Cycles: {evolution.get('evolution_cycles', 0):,}")
            print(f"   Self-Modification: {evolution.get('self_modification_strength', 0):.1%}")
            print(f"   Smoothed Energy: {evolution.get('smoothed_energy', 0):.4f}")
            print(f"   Smoothed Confidence: {evolution.get('smoothed_confidence', 0):.4f}")
            
            # Working memory
            wm = evolution.get('working_memory', {})
            if wm:
                print(f"   Working Memory Patterns: {wm.get('n_patterns', 0)}")
        
        # Field dynamics
        energy_state = data.get('energy_state', {})
        if energy_state:
            print(f"\n🌊 Field Dynamics:")
            print(f"   Energy Level: {energy_state.get('energy', 0):.3f}")
            print(f"   Novelty: {energy_state.get('novelty', 0):.3f}")
            print(f"   Exploration Drive: {energy_state.get('exploration_drive', 0):.3f}")
        
        # Topology regions
        topology = data.get('topology_regions', {})
        if topology:
            print(f"\n🏔️ Topology Regions:")
            print(f"   Total: {topology.get('total', 0)}")
            print(f"   Active: {topology.get('active', 0)}")
            print(f"   Abstract: {topology.get('abstract', 0)}")
            print(f"   Causal Links: {topology.get('causal_links', 0)}")
            print(f"   Memory Saturation: {data.get('memory_saturation', 0):.1%}")
        
        # Sensory organization
        sensory = data.get('sensory_organization', {})
        if sensory:
            print(f"\n👁️ Sensory Organization:")
            print(f"   Unique Patterns: {sensory.get('unique_patterns', 0)}")
            print(f"   Mapping Events: {sensory.get('mapping_events', 0):,}")
            print(f"   Clustering Coefficient: {sensory.get('clustering_coefficient', 0):.3f}")
            print(f"   Occupancy Ratio: {sensory.get('occupancy_ratio', 0):.1%}")
        
        # Reward topology
        reward = data.get('topology_shaping', {})
        if reward:
            print(f"\n🎯 Reward Topology:")
            print(f"   Active Impressions: {reward.get('active_impressions', 0)}")
            print(f"   Attractor Strength: {reward.get('attractor_strength', 0):.3f}")
            print(f"   Repulsor Strength: {reward.get('repulsor_strength', 0):.3f}")
        
        # Device info
        print(f"\n💻 System:")
        print(f"   Device: {data.get('device', 'unknown')}")
        print(f"   Tensor Shape: {data.get('tensor_shape', [])}")
    
    def run_continuous_monitoring(self, refresh_interval: float = 1.0):
        """Run continuous monitoring with periodic updates."""
        
        print(f"\n🔄 Starting continuous monitoring (refresh every {refresh_interval}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.connected:
                # Clear screen (works on Unix-like systems)
                os.system('clear' if os.name != 'nt' else 'cls')
                
                # Display timestamp
                print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get and display telemetry
                telemetry = self.get_telemetry()
                if telemetry and telemetry.get('status') == 'success':
                    self.display_evolved_telemetry(telemetry)
                else:
                    print("⚠️  Failed to get telemetry")
                
                # Get performance metrics
                metrics = self.get_performance_metrics()
                if metrics and metrics.get('status') == 'success':
                    perf_data = metrics.get('data', {})
                    print(f"\n📈 Performance Overview:")
                    print(f"   Total Cycles: {perf_data.get('total_cycles', 0):,}")
                    print(f"   Active Sessions: {perf_data.get('active_sessions', 0)}")
                    print(f"   Avg Cycle Time: {perf_data.get('average_cycle_time_ms', 0):.1f}ms")
                
                time.sleep(refresh_interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped")
    
    def run_interactive(self):
        """Run interactive monitoring mode."""
        
        print("\n🎮 Interactive Mode")
        print("Commands: telemetry, sessions, performance, quit")
        
        while self.connected:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                
                elif command == 'telemetry' or command == 't':
                    telemetry = self.get_telemetry()
                    if telemetry and telemetry.get('status') == 'success':
                        self.display_evolved_telemetry(telemetry)
                
                elif command == 'sessions' or command == 's':
                    sessions = self.get_session_info()
                    if sessions and sessions.get('status') == 'success':
                        self._display_sessions(sessions.get('data', []))
                
                elif command == 'performance' or command == 'p':
                    metrics = self.get_performance_metrics()
                    if metrics and metrics.get('status') == 'success':
                        self._display_performance(metrics.get('data', {}))
                
                elif command.startswith('telemetry '):
                    session_id = command.split(' ', 1)[1]
                    telemetry = self.get_telemetry(session_id)
                    if telemetry and telemetry.get('status') == 'success':
                        self.display_evolved_telemetry(telemetry)
                
                else:
                    print("❓ Unknown command. Try: telemetry, sessions, performance, quit")
            
            except KeyboardInterrupt:
                break
        
        print("\n👋 Goodbye!")
    
    def _display_sessions(self, sessions: list):
        """Display session information."""
        print(f"\n📊 Active Sessions ({len(sessions)})")
        print("-" * 60)
        
        for session in sessions:
            print(f"\n🔹 Session: {session.get('session_id', 'unknown')}")
            print(f"   Robot Type: {session.get('robot_type', 'unknown')}")
            print(f"   Dimensions: {session.get('brain_dimensions', 0)}")
            print(f"   Cycles: {session.get('cycles', 0):,}")
            print(f"   Experiences: {session.get('experiences', 0):,}")
            print(f"   Uptime: {session.get('uptime', 0):.1f}s")
    
    def _display_performance(self, metrics: dict):
        """Display performance metrics."""
        print("\n📈 Performance Metrics")
        print("-" * 40)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'time' in key:
                    print(f"   {key}: {value:.1f}ms")
                else:
                    print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Monitor evolved brain telemetry')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=9998, help='Server port')
    parser.add_argument('--mode', choices=['continuous', 'interactive'], 
                       default='continuous', help='Monitoring mode')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Refresh interval for continuous mode (seconds)')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = EvolvedBrainMonitor(host=args.host, port=args.port)
    
    # Connect
    if not monitor.connect():
        sys.exit(1)
    
    try:
        # Run monitoring
        if args.mode == 'continuous':
            monitor.run_continuous_monitoring(refresh_interval=args.interval)
        else:
            monitor.run_interactive()
    
    finally:
        monitor.disconnect()


if __name__ == "__main__":
    main()
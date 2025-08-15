#!/usr/bin/env python3
"""
Dynamic Brain Server - Main Entry Point

This server implements the new architecture where brains are created
dynamically based on robot capabilities, not predetermined at startup.
"""

import sys
import os
import signal
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our new architecture components
from src.core.interfaces import *
from src.core.robot_registry import RobotRegistry
from src.core.single_brain_pool import SingleBrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.unified_adapters import UnifiedAdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.unified_brain_factory import UnifiedBrainFactory
from src.core.monitoring_server import DynamicMonitoringServer
from src.core.maintenance_scheduler import MaintenanceScheduler
from src.communication.clean_tcp_server import CleanTCPServer

# Optional: Stream listeners for multi-stream sensors
try:
    from src.streams.sensor_listeners import StreamManager
    from src.streams.field_injection_threads import SensorFieldInjectionManager
    STREAMS_AVAILABLE = True
except ImportError:
    STREAMS_AVAILABLE = False
    print("ℹ️  Stream listeners not available (optional feature)")

# Configuration
from src.adaptive_configuration import AdaptiveConfigurationManager
from src.config.enhanced_gpu_memory_manager import configure_gpu_memory, print_gpu_memory_report


class DynamicBrainServer:
    """
    Main server class that wires together all components.
    
    This server implements true dynamic brain creation where the brain
    structure adapts to each robot's capabilities.
    """
    
    def __init__(self, config_file: str = "settings.json", safe_mode: bool = False):
        print("\n🧠 Initializing Dynamic Brain Server...")
        
        # Use safe configuration if requested
        if safe_mode:
            config_file = "settings_safe.json"
            print("⚠️  SAFE MODE ENABLED - Using conservative parameters")
        
        # Load configuration
        self.config_manager = AdaptiveConfigurationManager(config_file)
        self.config = self.config_manager.config.__dict__
        
        # Configure GPU memory management
        configure_gpu_memory(self.config)
        
        # Display hardware detection results with final spatial resolution
        self.config_manager.print_summary()
        
        # Initialize components in dependency order
        print("   📦 Initializing components...")
        self._initialize_components()
        
        # Server state
        self.running = False
        self.tcp_server = None
        
        # Optional: Start stream listeners if available
        self.stream_manager = None
        if STREAMS_AVAILABLE and self.config.get('network', {}).get('enable_streams', False):
            print("   🌊 Starting sensor stream listeners...")
            self.stream_manager = StreamManager()
            self.stream_manager.start_all()
    
    def _initialize_components(self):
        """Initialize all components with proper dependency injection."""
        
        # 1. Robot Registry - manages robot profiles
        profiles_dir = Path(__file__).parent.parent / "client_picarx"
        self.robot_registry = RobotRegistry(profiles_dir=profiles_dir)
        
        # 2. Brain Factory - creates brains
        self.brain_factory = UnifiedBrainFactory(brain_config=self.config)
        
        # 3. Brain Pool - single brain locked to first brainstem dimensions
        self.brain_pool = SingleBrainPool(brain_factory=self.brain_factory)
        
        # 4. Adapter Factory - creates sensory/motor adapters
        # Use simplified adapters when using simplified brain
        if isinstance(self.brain_factory, UnifiedBrainFactory):
            self.adapter_factory = UnifiedAdapterFactory()
        else:
            self.adapter_factory = AdapterFactory()
        
        # 5. Brain Service - manages sessions with persistence
        persistence_config = self.config.get('persistence', {})
        self.brain_service = BrainService(
            brain_pool=self.brain_pool,
            adapter_factory=self.adapter_factory,
            memory_path=persistence_config.get('memory_path', './brain_memory'),
            save_interval_cycles=persistence_config.get('save_interval', 1000),
            auto_save=persistence_config.get('auto_save', True),
            enable_logging=self.config.get('logging', {}).get('enabled', True),
            quiet=True  # Suppress duplicate persistence prints
        )
        
        # 6. Connection Handler - orchestrates everything
        self.connection_handler = ConnectionHandler(
            robot_registry=self.robot_registry,
            brain_service=self.brain_service
        )
        
        # 7. Maintenance Scheduler - manages brain health
        self.maintenance_scheduler = MaintenanceScheduler(
            brain_pool=self.brain_pool,
            brain_service=self.brain_service,
            config={
                'field_maintenance_interval': 300,  # 5 minutes
                'memory_check_interval': 60,        # 1 minute
                'performance_check_interval': 120   # 2 minutes
            }
        )
        
        # Components initialized silently
    
    def _display_system_info(self):
        """Display consolidated system information."""
        # Skip - we'll print a cleaner summary at the end
        pass
    
    def start(self):
        """Start the dynamic brain server."""
        
        self.running = True
        
        # Get network configuration
        host = self.config.get('network', {}).get('host', '0.0.0.0')
        port = self.config.get('network', {}).get('port', 9999)
        monitoring_port = port - 1  # Use port 9998 for monitoring
        
        # Create TCP server
        self.tcp_server = CleanTCPServer(
            connection_handler=self.connection_handler,
            host=host,
            port=port
        )
        
        # Create and start monitoring server
        self.monitoring_server = DynamicMonitoringServer(
            brain_service=self.brain_service,
            connection_handler=self.connection_handler,
            host=host,
            port=monitoring_port
        )
        self.monitoring_server.start()
        
        # Start maintenance scheduler
        self.maintenance_scheduler.start()
        
        # Print clean startup summary
        import platform
        import psutil
        import torch
        
        print("\n" + "=" * 70)
        print("🧠 DYNAMIC BRAIN SERVER READY")
        print("=" * 70)
        
        # System summary
        mem = psutil.virtual_memory()
        device = self.config_manager.get_device()
        spatial_res = self.config_manager.config.spatial_resolution
        
        # Hardware line
        print(f"💻 Hardware: {device.type.upper()} | {psutil.cpu_count()} cores | "
              f"{mem.total/(1024**3):.1f}GB RAM", end="")
        if self.config_manager.config.gpu_memory_gb > 0:
            print(f" | {self.config_manager.config.gpu_memory_gb:.1f}GB GPU")
        else:
            print()
        
        # Brain configuration line
        print(f"🧠 Brain: 4D Field [{spatial_res}³×64] | ", end="")
        if self.brain_pool.brain is None:
            print("Dimensions: <pending robot connection>")
        else:
            dims = self.brain_pool.locked_dimensions
            if dims:
                print(f"Dimensions: {dims['sensory_dim']}→{dims['motor_dim']} (locked)")
            else:
                print("Dimensions: <initializing>")
        
        # Field dynamics parameters (show PureFieldBrain's aggressive settings)
        print(f"⚡ Dynamics: Aggressive learning enabled | Auto-scaling by hardware")
        
        # Network line
        print(f"🌐 Network: {host}:{port} (main) | {monitoring_port} (monitoring)")
        
        # Features line
        features = []
        if self.config_manager.config.enhanced_dynamics:
            features.append("enhanced dynamics")
        if self.config_manager.config.attention_guidance:
            features.append("attention")
        if self.config_manager.config.hierarchical_processing:
            features.append("hierarchical")
        if features:
            print(f"✨ Features: {', '.join(features)}")
        
        print("=" * 70)
        
        try:
            # Start server (blocks until stopped)
            self.tcp_server.start()
        except KeyboardInterrupt:
            print("\n⌨️  Keyboard interrupt received")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server gracefully."""
        
        if not self.running:
            return
        
        print("\n🛑 Shutting down Dynamic Brain Server...")
        self.running = False
        
        # Stop maintenance scheduler first
        if hasattr(self, 'maintenance_scheduler') and self.maintenance_scheduler:
            self.maintenance_scheduler.stop()
        
        # Stop monitoring server
        if hasattr(self, 'monitoring_server') and self.monitoring_server:
            self.monitoring_server.stop()
        
        # Stop TCP server
        if self.tcp_server:
            self.tcp_server.stop()
        
        # Shutdown brain service (includes persistence)
        if self.brain_service:
            print("\n💾 Saving brain states...")
            self.brain_service.shutdown()
        
        # Print final statistics
        self._print_statistics()
        
        print("\n👋 Dynamic Brain Server stopped")
    
    def _print_statistics(self):
        """Print server statistics."""
        
        print("\n📊 Server Statistics")
        print("=" * 30)
        
        # Connection stats
        conn_stats = self.connection_handler.get_stats()
        print(f"Total connections: {conn_stats['total_connections']}")
        print(f"Active connections: {conn_stats['active_connections']}")
        print(f"Total messages: {conn_stats['total_messages']}")
        
        # Brain stats
        active_brains = self.brain_pool.get_active_brains()
        print(f"\nActive brains: {len(active_brains)}")
        for profile, brain in active_brains.items():
            print(f"  - {profile}: {brain.get_field_dimensions()}D field")
        
        # Session stats
        if conn_stats['active_sessions']:
            print("\nActive sessions:")
            for session in conn_stats['active_sessions']:
                print(f"  - {session['robot_type']} robot: "
                      f"{session['cycles_processed']} cycles, "
                      f"{session['average_cycle_time_ms']:.1f}ms avg")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Dynamic Brain Server - Adaptive robotic intelligence'
    )
    parser.add_argument(
        '--config', '-c',
        default='settings.json',
        help='Configuration file (default: settings.json)'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run in test mode (exits after initialization)'
    )
    parser.add_argument(
        '--safe-mode', '-s',
        action='store_true',
        help='Run in safe mode with conservative parameters for initial robot testing'
    )
    parser.add_argument(
        '--aggressive', '-a',
        action='store_true',
        help='Run with aggressive learning parameters (use after validation)'
    )
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.safe_mode and args.aggressive:
        print("❌ Cannot use both --safe-mode and --aggressive")
        return 1
    
    # Create server with appropriate mode
    server = DynamicBrainServer(
        config_file=args.config if not args.safe_mode else args.config,
        safe_mode=args.safe_mode
    )
    
    # Test mode - just verify initialization
    if args.test:
        print("\n✅ Server initialized successfully (test mode)")
        return 0
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    try:
        server.start()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
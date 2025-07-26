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
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.monitoring_server import DynamicMonitoringServer
from src.core.maintenance_scheduler import MaintenanceScheduler
from src.communication.clean_tcp_server import CleanTCPServer

# Configuration
from src.adaptive_configuration import AdaptiveConfigurationManager


class DynamicBrainServer:
    """
    Main server class that wires together all components.
    
    This server implements true dynamic brain creation where the brain
    structure adapts to each robot's capabilities.
    """
    
    def __init__(self, config_file: str = "settings.json"):
        print("🧠 Initializing Dynamic Brain Server")
        print("=" * 50)
        
        # Load configuration
        self.config_manager = AdaptiveConfigurationManager(config_file)
        self.config = self.config_manager.config.__dict__
        
        # Display system information
        self._display_system_info()
        
        # Initialize components in dependency order
        self._initialize_components()
        
        # Server state
        self.running = False
        self.tcp_server = None
    
    def _initialize_components(self):
        """Initialize all components with proper dependency injection."""
        
        print("📦 Initializing components...")
        
        # 1. Robot Registry - manages robot profiles
        profiles_dir = Path(__file__).parent.parent / "client_picarx"
        self.robot_registry = RobotRegistry(profiles_dir=profiles_dir)
        print("   ✓ Robot Registry")
        
        # 2. Brain Factory - creates brains
        self.brain_factory = DynamicBrainFactory(config=self.config)
        print("   ✓ Dynamic Brain Factory")
        
        # 3. Brain Pool - manages brain instances
        self.brain_pool = BrainPool(brain_factory=self.brain_factory)
        print("   ✓ Brain Pool")
        
        # 4. Adapter Factory - creates sensory/motor adapters
        self.adapter_factory = AdapterFactory()
        print("   ✓ Adapter Factory")
        
        # 5. Brain Service - manages sessions
        self.brain_service = BrainService(
            brain_pool=self.brain_pool,
            adapter_factory=self.adapter_factory
        )
        print("   ✓ Brain Service")
        
        # 6. Connection Handler - orchestrates everything
        self.connection_handler = ConnectionHandler(
            robot_registry=self.robot_registry,
            brain_service=self.brain_service
        )
        print("   ✓ Connection Handler")
        
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
        print("   ✓ Maintenance Scheduler")
        
        print("✅ All components initialized")
    
    def _display_system_info(self):
        """Display system information on startup."""
        import platform
        import psutil
        import torch
        
        print("\n📱 System Information")
        print("=" * 50)
        
        # OS and Python info
        print(f"🖥️  Operating System: {platform.system()} {platform.release()}")
        print(f"🐍 Python Version: {platform.python_version()}")
        print(f"📍 Working Directory: {os.getcwd()}")
        
        # CPU info
        print(f"\n💻 CPU Information:")
        print(f"   Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print(f"   Architecture: {platform.machine()}")
        
        # Memory info
        mem = psutil.virtual_memory()
        print(f"\n🧠 Memory Information:")
        print(f"   Total: {mem.total / (1024**3):.1f} GB")
        print(f"   Available: {mem.available / (1024**3):.1f} GB")
        print(f"   Used: {mem.percent:.1f}%")
        
        # GPU info
        print(f"\n🎮 GPU Information:")
        if torch.cuda.is_available():
            print(f"   CUDA Available: ✅")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Device Count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                props = torch.cuda.get_device_properties(0)
                print(f"   GPU 0: {props.name}")
                print(f"   GPU Memory: {props.total_memory / (1024**3):.1f} GB")
        elif torch.backends.mps.is_available():
            print(f"   MPS (Metal) Available: ✅")
            print(f"   Device: Apple Silicon GPU")
        else:
            print(f"   No GPU acceleration available")
        
        # PyTorch info
        print(f"\n🔥 PyTorch Information:")
        print(f"   Version: {torch.__version__}")
        print(f"   Device: {self.config_manager.get_device()}")
        
        # Network info
        print(f"\n🌐 Network Configuration:")
        print(f"   Brain Server Port: {self.config.get('port', 9999)}")
        print(f"   Monitoring Port: {self.config.get('port', 9999) - 1}")
        print(f"   Bind Address: {self.config.get('host', '0.0.0.0')}")
        
        # Brain configuration
        print(f"\n🧠 Brain Configuration:")
        print(f"   Type: Dynamic Field Brain")
        print(f"   Dimensions: Adaptive (based on robot)")
        print(f"   Persistence: {'Enabled' if self.config.get('enable_persistence', True) else 'Disabled'}")
        print(f"   Logging: {'Enabled' if self.config.get('enable_async_logging', True) else 'Disabled'}")
        
        print("=" * 50)
    
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
        
        print("\n🌍 Dynamic Brain Server Architecture Active")
        print("   - Brains created on-demand per robot type")
        print("   - Dimensions adapt to robot capabilities")
        print("   - Clean separation of concerns")
        print("   - Automatic maintenance and optimization")
        print("\n" + "=" * 50 + "\n")
        
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
    
    args = parser.parse_args()
    
    # Create server
    server = DynamicBrainServer(config_file=args.config)
    
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
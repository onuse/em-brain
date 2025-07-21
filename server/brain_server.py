"""
Minimal Brain TCP Server Entry Point

Simple server that hosts the minimal brain and accepts connections
from any client (Pi Zero brainstems, simulations, mobile apps, etc.)
"""

import sys
import signal
import json
import os
import platform
import psutil
from pathlib import Path
from typing import Optional, Dict, Any

from src.brain_factory import BrainFactory
from src.communication import MinimalTCPServer


class MinimalBrainServer:
    """Main server process for the minimal brain."""
    
    def __init__(self, host: str = None, port: int = None):
        # Load settings first to get network configuration
        self.config = self._load_settings()
        
        # Use network settings from config
        network_config = self.config.get('network', {})
        self.host = host or network_config.get('host', '0.0.0.0')
        self.port = port or network_config.get('port', 9999)
        
        # Setup paths
        self._setup_paths()
        
        # Initialize brain factory with config - use quiet mode to reduce startup spam
        self.brain = BrainFactory(config=self.config, quiet_mode=True)
        self.tcp_server = MinimalTCPServer(self.brain, self.host, self.port)
        self.running = False
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings using adaptive configuration system."""
        try:
            # Try new adaptive configuration system first
            from src.config.adaptive_config import create_adaptive_config
            
            # Check if simplified config exists, otherwise use legacy
            server_dir = Path(__file__).parent
            simple_config = server_dir / "settings_simple.json"
            legacy_config = server_dir / "settings.json"
            
            if simple_config.exists():
                print(f"🔧 Using adaptive configuration from {simple_config}")
                return create_adaptive_config("settings_simple.json")
            elif legacy_config.exists():
                print(f"⚠️  Using legacy configuration from {legacy_config}")
                with open(legacy_config, 'r') as f:
                    return json.load(f)
            else:
                print(f"🔧 No config found, using hardware-adaptive defaults")
                return create_adaptive_config("nonexistent.json")  # Will use built-in defaults
                
        except Exception as e:
            print(f"❌ Configuration system error: {e}")
            print(f"🔄 Falling back to legacy settings.json")
            
            # Fallback to legacy approach
            server_dir = Path(__file__).parent
            settings_file = server_dir / "settings.json"
            
            if settings_file.exists():
                try:
                    with open(settings_file, 'r') as f:
                        return json.load(f)
                except Exception as e2:
                    print(f"⚠️ Warning: Could not load settings.json: {e2}")
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
        
        # Print comprehensive hardware and software stack info
        self._print_system_info()
        
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
    
    def _print_system_info(self):
        """Print comprehensive hardware and software stack information."""
        print("\n" + "="*60)
        print("🖥️  HARDWARE & SOFTWARE STACK INFORMATION")
        print("="*60)
        
        # Operating System Info
        print(f"🐧 Operating System:")
        print(f"   Platform: {platform.system()} {platform.release()}")
        print(f"   Architecture: {platform.machine()}")
        print(f"   Python: {platform.python_version()}")
        
        # Hardware Info
        print(f"\n🔧 Hardware Configuration:")
        print(f"   CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} logical)")
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
        
        # PyTorch Backend Detection
        print(f"\n🧠 PyTorch Backend:")
        try:
            import torch
            print(f"   PyTorch version: {torch.__version__}")
            
            # Detect and report backend priority
            if torch.cuda.is_available():
                device = 'cuda'
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device_count > 0 else 0
                print(f"   🚀 CUDA Backend: AVAILABLE")
                print(f"      Devices: {device_count} GPU(s)")
                print(f"      Primary: {device_name}")
                print(f"      Memory: {device_memory:.1f}GB")
                print(f"      CUDA Version: {torch.version.cuda}")
            elif torch.backends.mps.is_available():
                device = 'mps'
                print(f"   🍎 MPS Backend: AVAILABLE (Apple Silicon)")
                print(f"      Metal Performance Shaders enabled")
                # Estimate shared memory (MPS uses unified memory)
                system_memory = psutil.virtual_memory().total / (1024**3)
                estimated_gpu_memory = min(system_memory * 0.6, 32.0)  # Conservative estimate
                print(f"      Estimated GPU Memory: {estimated_gpu_memory:.1f}GB (unified)")
            else:
                device = 'cpu'
                print(f"   💻 CPU Backend: FALLBACK")
                print(f"      No GPU acceleration available")
            
            print(f"   Selected device: {device}")
            
        except ImportError:
            print(f"   ❌ PyTorch not available")
            device = 'cpu'
        
        # Brain Configuration from settings.json
        print(f"\n⚙️  Brain Configuration:")
        brain_config = self.config.get('brain', {})
        brain_type = brain_config.get('type', 'sparse_goldilocks')
        print(f"   Type: {brain_type}")
        print(f"   Sensory dimensions: {brain_config.get('sensory_dim', 16)}")
        print(f"   Motor dimensions: {brain_config.get('motor_dim', 4)}")
        
        # Hardware adaptation status
        if brain_type != 'field':
            print(f"   Hardware adaptation: enabled")
            target_cycle = brain_config.get('target_cycle_time_ms', 150.0)
            print(f"   Target cycle time: {target_cycle}ms")
        else:
            print(f"   Hardware adaptation: field-optimized")
        
        # Warn about manual tuning requirements
        manual_settings = []
        field_config = brain_config.get('field', {})
        if 'spatial_resolution' in field_config:
            manual_settings.append(f"spatial_resolution={field_config['spatial_resolution']}")
        if 'field_dimensions' in field_config:
            manual_settings.append(f"field_dimensions={field_config['field_dimensions']}")
        if 'learning_rate' in field_config:
            manual_settings.append(f"learning_rate={field_config['learning_rate']}")
        
        if manual_settings:
            print(f"\n⚠️  Manual Hardware Tuning Required:")
            print(f"   The following settings in settings.json may need adjustment for this hardware:")
            for setting in manual_settings:
                print(f"   - {setting}")
            print(f"   Consider implementing automatic hardware-based parameter scaling")
        
        print("="*60 + "\n")
    
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
#!/usr/bin/env python3
"""
Validation Runner with Integrated Server Management

This runner handles the full lifecycle:
1. Start brain server in subprocess
2. Wait for server to be ready
3. Run validation experiments
4. Clean up server process

Usage:
  python3 validation_runner_with_server.py biological_embodied_learning --hours 1
"""

import sys
import os
import time
import signal
import socket
import subprocess
import threading
from pathlib import Path
from typing import Optional

class BrainServerManager:
    """Manages brain server lifecycle for validation experiments."""
    
    def __init__(self, server_port: int = 9999):
        self.server_port = server_port
        self.server_process: Optional[subprocess.Popen] = None
        self.server_ready = False
        
    def start_server(self, timeout: float = 30.0) -> bool:
        """Start brain server and wait for it to be ready."""
        print(f"🚀 Starting brain server on port {self.server_port}...")
        
        try:
            # Start server process
            server_path = Path("server/brain_server.py")
            if not server_path.exists():
                print(f"❌ Server not found: {server_path}")
                return False
            
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path("server")
            )
            
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._check_server_ready():
                    self.server_ready = True
                    print(f"✅ Brain server ready on port {self.server_port}")
                    return True
                
                # Check if process died
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    print(f"❌ Server process died:")
                    print(f"   stdout: {stdout}")
                    print(f"   stderr: {stderr}")
                    return False
                
                time.sleep(0.5)
            
            print(f"❌ Server failed to start within {timeout} seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def _check_server_ready(self) -> bool:
        """Check if server is ready by attempting connection."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                result = sock.connect_ex(('localhost', self.server_port))
                return result == 0
        except:
            return False
    
    def stop_server(self):
        """Stop the brain server gracefully."""
        if self.server_process is None:
            return
        
        print("🛑 Stopping brain server...")
        
        try:
            # Try graceful shutdown first
            self.server_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.server_process.wait(timeout=5.0)
                print("✅ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                print("⚠️ Forcing server shutdown...")
                self.server_process.kill()
                self.server_process.wait()
                print("✅ Server force-stopped")
                
        except Exception as e:
            print(f"⚠️ Error stopping server: {e}")
        
        finally:
            self.server_process = None
            self.server_ready = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()

def run_validation_with_server(experiment_name: str, args: list) -> bool:
    """Run validation experiment with managed server."""
    
    # Map experiment names to paths
    experiment_paths = {
        'biological_embodied_learning': 'validation/embodied_learning/experiments/biological_embodied_learning.py',
        'biological_timescales': 'tests/test_biological_timescales.py'
    }
    
    if experiment_name not in experiment_paths:
        print(f"❌ Unknown experiment: {experiment_name}")
        print(f"Available experiments: {list(experiment_paths.keys())}")
        return False
    
    experiment_path = experiment_paths[experiment_name]
    
    if not Path(experiment_path).exists():
        print(f"❌ Experiment not found: {experiment_path}")
        return False
    
    # Start server and run experiment
    with BrainServerManager() as server:
        if not server.start_server():
            return False
        
        try:
            print(f"🧪 Running validation experiment: {experiment_name}")
            print(f"   Path: {experiment_path}")
            print(f"   Args: {args}")
            
            # Run the experiment
            cmd = [sys.executable, experiment_path] + args
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ Experiment completed successfully")
                return True
            else:
                print(f"❌ Experiment failed with return code {result.returncode}")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️ Experiment interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            return False

def main():
    """Main validation runner with server management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validation Runner with Server Management')
    parser.add_argument('experiment', help='Experiment name')
    parser.add_argument('--hours', type=float, help='Duration in hours')
    parser.add_argument('--session-minutes', type=int, help='Session duration in minutes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args, unknown_args = parser.parse_known_args()
    
    # Build experiment arguments
    experiment_args = unknown_args.copy()
    
    if args.hours:
        experiment_args.extend(['--hours', str(args.hours)])
    
    if args.session_minutes:
        experiment_args.extend(['--session-minutes', str(args.session_minutes)])
    
    if args.seed:
        experiment_args.extend(['--seed', str(args.seed)])
    
    # Run experiment with server management
    success = run_validation_with_server(args.experiment, experiment_args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
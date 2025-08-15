#!/usr/bin/env python3
"""
Simple Brain Loop Test

Test a minimal brain loop without importing the full brain system.
This helps isolate any blocking issues.
"""

import sys
import os
import time
import threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SimpleBrainLoop:
    """Minimal brain loop for testing."""
    
    def __init__(self, cycle_time_ms: float = 50.0):
        self.cycle_time_s = cycle_time_ms / 1000.0
        self.running = False
        self.thread = None
        self.total_cycles = 0
        
        print(f"SimpleBrainLoop initialized (cycle: {cycle_time_ms}ms)")
    
    def start(self):
        """Start the brain loop."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("Brain loop started")
    
    def stop(self):
        """Stop the brain loop."""
        if not self.running:
            return
        
        print("Stopping brain loop...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        print("Brain loop stopped")
    
    def _loop(self):
        """Main loop."""
        print("Loop thread started")
        
        while self.running:
            cycle_start = time.time()
            
            # Simple work
            self.total_cycles += 1
            
            # Sleep for remainder of cycle
            cycle_time = time.time() - cycle_start
            remaining_time = self.cycle_time_s - cycle_time
            if remaining_time > 0:
                time.sleep(remaining_time)
        
        print("Loop thread finished")
    
    def get_stats(self):
        """Get statistics."""
        return {
            'running': self.running,
            'total_cycles': self.total_cycles
        }


def main():
    """Test simple brain loop."""
    print("🧪 Testing Simple Brain Loop")
    print("=" * 40)
    
    # Create and test simple loop
    loop = SimpleBrainLoop(cycle_time_ms=100.0)
    
    print("Starting loop...")
    loop.start()
    
    print("Running for 0.5 seconds...")
    time.sleep(0.5)
    
    stats = loop.get_stats()
    print(f"Stats: {stats}")
    
    print("Stopping loop...")
    loop.stop()
    
    print("✅ Simple brain loop test completed")
    
    if stats['total_cycles'] > 0:
        print(f"   Completed {stats['total_cycles']} cycles")
        return True
    else:
        print("   ❌ No cycles completed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
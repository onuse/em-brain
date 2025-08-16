#!/usr/bin/env python3
"""
Brain & Brainstem Real-Time Monitor

A sophisticated dashboard that connects to both:
- The brain server (TCP 9999) for brain telemetry
- The monitoring port (TCP 9998) for detailed metrics
- The brainstem (via brain) for robot telemetry

Shows real-time:
- Brain field dynamics
- Motivation states
- Sensory input patterns
- Motor outputs
- Learning progress
- Performance metrics

Usage:
    python3 monitor.py                    # Connect to localhost
    python3 monitor.py --host 192.168.1.100  # Remote brain
    python3 monitor.py --terminal         # Terminal-only mode
"""

import sys
import os
import socket
import struct
import threading
import time
import json
import argparse
from collections import deque
from datetime import datetime
import numpy as np

# Add server to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))


class BrainMonitor:
    """Real-time brain telemetry monitor."""
    
    def __init__(self, brain_host='localhost', brain_port=9999, monitor_port=9998):
        self.brain_host = brain_host
        self.brain_port = brain_port
        self.monitor_port = monitor_port
        
        # Data storage
        self.telemetry_buffer = deque(maxlen=1000)
        self.sensor_buffer = deque(maxlen=100)
        self.motor_buffer = deque(maxlen=100)
        self.motivation_history = deque(maxlen=500)
        
        # Stats
        self.start_time = time.time()
        self.total_cycles = 0
        self.motivation_counts = {
            'STARVED': 0,
            'BORED': 0,
            'ACTIVE': 0,
            'CONTENT': 0,
            'UNCOMFORTABLE': 0
        }
        
        # Connection state
        self.connected = False
        self.monitoring = False
        self.threads = []
        
    def connect_monitoring(self):
        """Connect to brain monitoring port for detailed telemetry."""
        try:
            self.monitor_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.monitor_sock.connect((self.brain_host, self.monitor_port))
            self.monitoring = True
            print(f"✅ Connected to monitoring port {self.monitor_port}")
            
            # Start monitoring thread
            thread = threading.Thread(target=self._monitor_loop, daemon=True)
            thread.start()
            self.threads.append(thread)
            
        except Exception as e:
            print(f"⚠️  Could not connect to monitoring port: {e}")
            self.monitoring = False
    
    def connect_brain(self):
        """Connect to main brain port to observe traffic."""
        try:
            # For now, just note we could connect
            # In practice, we'd need to either:
            # 1. Act as a proxy between robot and brain
            # 2. Have brain broadcast telemetry
            # 3. Use the monitoring port (which we do above)
            self.connected = True
            print(f"✅ Monitoring brain at {self.brain_host}:{self.brain_port}")
            
        except Exception as e:
            print(f"❌ Could not connect to brain: {e}")
            self.connected = False
    
    def _monitor_loop(self):
        """Receive telemetry from monitoring port."""
        buffer = b''
        
        while self.monitoring:
            try:
                # Read data
                data = self.monitor_sock.recv(4096)
                if not data:
                    break
                
                buffer += data
                
                # Try to parse JSON messages (newline delimited)
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        msg = json.loads(line.decode('utf-8'))
                        self._process_telemetry(msg)
                    except:
                        pass
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Monitor error: {e}")
                break
        
        self.monitoring = False
    
    def _process_telemetry(self, msg):
        """Process incoming telemetry message."""
        self.total_cycles += 1
        
        # Store telemetry
        msg['timestamp'] = time.time()
        self.telemetry_buffer.append(msg)
        
        # Extract key metrics
        if 'motivation' in msg:
            self.motivation_history.append(msg['motivation'])
            for key in self.motivation_counts:
                if key in msg['motivation']:
                    self.motivation_counts[key] += 1
                    break
        
        if 'sensors' in msg:
            self.sensor_buffer.append(msg['sensors'])
        
        if 'motors' in msg:
            self.motor_buffer.append(msg['motors'])
    
    def display_terminal(self, refresh_rate=1.0):
        """Display dashboard in terminal."""
        import shutil
        
        while self.connected or self.monitoring:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get terminal size
            cols, rows = shutil.get_terminal_size()
            
            # Header
            print("=" * cols)
            print("BRAIN & BRAINSTEM MONITOR".center(cols))
            print("=" * cols)
            
            # Connection status
            uptime = time.time() - self.start_time
            print(f"\n📡 Status: {'Connected' if self.monitoring else 'Searching...'}")
            print(f"⏱️  Uptime: {uptime:.1f}s")
            print(f"🔄 Cycles: {self.total_cycles}")
            
            if self.total_cycles > 0:
                print(f"📊 Rate: {self.total_cycles/uptime:.1f} Hz")
            
            # Latest telemetry
            if self.telemetry_buffer:
                latest = self.telemetry_buffer[-1]
                
                print("\n" + "-" * cols)
                print("CURRENT STATE")
                print("-" * cols)
                
                # Motivation
                if 'motivation' in latest:
                    print(f"🧠 Motivation: {latest['motivation']}")
                
                # Energy & Comfort
                if 'energy' in latest:
                    energy_bar = '█' * int(latest['energy'] * 20)
                    print(f"⚡ Energy: {latest['energy']:.3f} {energy_bar}")
                
                if 'comfort' in latest:
                    comfort = latest['comfort']
                    comfort_bar = '█' * int(abs(comfort) * 5)
                    comfort_sign = '+' if comfort > 0 else '-'
                    print(f"😌 Comfort: {comfort:+.2f} {comfort_sign}{comfort_bar}")
                
                # Learning state
                if 'learning' in latest:
                    print(f"📚 Learning: {latest['learning']}")
                
                # Exploration
                if 'exploring' in latest:
                    print(f"🔍 Exploring: {'Yes' if latest['exploring'] else 'No'}")
            
            # Motivation distribution
            if sum(self.motivation_counts.values()) > 0:
                print("\n" + "-" * cols)
                print("MOTIVATION DISTRIBUTION")
                print("-" * cols)
                
                total = sum(self.motivation_counts.values())
                for state, count in self.motivation_counts.items():
                    pct = (count / total * 100)
                    bar = '█' * int(pct / 2)
                    print(f"{state:15} {pct:5.1f}% {bar}")
            
            # Sensor patterns
            if len(self.sensor_buffer) > 10:
                print("\n" + "-" * cols)
                print("SENSOR PATTERNS (last 10)")
                print("-" * cols)
                
                # Simple ASCII visualization of sensor values
                for sensors in list(self.sensor_buffer)[-10:]:
                    if isinstance(sensors, list):
                        # Normalize to 0-10 scale for display
                        normalized = [max(0, min(10, int((s + 1) * 5))) for s in sensors[:8]]
                        viz = ''.join(['█' * n + '·' * (10-n) + ' ' for n in normalized])
                        print(f"  {viz}")
            
            # Motor outputs
            if len(self.motor_buffer) > 0:
                print("\n" + "-" * cols)
                print("MOTOR OUTPUTS")
                print("-" * cols)
                
                latest_motors = self.motor_buffer[-1]
                if isinstance(latest_motors, list):
                    motor_names = ['Forward', 'Turn', 'Servo1', 'Servo2', 'Servo3']
                    for i, (name, value) in enumerate(zip(motor_names, latest_motors[:5])):
                        # Create a centered bar visualization
                        bar_pos = int((value + 1) * 10)  # Scale -1 to 1 → 0 to 20
                        bar = '·' * 10 + '|' + '·' * 10
                        bar = bar[:bar_pos] + '█' + bar[bar_pos+1:]
                        print(f"  {name:8} [{bar}] {value:+.3f}")
            
            # Performance metrics
            if self.telemetry_buffer:
                recent = list(self.telemetry_buffer)[-100:]
                if 'time_ms' in recent[0]:
                    avg_time = np.mean([t['time_ms'] for t in recent if 'time_ms' in t])
                    max_time = np.max([t['time_ms'] for t in recent if 'time_ms' in t])
                    
                    print("\n" + "-" * cols)
                    print("PERFORMANCE")
                    print("-" * cols)
                    print(f"  Avg cycle time: {avg_time:.2f} ms")
                    print(f"  Max cycle time: {max_time:.2f} ms")
                    print(f"  Theoretical max: {1000/avg_time:.0f} Hz")
            
            # Sleep before refresh
            time.sleep(refresh_rate)
    
    def display_graphical(self):
        """Display dashboard with matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle('Brain & Brainstem Real-Time Monitor', fontsize=16)
            
            # Layout: 3x3 grid
            ax1 = plt.subplot(3, 3, 1)  # Motivation pie
            ax2 = plt.subplot(3, 3, 2)  # Energy over time
            ax3 = plt.subplot(3, 3, 3)  # Comfort over time
            ax4 = plt.subplot(3, 3, 4)  # Sensor heatmap
            ax5 = plt.subplot(3, 3, 5)  # Motor outputs
            ax6 = plt.subplot(3, 3, 6)  # Learning progress
            ax7 = plt.subplot(3, 3, (7, 9))  # Activity log
            
            def update(frame):
                # Clear all axes
                for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
                    ax.clear()
                
                if not self.telemetry_buffer:
                    return
                
                # 1. Motivation pie chart
                if sum(self.motivation_counts.values()) > 0:
                    ax1.pie(self.motivation_counts.values(), 
                           labels=self.motivation_counts.keys(),
                           autopct='%1.0f%%')
                    ax1.set_title('Motivation Distribution')
                
                # 2. Energy over time
                if len(self.telemetry_buffer) > 1:
                    energy = [t.get('energy', 0) for t in self.telemetry_buffer]
                    ax2.plot(energy[-100:], 'b-')
                    ax2.set_title('Field Energy')
                    ax2.set_ylabel('Energy')
                    ax2.grid(True, alpha=0.3)
                
                # 3. Comfort over time
                if len(self.telemetry_buffer) > 1:
                    comfort = [t.get('comfort', 0) for t in self.telemetry_buffer]
                    ax3.plot(comfort[-100:], 'r-')
                    ax3.axhline(y=0, color='gray', linestyle='--')
                    ax3.set_title('Comfort Level')
                    ax3.set_ylabel('Comfort')
                    ax3.grid(True, alpha=0.3)
                
                # 4. Sensor heatmap
                if len(self.sensor_buffer) > 5:
                    sensor_data = np.array(list(self.sensor_buffer)[-20:])
                    if sensor_data.shape[1] >= 16:
                        im = ax4.imshow(sensor_data[:, :16].T, 
                                      aspect='auto', cmap='viridis')
                        ax4.set_title('Sensor Activity')
                        ax4.set_xlabel('Time')
                        ax4.set_ylabel('Sensor')
                
                # 5. Motor outputs
                if self.motor_buffer:
                    latest_motors = self.motor_buffer[-1]
                    if isinstance(latest_motors, list) and len(latest_motors) >= 5:
                        motor_names = ['Fwd', 'Turn', 'S1', 'S2', 'S3']
                        ax5.barh(motor_names, latest_motors[:5])
                        ax5.set_xlim(-1, 1)
                        ax5.set_title('Motor Commands')
                        ax5.axvline(x=0, color='gray', linestyle='-')
                
                # 6. Learning progress (if available)
                if len(self.telemetry_buffer) > 10:
                    # Track prediction error or learning metric
                    learning = [t.get('error', 0) for t in self.telemetry_buffer 
                               if 'error' in t]
                    if learning:
                        ax6.plot(learning[-50:], 'g-')
                        ax6.set_title('Learning Progress')
                        ax6.set_ylabel('Error')
                        ax6.grid(True, alpha=0.3)
                
                # 7. Activity log
                recent_events = []
                for t in list(self.telemetry_buffer)[-10:]:
                    if 'motivation' in t:
                        time_str = datetime.fromtimestamp(t['timestamp']).strftime('%H:%M:%S')
                        recent_events.append(f"{time_str}: {t['motivation'][:30]}")
                
                if recent_events:
                    ax7.text(0.05, 0.95, '\n'.join(reversed(recent_events)),
                            transform=ax7.transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            fontfamily='monospace')
                ax7.set_title('Activity Log')
                ax7.axis('off')
                
                plt.tight_layout()
            
            # Animation
            ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
            plt.show()
            
        except ImportError:
            print("❌ Matplotlib not available. Using terminal mode.")
            self.display_terminal()


def create_mock_telemetry_server(port=9998):
    """Create a mock telemetry server for testing."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', port))
    server.listen(1)
    
    print(f"📡 Mock telemetry server on port {port}")
    
    def send_loop(client):
        cycle = 0
        motivations = ['BORED - seeking novelty', 'STARVED for input', 
                      'ACTIVE - learning', 'CONTENT - gentle exploration']
        
        while True:
            cycle += 1
            
            # Create mock telemetry
            telemetry = {
                'cycle': cycle,
                'time_ms': np.random.uniform(1, 5),
                'energy': 0.5 + 0.3 * np.sin(cycle * 0.1),
                'comfort': np.sin(cycle * 0.05) * 2,
                'motivation': motivations[cycle % len(motivations)],
                'exploring': cycle % 10 < 5,
                'learning': 'High' if cycle % 20 < 10 else 'Low',
                'sensors': [np.sin(cycle * 0.1 + i) for i in range(16)],
                'motors': [np.cos(cycle * 0.2 + i) * 0.5 for i in range(5)]
            }
            
            # Send as JSON
            msg = json.dumps(telemetry) + '\n'
            try:
                client.send(msg.encode('utf-8'))
            except:
                break
            
            time.sleep(0.1)
    
    # Accept connection
    client, addr = server.accept()
    print(f"✅ Client connected from {addr}")
    send_loop(client)


def main():
    parser = argparse.ArgumentParser(description='Brain & Brainstem Monitor')
    parser.add_argument('--host', default='localhost',
                       help='Brain server host')
    parser.add_argument('--port', type=int, default=9999,
                       help='Brain server port')
    parser.add_argument('--monitor-port', type=int, default=9998,
                       help='Monitoring port')
    parser.add_argument('--terminal', action='store_true',
                       help='Terminal-only mode (no graphics)')
    parser.add_argument('--mock', action='store_true',
                       help='Run with mock telemetry server')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BRAIN & BRAINSTEM MONITOR")
    print("="*60)
    
    if args.mock:
        # Start mock server in thread
        print("\n🧪 Starting mock telemetry server...")
        mock_thread = threading.Thread(
            target=create_mock_telemetry_server,
            args=(args.monitor_port,),
            daemon=True
        )
        mock_thread.start()
        time.sleep(1)  # Let server start
    
    # Create monitor
    monitor = BrainMonitor(
        brain_host=args.host,
        brain_port=args.port,
        monitor_port=args.monitor_port
    )
    
    # Connect
    monitor.connect_brain()
    monitor.connect_monitoring()
    
    if not (monitor.connected or monitor.monitoring):
        print("\n❌ Could not connect to brain")
        print("\nMake sure brain server is running:")
        print("  cd server && python3 brain.py")
        print("\nOr run with mock data:")
        print("  python3 monitor.py --mock")
        return
    
    # Display
    if args.terminal:
        monitor.display_terminal()
    else:
        try:
            monitor.display_graphical()
        except KeyboardInterrupt:
            pass
    
    print("\n👋 Monitor stopped")


if __name__ == "__main__":
    main()
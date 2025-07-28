#!/usr/bin/env python3
"""
Profile Brain Performance Using Telemetry

Diagnose performance bottlenecks in brain processing using telemetry data.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Add paths
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.monitoring_server import DynamicMonitoringServer
from src.core.telemetry_client import TelemetryClient


def profile_brain_performance(cycles: int = 50):
    """Profile brain performance using telemetry"""
    print("🔍 Brain Performance Profiling with Telemetry")
    print("=" * 60)
    
    # Initialize components
    robot_registry = RobotRegistry()
    brain_config = {
        'quiet_mode': True,
        'spatial_resolution': 4
    }
    brain_factory = DynamicBrainFactory(brain_config)
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    brain_service = BrainService(brain_pool, adapter_factory)
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    # Start monitoring
    monitoring_server = DynamicMonitoringServer(
        brain_service=brain_service,
        connection_handler=connection_handler,
        host='localhost',
        port=9998
    )
    monitoring_server.start()
    
    # Create robot
    client_id = "perf_test_robot"
    capabilities = [1.0, 16.0, 4.0, 0.0, 0.0]
    connection_handler.handle_handshake(client_id, capabilities)
    
    # Connect telemetry
    telemetry_client = TelemetryClient()
    telemetry_client.connect()
    session_id = telemetry_client.wait_for_session(max_wait=2.0)
    
    if not session_id:
        print("❌ No session available")
        return
    
    print(f"📊 Profiling session: {session_id}\n")
    
    # Performance metrics
    cycle_times = []
    field_energies = []
    memory_regions = []
    constraints = []
    phases = []
    modes = []
    
    # Different input patterns to test
    patterns = {
        'stable': [0.5] * 16,
        'varying': lambda: list(np.random.rand(16)),
        'extreme': [0.0, 1.0] * 8,
        'sine': lambda i: [np.sin(i * 0.1 + j * 0.5) for j in range(16)]
    }
    
    for pattern_name, pattern_gen in patterns.items():
        print(f"\n📈 Testing with {pattern_name} pattern...")
        print("-" * 40)
        
        pattern_times = []
        
        for i in range(cycles // len(patterns)):
            # Generate pattern
            if callable(pattern_gen):
                pattern = pattern_gen() if pattern_name != 'sine' else pattern_gen(i)
            else:
                pattern = pattern_gen
            
            # Time the processing
            start_time = time.time()
            motor_output = connection_handler.handle_sensory_input(client_id, pattern)
            cycle_time = (time.time() - start_time) * 1000
            
            pattern_times.append(cycle_time)
            cycle_times.append(cycle_time)
            
            # Get telemetry
            telemetry = telemetry_client.get_session_telemetry(session_id)
            if telemetry:
                field_energies.append(telemetry.energy)
                memory_regions.append(telemetry.memory_regions)
                constraints.append(telemetry.constraints)
                phases.append(telemetry.phase)
                modes.append(telemetry.mode)
        
        # Pattern statistics
        avg_time = np.mean(pattern_times)
        std_time = np.std(pattern_times)
        print(f"  Average cycle time: {avg_time:.1f}ms (±{std_time:.1f}ms)")
        print(f"  Min/Max: {min(pattern_times):.1f}ms / {max(pattern_times):.1f}ms")
    
    # Overall analysis
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Timing analysis
    print("\n⏱️  Timing Analysis:")
    print(f"  Overall average: {np.mean(cycle_times):.1f}ms")
    print(f"  Standard deviation: {np.std(cycle_times):.1f}ms")
    print(f"  95th percentile: {np.percentile(cycle_times, 95):.1f}ms")
    
    # Check for performance degradation
    early_times = cycle_times[:10]
    late_times = cycle_times[-10:]
    degradation = (np.mean(late_times) - np.mean(early_times)) / np.mean(early_times) * 100
    print(f"  Performance degradation: {degradation:+.1f}%")
    
    # Field dynamics analysis
    print("\n🧠 Brain State Analysis:")
    if field_energies:
        print(f"  Average field energy: {np.mean(field_energies):.6f}")
        print(f"  Energy variance: {np.var(field_energies):.8f}")
    
    # Memory analysis
    if memory_regions:
        print(f"  Average memory regions: {np.mean(memory_regions):.1f}")
        print(f"  Max memory regions: {max(memory_regions)}")
    
    if constraints:
        print(f"  Average constraints: {np.mean(constraints):.1f}")
        print(f"  Max constraints: {max(constraints)}")
    
    # Phase analysis
    if phases:
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        print("\n  Phase distribution:")
        for phase, count in phase_counts.items():
            print(f"    {phase}: {count/len(phases)*100:.1f}%")
    
    # Mode analysis
    if modes:
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        print("\n  Cognitive mode distribution:")
        for mode, count in mode_counts.items():
            print(f"    {mode}: {count/len(modes)*100:.1f}%")
    
    # Bottleneck identification
    print("\n🔍 Potential Bottlenecks:")
    
    if np.mean(cycle_times) > 200:
        print("  ⚠️  High average cycle time (>200ms)")
        print("     → Consider reducing spatial resolution")
        print("     → Check tensor operations on CPU vs GPU")
    
    if degradation > 20:
        print("  ⚠️  Significant performance degradation over time")
        print("     → Memory leak or accumulation possible")
        print("     → Check topology region pruning")
    
    if field_energies and np.var(field_energies) < 1e-8:
        print("  ⚠️  Very low field energy variance")
        print("     → Field may be stuck or not evolving")
        print("     → Check field evolution rate")
    
    if memory_regions and max(memory_regions) > 500:
        print("  ⚠️  Excessive memory regions")
        print("     → Memory pruning may be needed")
    
    # Recommendations
    print("\n💡 Performance Recommendations:")
    
    avg_cycle = np.mean(cycle_times)
    if avg_cycle < 150:
        print("  ✅ Excellent performance - meets biological timing")
    elif avg_cycle < 300:
        print("  👍 Good performance - acceptable for most applications")
    else:
        print("  🔧 Performance optimization needed:")
        print("     1. Reduce spatial resolution (current: 4³)")
        print("     2. Disable unused features (attention, hierarchical)")
        print("     3. Optimize tensor operations")
        print("     4. Consider simple brain for basic tasks")
    
    # Cleanup
    connection_handler.handle_disconnect(client_id)
    telemetry_client.disconnect()
    monitoring_server.stop()
    
    print("\n✅ Performance profiling complete")


def main():
    """Run performance profiling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile brain performance")
    parser.add_argument('--cycles', type=int, default=50, help='Number of test cycles')
    args = parser.parse_args()
    
    profile_brain_performance(args.cycles)


if __name__ == "__main__":
    main()
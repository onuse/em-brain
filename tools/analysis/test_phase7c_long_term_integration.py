#!/usr/bin/env python3
"""
Phase 7c: Long-Term Constraint System Integration Test

Tests constraint propagation patterns over extended sessions to detect
emergent behaviors that only appear during longer operation periods.
"""

import sys
import os
import time
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.brain_factory import MinimalBrain
from server.src.vector_stream.stream_types import StreamType, ConstraintType
from tools.analysis.constraint_analysis_logger import create_constraint_logger


def test_long_term_constraint_integration(duration_seconds: float = 30.0, 
                                        cycles_per_second: float = 10.0):
    """
    Test constraint system integration over an extended period.
    
    Args:
        duration_seconds: How long to run the test
        cycles_per_second: Target processing frequency
    """
    print(f"🕐 Testing Long-Term Constraint Integration ({duration_seconds}s)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration optimized for constraint detection
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': False
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'target_cycle_time_ms': 25.0,  # 40Hz gamma frequency
                'enable_biological_timing': True,
                'enable_parallel_processing': True
            },
            'logging': {
                'log_brain_cycles': False
            }
        }
        
        print("🔧 Creating brain with constraint propagation...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)
        brain.enable_parallel_processing(True)
        
        # Create constraint analysis logger
        session_name = f"long_term_integration_{int(time.time())}"
        constraint_logger = create_constraint_logger(session_name)
        
        # Hook into constraint propagation system for direct event logging
        shared_state = brain.parallel_coordinator.shared_state
        original_propagate = shared_state.propagate_constraint
        
        def logged_propagate_constraint(source_stream, constraint_type, intensity, metadata=None):
            """Wrapper to log constraint propagation events."""
            result = original_propagate(source_stream, constraint_type, intensity, metadata)
            if result:
                # Log successful propagation
                target_streams = []
                for target in StreamType:
                    if target != source_stream:
                        pressures = shared_state.get_constraint_pressures(target)
                        if constraint_type in pressures and pressures[constraint_type] > 0.05:
                            target_streams.append(target.value)
                
                constraint_logger.log_constraint_event(
                    source_stream.value,
                    constraint_type.value,
                    intensity,
                    target_streams,
                    metadata or {}
                )
            return result
        
        # Monkey patch for logging
        shared_state.propagate_constraint = logged_propagate_constraint
        
        print(f"🧠 Running extended session: {duration_seconds}s at {cycles_per_second} Hz")
        print("   Monitoring constraint emergence patterns...")
        
        start_time = time.time()
        cycle_count = 0
        target_cycle_time = 1.0 / cycles_per_second
        
        # Pattern generators for varied input
        def generate_sensory_pattern(cycle: int, complexity_level: float = 1.0):
            """Generate varied sensory input patterns."""
            base_pattern = [0.1 * i for i in range(16)]
            
            # Add periodic variations
            phase = (cycle * 0.1) % (2 * 3.14159)
            variation = [0.2 * complexity_level * (0.5 + 0.5 * (i % 3) * phase) for i in range(16)]
            
            # Combine base + variation
            return [base_pattern[i] + variation[i] for i in range(16)]
        
        complexity_phases = [
            (0.0, 5.0, 0.5),   # Phase 1: Low complexity
            (5.0, 15.0, 1.5),  # Phase 2: High complexity (should trigger constraints)
            (15.0, 25.0, 0.8), # Phase 3: Medium complexity
            (25.0, 30.0, 2.0), # Phase 4: Very high complexity
        ]
        
        try:
            while (time.time() - start_time) < duration_seconds:
                cycle_start = time.time()
                
                # Determine current complexity level
                elapsed = time.time() - start_time
                complexity = 1.0
                for start_phase, end_phase, level in complexity_phases:
                    if start_phase <= elapsed < end_phase:
                        complexity = level
                        break
                
                # Generate input pattern
                sensory_input = generate_sensory_pattern(cycle_count, complexity)
                
                # Process through brain
                motor_output, brain_state = brain.process_sensory_input(sensory_input)
                
                # Log cycle information
                constraint_logger.log_cycle(brain_state, cycle_count)
                
                cycle_count += 1
                
                # Progress indication
                if cycle_count % 50 == 0:
                    progress = (time.time() - start_time) / duration_seconds
                    print(f"   Progress: {progress:.1%} - Cycle {cycle_count} - Complexity: {complexity:.1f}")
                
                # Maintain target cycle time
                cycle_duration = time.time() - cycle_start
                if cycle_duration < target_cycle_time:
                    time.sleep(target_cycle_time - cycle_duration)
        
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
        
        finally:
            # Restore original function
            shared_state.propagate_constraint = original_propagate
            brain.finalize_session()
        
        print(f"\n📊 Session completed: {cycle_count} cycles in {time.time() - start_time:.1f}s")
        
        # Analyze results
        constraint_logger.print_summary()
        
        # Save detailed analysis
        analysis_file = constraint_logger.save_analysis()
        
        # Evaluate success criteria
        analysis = constraint_logger.analyze()
        
        success_criteria = {
            'min_events': 10,
            'min_constraint_types': 2,
            'min_emergent_patterns': 1,
            'min_affected_streams': 2
        }
        
        success = (
            analysis.total_events >= success_criteria['min_events'] and
            len(analysis.constraint_types_seen) >= success_criteria['min_constraint_types'] and
            len(analysis.emergent_patterns) >= success_criteria['min_emergent_patterns'] and
            len(set(constraint_logger.target_counts.keys())) >= success_criteria['min_affected_streams']
        )
        
        print(f"\n🎯 Success Criteria Evaluation:")
        print(f"  Events: {analysis.total_events} >= {success_criteria['min_events']} ✅" if analysis.total_events >= success_criteria['min_events'] else f"  Events: {analysis.total_events} < {success_criteria['min_events']} ❌")
        print(f"  Constraint types: {len(analysis.constraint_types_seen)} >= {success_criteria['min_constraint_types']} ✅" if len(analysis.constraint_types_seen) >= success_criteria['min_constraint_types'] else f"  Constraint types: {len(analysis.constraint_types_seen)} < {success_criteria['min_constraint_types']} ❌")
        print(f"  Emergent patterns: {len(analysis.emergent_patterns)} >= {success_criteria['min_emergent_patterns']} ✅" if len(analysis.emergent_patterns) >= success_criteria['min_emergent_patterns'] else f"  Emergent patterns: {len(analysis.emergent_patterns)} < {success_criteria['min_emergent_patterns']} ❌")
        print(f"  Affected streams: {len(set(constraint_logger.target_counts.keys()))} >= {success_criteria['min_affected_streams']} ✅" if len(set(constraint_logger.target_counts.keys())) >= success_criteria['min_affected_streams'] else f"  Affected streams: {len(set(constraint_logger.target_counts.keys()))} < {success_criteria['min_affected_streams']} ❌")
        
        if success:
            print("🎉 Long-term constraint integration test PASSED!")
        else:
            print("⚠️ Long-term constraint integration test needs longer observation period")
        
        return success, analysis, analysis_file


def test_constraint_pattern_detection(cycles: int = 100):
    """
    Focused test for constraint pattern detection in a shorter session.
    """
    print(f"🔍 Testing Constraint Pattern Detection ({cycles} cycles)")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {'persistent_memory_path': temp_dir, 'enable_persistence': False},
            'brain': {
                'type': 'sparse_goldilocks',
                'enable_biological_timing': True,
                'enable_parallel_processing': True
            },
            'logging': {'log_brain_cycles': False}
        }
        
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)
        brain.enable_parallel_processing(True)
        
        constraint_logger = create_constraint_logger("pattern_detection")
        
        # Create deliberate constraint scenarios
        scenarios = [
            {"name": "High Load", "complexity": 3.0, "cycles": 20},
            {"name": "Resource Scarcity", "complexity": 1.0, "cycles": 15},
            {"name": "Mixed Stress", "complexity": 2.5, "cycles": 25},
            {"name": "Low Activity", "complexity": 0.5, "cycles": 10},
            {"name": "Burst Activity", "complexity": 4.0, "cycles": 30}
        ]
        
        cycle_count = 0
        for scenario in scenarios:
            print(f"📋 Running {scenario['name']} scenario...")
            
            for i in range(scenario['cycles']):
                # Generate scenario-specific input
                complexity = scenario['complexity']
                sensory_input = [0.1 * j * complexity * (1 + 0.3 * (i % 4)) for j in range(16)]
                
                motor_output, brain_state = brain.process_sensory_input(sensory_input)
                constraint_logger.log_cycle(brain_state, cycle_count)
                cycle_count += 1
        
        brain.finalize_session()
        
        # Analyze patterns
        constraint_logger.print_summary()
        analysis = constraint_logger.analyze()
        
        # Check for specific patterns
        pattern_success = len(analysis.emergent_patterns) > 0 and analysis.total_events > 5
        
        return pattern_success, analysis


if __name__ == "__main__":
    print("🚀 Phase 7c: Long-Term Constraint System Integration Tests")
    print("=" * 70)
    
    # Short pattern detection test first
    print("\n🔍 Quick pattern detection test...")
    pattern_success, pattern_analysis = test_constraint_pattern_detection(50)
    
    if pattern_success:
        print("\n✅ Pattern detection working - running extended test...")
        # Full long-term test
        success, analysis, log_file = test_long_term_constraint_integration(15.0, 8.0)
        
        if log_file:
            print(f"\n📁 Detailed analysis saved to: {log_file}")
            print("   This file can be analyzed later to detect longer-term emergent patterns")
    else:
        print("\n⚠️ Basic pattern detection needs investigation before extended testing")
        success = False
    
    print(f"\n🎯 Overall Result: {'✅ SUCCESS' if success else '❌ NEEDS LONGER OBSERVATION'}")
    
    exit(0 if success else 1)
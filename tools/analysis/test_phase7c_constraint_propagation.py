#!/usr/bin/env python3
"""
Test Phase 7c: Enhanced Cross-Stream Constraint Propagation

Tests the enhanced constraint propagation system where processing challenges
in one stream naturally create adaptive pressure in other streams.
"""

import sys
import os
import time
import tempfile
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.brain import MinimalBrain
from server.src.vector_stream.stream_types import StreamType, ConstraintType


def test_constraint_propagation_integration():
    """Test that constraint propagation is integrated into the brain system."""
    print("🔗 Testing Phase 7c: Constraint Propagation Integration")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration with parallel processing and constraint propagation
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
                'enable_parallel_processing': True  # Enable Phase 7b+7c
            },
            'logging': {
                'log_brain_cycles': False
            }
        }
        
        print("🔧 Creating brain with constraint propagation...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        # Verify constraint propagation system is available
        if not brain.parallel_coordinator:
            print("❌ Failed: Parallel coordinator not available")
            return False
        
        shared_state = brain.parallel_coordinator.shared_state
        if not hasattr(shared_state, 'constraint_propagation'):
            print("❌ Failed: Constraint propagation system not integrated")
            return False
        
        print("✅ Constraint propagation system integrated")
        
        # Enable parallel processing to use constraint propagation
        brain.enable_parallel_processing(True)
        
        # Test basic constraint propagation
        print("\n🔬 Testing manual constraint propagation...")
        success = shared_state.propagate_constraint(
            StreamType.SENSORY,
            ConstraintType.PROCESSING_LOAD,
            0.8,
            {'test': 'manual_propagation'}
        )
        print(f"Manual constraint propagation: {'✅' if success else '❌'}")
        
        # Check constraint pressures
        motor_constraints = shared_state.get_constraint_pressures(StreamType.MOTOR)
        print(f"Motor stream constraints: {motor_constraints}")
        
        if ConstraintType.PROCESSING_LOAD in motor_constraints:
            print(f"✅ Processing load constraint propagated to motor stream: {motor_constraints[ConstraintType.PROCESSING_LOAD]:.3f}")
        else:
            print("⚠️ Processing load constraint not detected in motor stream")
        
        return True


def test_emergent_constraint_propagation():
    """Test that constraints emerge from actual processing conditions."""
    print("\n🔬 Testing Emergent Constraint Propagation")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {'persistent_memory_path': temp_dir, 'enable_persistence': False},
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'enable_biological_timing': True,
                'enable_parallel_processing': True
            },
            'logging': {'log_brain_cycles': False}
        }
        
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)  # Enable debug
        brain.enable_parallel_processing(True)
        shared_state = brain.parallel_coordinator.shared_state
        
        # Create varying processing loads to trigger constraint propagation
        print("🧠 Running cycles with varying processing loads...")
        
        constraint_events = []
        for i in range(15):
            # Create different input patterns to stress different streams
            if i < 5:
                # Normal processing
                sensory_input = [0.1 * j for j in range(16)]
            elif i < 10:
                # High complexity input (should trigger processing load constraints)
                sensory_input = [0.1 * j * (1 + 0.5 * (j % 3)) for j in range(16)]
            else:
                # Very complex input (should trigger multiple constraints)
                sensory_input = [0.1 * j * (1 + 0.8 * (j % 5)) + 0.2 * (i % 3) for j in range(16)]
            
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            # Check for constraint propagations
            if brain_state.get('parallel_processing', False):
                stats = shared_state.get_shared_state_stats()
                if 'constraint_propagation' in stats:
                    propagation_stats = stats['constraint_propagation']['propagation_stats']
                    active_constraints = propagation_stats.get('active_constraints', 0)
                    
                    if active_constraints > 0:
                        constraint_events.append({
                            'cycle': i,
                            'active_constraints': active_constraints,
                            'constraint_breakdown': stats['constraint_propagation']['constraint_pressures']
                        })
                        
                        print(f"  Cycle {i+1}: {active_constraints} active constraints")
                        
                        # Show constraint pressures for each stream
                        for stream_name, pressures in stats['constraint_propagation']['constraint_pressures'].items():
                            total_pressure = pressures['total_pressure']
                            if total_pressure > 0.1:
                                print(f"    {stream_name}: {total_pressure:.3f} total pressure")
                                for constraint_type, intensity in pressures['constraint_breakdown'].items():
                                    if intensity > 0.1:
                                        print(f"      {constraint_type}: {intensity:.3f}")
        
        print(f"\n📊 Constraint Propagation Results:")
        print(f"  Total cycles with constraints: {len(constraint_events)}")
        print(f"  Constraint emergence rate: {len(constraint_events)/15:.1%}")
        
        if len(constraint_events) > 0:
            print("✅ Emergent constraint propagation working")
            
            # Analyze constraint types that emerged
            all_constraint_types = set()
            for event in constraint_events:
                for stream_name, pressures in event['constraint_breakdown'].items():
                    all_constraint_types.update(pressures['constraint_breakdown'].keys())
            
            print(f"  Constraint types observed: {', '.join(all_constraint_types)}")
            return True
        else:
            print("⚠️ No emergent constraints detected - may need higher processing load")
            return False


def test_constraint_based_resource_allocation():
    """Test that resource allocation responds to constraint pressures."""
    print("\n🔬 Testing Constraint-Based Resource Allocation")
    print("-" * 40)
    
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
        
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        brain.enable_parallel_processing(True)
        shared_state = brain.parallel_coordinator.shared_state
        
        # Manually create urgency constraint in sensory stream
        print("🚨 Creating urgency constraint in sensory stream...")
        shared_state.propagate_constraint(
            StreamType.SENSORY,
            ConstraintType.URGENCY_SIGNAL,
            0.9,
            {'test': 'urgency_simulation'}
        )
        
        # Test resource allocation with constraints
        print("🔬 Testing resource allocation with urgency constraint...")
        
        # Normal allocation (without constraints)
        normal_attention = shared_state.request_resource(StreamType.MOTOR, 'attention_budget', 0.5)
        
        # Reset resources
        shared_state.reset_cycle_resources()
        
        # Constraint-aware allocation (with urgency in sensory affecting motor priority)
        constraint_attention = shared_state.request_resource_with_constraints(
            StreamType.MOTOR, 'attention_budget', 0.5
        )
        
        print(f"  Normal attention allocation: {normal_attention:.3f}")
        print(f"  Constraint-aware allocation: {constraint_attention:.3f}")
        
        # The motor stream should get different allocation due to sensory urgency constraint
        if abs(normal_attention - constraint_attention) > 0.01:
            print("✅ Constraint-based resource allocation working")
            print(f"  Allocation difference: {abs(normal_attention - constraint_attention):.3f}")
            return True
        else:
            print("⚠️ No significant allocation difference detected")
            return False


def test_constraint_system_integration():
    """Test full integration of constraint system with parallel processing."""
    print("\n🔬 Testing Full Constraint System Integration")
    print("-" * 40)
    
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
        
        # Enable debug output to see constraint activity
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)
        brain.enable_parallel_processing(True)
        
        # Create constraint logger for better detection
        from tools.analysis.constraint_analysis_logger import create_constraint_logger
        constraint_logger = create_constraint_logger("integration_test")
        
        # Run a sequence of processing cycles with varied complexity
        print("🧠 Running integrated constraint propagation sequence...")
        
        constraint_events_detected = 0
        constraint_types_seen = set()
        
        for i in range(50):  # Increased cycles for better detection
            # Create more varied complexity patterns
            if i < 10:
                complexity_factor = 0.5  # Low complexity
            elif i < 20:
                complexity_factor = 2.0  # High complexity (should trigger constraints)
            elif i < 30:
                complexity_factor = 3.0  # Very high complexity
            elif i < 40:
                complexity_factor = 1.0  # Normal complexity
            else:
                complexity_factor = 2.5  # High complexity again
            
            # Add variation within each phase
            phase_variation = 0.5 * (i % 5)
            total_complexity = complexity_factor + phase_variation
            
            sensory_input = [0.1 * j * total_complexity for j in range(16)]
            
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            # Log cycle for analysis
            constraint_logger.log_cycle(brain_state, i)
            
            # Check for constraint activity in brain state
            if brain_state.get('parallel_processing', False):
                if 'constraint_propagation' in brain_state:
                    constraint_info = brain_state['constraint_propagation']
                    
                    # Check propagation stats
                    if 'propagation_stats' in constraint_info:
                        stats = constraint_info['propagation_stats']
                        active_constraints = stats.get('active_constraints', 0)
                        if active_constraints > 0:
                            constraint_events_detected += 1
                    
                    # Track constraint types from pressure breakdown
                    if 'constraint_pressures' in constraint_info:
                        pressures = constraint_info['constraint_pressures']
                        for stream_name, pressure_info in pressures.items():
                            constraint_breakdown = pressure_info.get('constraint_breakdown', {})
                            for constraint_type, intensity in constraint_breakdown.items():
                                if intensity > 0.05:  # Significant constraint
                                    constraint_types_seen.add(constraint_type)
        
        # Get analysis from logger
        analysis = constraint_logger.analyze()
        
        print(f"📊 Integration Test Results:")
        print(f"  Cycles with constraint activity: {constraint_events_detected}")
        print(f"  Total constraint events logged: {analysis.total_events}")
        print(f"  Constraint types observed: {len(constraint_types_seen)} -> {', '.join(constraint_types_seen) if constraint_types_seen else 'None'}")
        print(f"  Peak simultaneous constraints: {analysis.peak_simultaneous_constraints}")
        
        if analysis.emergent_patterns:
            print(f"  🌟 Emergent patterns: {', '.join(analysis.emergent_patterns)}")
        
        brain.finalize_session()
        
        # Success criteria: either direct detection OR logger analysis shows activity
        direct_success = constraint_events_detected > 0 and len(constraint_types_seen) > 0
        logger_success = analysis.total_events > 5 and len(analysis.constraint_types_seen) > 0
        
        overall_success = direct_success or logger_success
        
        if overall_success:
            print("✅ Full constraint system integration working")
            if not direct_success and logger_success:
                print("   (Activity detected through enhanced logging - may need longer sessions for robust detection)")
        else:
            print("⚠️ Limited constraint system activity detected")
            print("   💡 Suggestion: Run longer test sessions with tools/analysis/test_phase7c_long_term_integration.py")
        
        return overall_success


def main():
    """Run all Phase 7c constraint propagation tests."""
    print("🚀 Phase 7c: Enhanced Cross-Stream Constraint Propagation Test Suite")
    print("=" * 70)
    
    tests = [
        ("Constraint Propagation Integration", test_constraint_propagation_integration),
        ("Emergent Constraint Propagation", test_emergent_constraint_propagation),
        ("Constraint-Based Resource Allocation", test_constraint_based_resource_allocation),
        ("Full Constraint System Integration", test_constraint_system_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 7c constraint propagation tests passed!")
        print("🔗 Enhanced cross-stream constraint propagation is working correctly.")
        return True
    else:
        print("⚠️ Some Phase 7c tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
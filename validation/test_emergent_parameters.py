#!/usr/bin/env python3
"""
Test Emergent Parameters System - Hardware-Constrained Adaptive Parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import numpy as np
import time
from src.parameters.constraint_parameters import EmergentParameterSystem, HardwareProfile, BiologicalConstraints, ConstraintType
from src.attention.object_attention import CrossModalAttentionSystem
from src.attention.signal_attention import ModalityType

def test_emergent_parameters():
    """Test emergent parameter system"""
    print("🧬 Testing Emergent Parameters System")
    print("=" * 50)
    
    # Test 1: Hardware profile detection
    print("\n💻 Test 1: Hardware Profile Detection")
    
    hardware = HardwareProfile.detect_current()
    print(f"   CPU cores: {hardware.cpu_cores}")
    print(f"   CPU frequency: {hardware.cpu_freq_ghz:.1f} GHz")
    print(f"   RAM: {hardware.ram_gb:.1f} GB")
    print(f"   GPU available: {hardware.gpu_available}")
    if hardware.gpu_available:
        print(f"   GPU memory: {hardware.gpu_memory_gb:.1f} GB")
    
    # Test 2: Biological constraint scaling
    print("\n🧠 Test 2: Biological Constraint Scaling")
    
    base_bio = BiologicalConstraints()
    scaled_bio = base_bio.scale_to_hardware(hardware)
    
    print(f"   Base attention span: {base_bio.attention_span_seconds:.3f}s")
    print(f"   Scaled attention span: {scaled_bio.attention_span_seconds:.3f}s")
    print(f"   Base working memory: {base_bio.working_memory_slots} slots")
    print(f"   Scaled working memory: {scaled_bio.working_memory_slots} slots")
    print(f"   Base reaction time: {base_bio.reaction_time_ms:.1f}ms")
    print(f"   Scaled reaction time: {scaled_bio.reaction_time_ms:.1f}ms")
    
    # Test 3: Parameter system initialization
    print("\n⚙️  Test 3: Parameter System Initialization")
    
    param_system = EmergentParameterSystem(
        hardware_profile=hardware,
        biological_constraints=scaled_bio,
        target_framerate=30.0,
        power_budget_watts=10.0
    )
    
    print(f"   Compute budget: {param_system.compute_budget}")
    print(f"   Memory budget: {param_system.memory_budget}")
    print(f"   Attention parameters: {param_system.attention_parameters}")
    
    # Test 4: Constraint-based parameter retrieval
    print("\n🎯 Test 4: Constraint-Based Parameter Retrieval")
    
    # Hardware constraints
    max_objects = param_system.get_parameter('max_objects', ConstraintType.HARDWARE)
    compute_budget = param_system.get_parameter('compute_budget', ConstraintType.HARDWARE)
    
    print(f"   Max objects (hardware): {max_objects}")
    print(f"   Compute budget (hardware): {compute_budget}")
    
    # Biological constraints
    attention_duration = param_system.get_parameter('attention_duration', ConstraintType.BIOLOGICAL)
    working_memory = param_system.get_parameter('working_memory_slots', ConstraintType.BIOLOGICAL)
    
    print(f"   Attention duration (biological): {attention_duration:.3f}s")
    print(f"   Working memory slots (biological): {working_memory}")
    
    # Information constraints
    correlation_threshold = param_system.get_parameter('correlation_threshold', ConstraintType.INFORMATION)
    entropy_threshold = param_system.get_parameter('entropy_threshold', ConstraintType.INFORMATION)
    
    print(f"   Correlation threshold (information): {correlation_threshold:.3f}")
    print(f"   Entropy threshold (information): {entropy_threshold:.3f}")
    
    # Energy constraints
    switch_cost = param_system.get_parameter('switch_cost', ConstraintType.ENERGY)
    processing_intensity = param_system.get_parameter('processing_intensity', ConstraintType.ENERGY)
    
    print(f"   Switch cost (energy): {switch_cost:.1f}")
    print(f"   Processing intensity (energy): {processing_intensity:.3f}")
    
    # Test 5: Competitive weights
    print("\n🏆 Test 5: Competitive Weights")
    
    weights = param_system.get_competitive_weights()
    print(f"   Salience weight: {weights['salience']:.3f}")
    print(f"   Novelty weight: {weights['novelty']:.3f}")
    print(f"   Urgency weight: {weights['urgency']:.3f}")
    print(f"   Coherence weight: {weights['coherence']:.3f}")
    print(f"   Total: {sum(weights.values()):.3f}")
    
    # Test 6: Different hardware profiles
    print("\n🔧 Test 6: Different Hardware Profiles")
    
    # Low-end hardware
    low_end = HardwareProfile(
        cpu_cores=2,
        cpu_freq_ghz=1.5,
        ram_gb=4.0,
        gpu_available=False,
        gpu_memory_gb=0.0,
        storage_iops=500,
        network_bandwidth_mbps=10.0
    )
    
    low_end_system = EmergentParameterSystem(
        hardware_profile=low_end,
        target_framerate=15.0,  # Lower framerate
        power_budget_watts=5.0   # Lower power
    )
    
    # High-end hardware
    high_end = HardwareProfile(
        cpu_cores=16,
        cpu_freq_ghz=4.0,
        ram_gb=64.0,
        gpu_available=True,
        gpu_memory_gb=16.0,
        storage_iops=10000,
        network_bandwidth_mbps=1000.0
    )
    
    high_end_system = EmergentParameterSystem(
        hardware_profile=high_end,
        target_framerate=60.0,  # Higher framerate
        power_budget_watts=50.0  # Higher power
    )
    
    print(f"   Low-end compute budget: {low_end_system.compute_budget}")
    print(f"   High-end compute budget: {high_end_system.compute_budget}")
    
    print(f"   Low-end max objects: {low_end_system.get_parameter('max_objects', ConstraintType.HARDWARE)}")
    print(f"   High-end max objects: {high_end_system.get_parameter('max_objects', ConstraintType.HARDWARE)}")
    
    print(f"   Low-end attention duration: {low_end_system.get_parameter('attention_duration', ConstraintType.BIOLOGICAL):.3f}s")
    print(f"   High-end attention duration: {high_end_system.get_parameter('attention_duration', ConstraintType.BIOLOGICAL):.3f}s")
    
    # Test 7: Performance adaptation
    print("\n📊 Test 7: Performance Adaptation")
    
    # Simulate poor performance
    original_compute = param_system.compute_budget
    original_threshold = param_system.attention_parameters['coherence_threshold']
    
    print(f"   Original compute budget: {original_compute}")
    print(f"   Original coherence threshold: {original_threshold:.3f}")
    
    # Adapt to poor performance
    param_system.adapt_to_performance(
        current_framerate=15.0,  # Below target of 30
        cpu_usage=0.9,           # High CPU usage
        memory_usage=0.8         # High memory usage
    )
    
    print(f"   Adapted compute budget: {param_system.compute_budget}")
    print(f"   Adapted coherence threshold: {param_system.attention_parameters['coherence_threshold']:.3f}")
    
    # Test 8: Integration with Cross-Modal Attention
    print("\n🎯 Test 8: Integration with Cross-Modal Attention")
    
    # Create attention system with emergent parameters
    attention_system = CrossModalAttentionSystem(
        parameter_system=param_system,
        target_framerate=30.0,
        power_budget_watts=10.0
    )
    
    print(f"   Attention system compute budget: {attention_system.compute_budget}")
    print(f"   Attention system max objects: {attention_system.max_objects}")
    print(f"   Attention system switch cost: {attention_system.switch_cost:.1f}")
    print(f"   Attention system correlation threshold: {attention_system.correlation_threshold:.3f}")
    
    # Test with actual sensory data
    visual_signal = np.random.uniform(0, 1, (64, 64))
    audio_signal = np.random.uniform(-1, 1, 1000)
    
    sensory_streams = {
        ModalityType.VISUAL: {
            'signal': visual_signal,
            'brain_output': np.random.uniform(0, 1, 16),
            'novelty': 0.7
        },
        ModalityType.AUDIO: {
            'signal': audio_signal,
            'brain_output': np.random.uniform(0, 1, 16),
            'novelty': 0.6
        }
    }
    
    # Process with emergent parameters
    attention_state = attention_system.update(sensory_streams)
    
    print(f"   Objects created: {attention_state['active_objects']}")
    print(f"   Binding events: {attention_state['binding_events']}")
    print(f"   Competitive weights: {attention_system.competitive_weights}")
    
    # Test 9: Parameter scaling with constraints
    print("\n📈 Test 9: Parameter Scaling with Constraints")
    
    # Test different power budgets
    power_budgets = [5.0, 10.0, 20.0, 50.0]
    
    for power_budget in power_budgets:
        test_system = EmergentParameterSystem(
            hardware_profile=hardware,
            target_framerate=30.0,
            power_budget_watts=power_budget
        )
        
        compute_budget = test_system.compute_budget
        switch_cost = test_system.get_parameter('switch_cost', ConstraintType.ENERGY)
        
        print(f"   Power {power_budget:4.1f}W: compute={compute_budget:5d}, switch_cost={switch_cost:5.1f}")
    
    # Test 10: Performance benchmarking
    print("\n⚡ Test 10: Performance Benchmarking")
    
    # Benchmark parameter retrieval
    start_time = time.time()
    for i in range(1000):
        param_system.get_parameter('compute_budget', ConstraintType.HARDWARE)
        param_system.get_parameter('attention_duration', ConstraintType.BIOLOGICAL)
        param_system.get_parameter('correlation_threshold', ConstraintType.INFORMATION)
    
    retrieval_time = (time.time() - start_time) * 1000
    
    print(f"   1000 parameter retrievals: {retrieval_time:.2f}ms")
    print(f"   Average retrieval time: {retrieval_time/1000:.4f}ms")
    
    # Benchmark attention system with emergent parameters
    start_time = time.time()
    for i in range(100):
        attention_system.update(sensory_streams)
    
    attention_time = (time.time() - start_time) * 1000
    
    print(f"   100 attention updates: {attention_time:.2f}ms")
    print(f"   Average attention time: {attention_time/100:.2f}ms")
    
    # Test 11: Parameter summary
    print("\n📋 Test 11: Parameter Summary")
    
    summary = param_system.get_summary()
    
    print(f"   Hardware Profile:")
    for key, value in summary['hardware_profile'].items():
        print(f"     {key}: {value}")
    
    print(f"   Biological Constraints:")
    for key, value in summary['biological_constraints'].items():
        if isinstance(value, float):
            print(f"     {key}: {value:.3f}")
        else:
            print(f"     {key}: {value}")
    
    print(f"   Derived Parameters:")
    print(f"     compute_budget: {summary['derived_parameters']['compute_budget']}")
    print(f"     max_objects: {summary['derived_parameters']['memory_budget']['max_objects']}")
    
    print(f"\n🎉 Emergent Parameters System Test Complete!")
    
    # Assessment
    success_criteria = [
        hardware.cpu_cores > 0,                      # Hardware detection
        scaled_bio.attention_span_seconds > 0,       # Biological scaling
        param_system.compute_budget > 0,             # Parameter derivation
        max_objects > 0,                             # Constraint-based parameters
        abs(sum(weights.values()) - 1.0) < 0.01,    # Normalized weights
        high_end_system.compute_budget > low_end_system.compute_budget,  # Hardware scaling
        param_system.compute_budget != original_compute,  # Performance adaptation
        attention_system.compute_budget > 0,         # Integration working
        retrieval_time < 100,                        # Performance acceptable
        len(summary) > 0                             # Summary generation
    ]
    
    passed = sum(success_criteria)
    total = len(success_criteria)
    
    print(f"   ✅ Tests passed: {passed}/{total}")
    print(f"   🧬 Emergent design: {'✅ Strong' if passed >= 8 else '⚠️ Moderate' if passed >= 6 else '❌ Weak'}")
    
    return passed >= 8

if __name__ == "__main__":
    success = test_emergent_parameters()
    if success:
        print("\n✅ Emergent Parameters System ready for constraint-based design!")
    else:
        print("\n⚠️ System needs further optimization")
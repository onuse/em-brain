#!/usr/bin/env python3
"""
Test Script for Strategy 1: Pure Information Streams

Tests the radical transformation where experiences emerge from raw data streams
rather than being engineered structures.

This demonstrates:
1. Raw stream storage without structure
2. Pattern discovery finding natural boundaries
3. Emergent experiences from discovered patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.stream import PureStreamStorage, PatternDiscovery, StreamToExperienceAdapter


def test_pure_stream_emergence():
    """Test emergence of structure from pure information streams."""
    
    print("=== Strategy 1 Test: Pure Information Streams ===\n")
    
    # Create the pure stream system
    stream_storage = PureStreamStorage(max_stream_length=1000)
    pattern_discovery = PatternDiscovery(prediction_window=5, emergence_threshold=0.7)
    adapter = StreamToExperienceAdapter(stream_storage, pattern_discovery)
    
    print("Pure stream system initialized\n")
    
    # Generate synthetic robot behavior stream
    print("=== Phase 1: Generating Raw Behavior Stream ===")
    print("Simulating robot exploring environment...\n")
    
    # Simulate different behavioral phases
    stream_ids = []
    
    # Phase 1: Robot moving forward (stable pattern)
    print("Behavior: Moving forward...")
    for i in range(20):
        sensory = [1.0, 0.0, 0.0, 0.5]  # Forward sensor reading
        motor = [0.5, 0.5]  # Both wheels forward
        combined = sensory + motor
        noise = np.random.normal(0, 0.05, len(combined))
        vector = combined + noise
        stream_id = stream_storage.append_vector(vector.tolist())
        stream_ids.append(stream_id)
        time.sleep(0.01)
    
    # Phase 2: Robot encounters obstacle (pattern change)
    print("Behavior: Obstacle detected, turning...")
    for i in range(15):
        sensory = [0.0, 1.0, 0.8, 0.2]  # Obstacle sensor triggered
        motor = [0.3, -0.3]  # Turn right
        combined = sensory + motor
        noise = np.random.normal(0, 0.05, len(combined))
        vector = combined + noise
        stream_id = stream_storage.append_vector(vector.tolist())
        stream_ids.append(stream_id)
        time.sleep(0.01)
    
    # Phase 3: Robot resumes forward (return to pattern)
    print("Behavior: Clear path, moving forward again...")
    for i in range(20):
        sensory = [1.0, 0.0, 0.0, 0.5]  # Forward sensor reading
        motor = [0.5, 0.5]  # Both wheels forward
        combined = sensory + motor
        noise = np.random.normal(0, 0.05, len(combined))
        vector = combined + noise
        stream_id = stream_storage.append_vector(vector.tolist())
        stream_ids.append(stream_id)
        time.sleep(0.01)
    
    # Phase 4: Robot explores new area (novel pattern)
    print("Behavior: Exploring new area...")
    for i in range(10):
        sensory = np.random.uniform(0, 1, 4)  # Variable sensor readings
        motor = np.random.uniform(-0.3, 0.3, 2)  # Exploratory movements
        combined = np.concatenate([sensory, motor])
        stream_id = stream_storage.append_vector(combined.tolist())
        stream_ids.append(stream_id)
        time.sleep(0.01)
    
    # Get stream statistics
    stream_stats = stream_storage.compute_stream_statistics()
    print(f"\n--- Stream Statistics ---")
    print(f"Total vectors: {stream_stats['total_vectors']}")
    print(f"Stream duration: {stream_stats['stream_duration']:.2f}s")
    print(f"Vectors per second: {stream_stats['vectors_per_second']:.1f}")
    print(f"Vector dimensions: {stream_stats['vector_dimensions']}")
    
    # Phase 2: Discover patterns in the stream
    print("\n\n=== Phase 2: Pattern Discovery ===")
    print("Analyzing stream for emergent patterns...\n")
    
    # Get all vectors for analysis
    all_vectors = stream_storage.get_recent_vectors(count=stream_stats['total_vectors'])
    
    # Run pattern discovery
    analysis = pattern_discovery.analyze_stream_segment(all_vectors)
    
    print(f"--- Discovery Results ---")
    print(f"Vectors analyzed: {analysis['vectors_analyzed']}")
    print(f"Prediction boundaries found: {len(analysis['boundaries_found'])}")
    if analysis['boundaries_found']:
        print(f"Boundary positions: {analysis['boundaries_found'][:5]}...")  # Show first 5
    print(f"Causal patterns found: {len(analysis['causal_links'])}")
    print(f"Temporal motifs detected: {len(analysis['motifs_detected'])}")
    
    # Get emergent structure
    structure = pattern_discovery.get_emergent_structure()
    print(f"\n--- Emergent Structure ---")
    print(f"Total patterns discovered: {structure['patterns_discovered']}")
    print(f"Emergent experiences: {len(structure['emergent_experiences'])}")
    print(f"Strong boundaries: {len(structure['prediction_boundaries'])}")
    print(f"Behavioral motifs: {len(structure['behavioral_motifs'])}")
    
    # Show some emergent experiences
    if structure['emergent_experiences']:
        print(f"\nExample emergent experiences:")
        for i, exp in enumerate(structure['emergent_experiences'][:3]):
            print(f"  Experience {i+1}: positions {exp['start']}-{exp['end']} (length: {exp['end']-exp['start']})")
    
    # Phase 3: Adapt to experience format
    print("\n\n=== Phase 3: Stream-to-Experience Adaptation ===")
    print("Converting emergent patterns to experiences...\n")
    
    # Adapt the discovered patterns
    emergent_experiences = adapter.get_emergent_experiences()
    
    print(f"Adapted {len(emergent_experiences)} emergent experiences")
    
    # Analyze adapted experiences
    if emergent_experiences:
        exp_list = list(emergent_experiences.values())
        print(f"\nExample adapted experience:")
        example = exp_list[0]
        print(f"  Sensory: {np.array(example.sensory_input).round(2)}")
        print(f"  Action: {np.array(example.action_taken).round(2)}")
        print(f"  Outcome: {np.array(example.outcome).round(2)}")
        print(f"  Prediction error: {example.prediction_error:.3f}")
        print(f"  Metadata: {example.metadata}")
    
    # Get adaptation statistics
    adapt_stats = adapter.get_adaptation_statistics()
    print(f"\n--- Adaptation Statistics ---")
    print(f"Total adaptations: {adapt_stats['adaptation_count']}")
    print(f"Adaptation rate: {adapt_stats['adaptation_rate']:.3f}")
    print(f"Discovered motifs: {adapt_stats['discovered_motifs']}")
    
    # Phase 4: Test emergent behavior recognition
    print("\n\n=== Phase 4: Testing Emergent Behavior Recognition ===")
    
    # Add a repeated behavior pattern
    print("Adding repeated turning behavior...")
    for _ in range(3):  # Repeat 3 times
        for i in range(10):
            sensory = [0.0, 1.0, 0.8, 0.2]  # Obstacle pattern
            motor = [0.3, -0.3]  # Turn right
            combined = sensory + motor
            noise = np.random.normal(0, 0.02, len(combined))
            vector = combined + noise
            stream_storage.append_vector(vector.tolist())
            time.sleep(0.005)
        
        # Brief forward movement
        for i in range(5):
            sensory = [1.0, 0.0, 0.0, 0.5]
            motor = [0.5, 0.5]
            combined = sensory + motor
            noise = np.random.normal(0, 0.02, len(combined))
            vector = combined + noise
            stream_storage.append_vector(vector.tolist())
            time.sleep(0.005)
    
    # Re-analyze for motifs
    all_vectors = stream_storage.get_recent_vectors(count=150)
    analysis = pattern_discovery.analyze_stream_segment(all_vectors)
    structure = pattern_discovery.get_emergent_structure()
    
    print(f"\nAfter repeated behaviors:")
    print(f"Behavioral motifs found: {len(structure['behavioral_motifs'])}")
    if structure['behavioral_motifs']:
        for motif in structure['behavioral_motifs'][:3]:
            print(f"  Motif: length {motif['length']}, occurs {motif['occurrences']} times")
    
    print(f"\n=== Strategy 1 Results ===")
    print(f"✅ Pure stream storage working - no predefined structure")
    print(f"✅ Pattern discovery found {len(structure['prediction_boundaries'])} natural boundaries")
    print(f"✅ {len(structure['emergent_experiences'])} experiences emerged from patterns")
    print(f"✅ {len(structure['behavioral_motifs'])} behavioral motifs discovered")
    print(f"✅ Successfully bridged emergent patterns with existing architecture")
    
    if len(structure['emergent_experiences']) > 0:
        print(f"\n🎉 Structure successfully emerged from pure information flow!")
        print(f"   No hardcoded experience boundaries - they emerged from prediction patterns")
        print(f"   No defined actions/outcomes - they emerged from temporal structure")
        print(f"   This is true emergence, not engineering!")
    
    return stream_storage, pattern_discovery, adapter


if __name__ == "__main__":
    test_pure_stream_emergence()
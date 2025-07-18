#!/usr/bin/env python3
"""
Test the prediction engine.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from src.prediction import PredictionEngine
    from src.experience import Experience, ExperienceStorage
    from src.similarity import SimilarityEngine
    from src.activation import ActivationDynamics
    
    print("🔮 Testing prediction engine...")
    
    # Initialize systems
    prediction_engine = PredictionEngine()
    similarity_engine = SimilarityEngine(use_gpu=False)
    activation_dynamics = ActivationDynamics()
    storage = ExperienceStorage()
    
    # Add some experiences
    for i in range(5):
        exp = Experience(
            sensory_input=[float(i), float(i+1), 0.0],
            action_taken=[0.5, 0.5],
            outcome=[float(i+0.5), float(i+1.5), 0.0],
            prediction_error=0.2,
            timestamp=123456.0 + i
        )
        storage.add_experience(exp)
    
    # Test prediction
    context = [2.0, 3.0, 0.0]
    action, confidence, details = prediction_engine.predict_action(
        context, similarity_engine, activation_dynamics, 
        storage._experiences, action_dimensions=2
    )
    
    print(f"✅ Prediction engine works!")
    print(f"   Context: {context}")
    print(f"   Predicted action: {action}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Method: {details['method']}")
    
except Exception as e:
    print(f"❌ Prediction engine failed: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python3
"""
Micro-Experiment Runner

Runs targeted micro-experiments to validate brain assumptions before
committing to long experimental runs.

Usage:
  python3 validation/micro_experiment_runner.py core
  python3 validation/micro_experiment_runner.py all
  python3 validation/micro_experiment_runner.py --list
"""

import sys
import os
import argparse
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))

def run_core_experiments():
    """Run core assumption experiments."""
    print("🔬 Running Core Assumption Micro-Experiments")
    print("=" * 60)
    print("Testing fundamental brain architecture assumptions...")
    
    from validation.micro_experiments.core_assumptions import create_core_assumption_suite
    from validation.test_integration import IntegrationTestSuite
    
    # Start server for experiments
    integration_suite = IntegrationTestSuite()
    integration_suite._test_server_startup()
    
    if not integration_suite._is_server_ready():
        print("❌ Failed to start server for micro-experiments")
        return {"success_rate": 0.0, "avg_confidence": 0.0}
    
    try:
        suite = create_core_assumption_suite()
        summary = suite.run_all()
        suite.print_summary()
        return summary
    finally:
        # Clean up server
        integration_suite._test_server_shutdown()

def run_all_experiments():
    """Run all available micro-experiments."""
    print("🔬 Running All Micro-Experiments")
    print("=" * 60)
    
    # Run core experiments
    core_summary = run_core_experiments()
    
    # TODO: Add other experiment suites when implemented
    # embodied_summary = run_embodied_experiments()
    # learning_summary = run_learning_experiments()
    
    return {
        'core_experiments': core_summary
    }

def list_experiments():
    """List available experiments."""
    print("🔬 Available Micro-Experiments")
    print("=" * 60)
    
    print("\\n📋 Core Assumption Experiments:")
    print("   • Similarity Consistency - Similar inputs should produce similar outputs")
    print("   • Prediction Error Learning - Prediction error should decrease over time")
    print("   • Experience Scaling - More experience should improve performance")
    print("   • Sensory-Motor Coordination - 16D input should enable meaningful 4D actions")
    
    print("\\n📋 Future Experiment Suites:")
    print("   • Embodied Free Energy - Action selection through Free Energy minimization")
    print("   • Learning Dynamics - Activation spreading and memory effects")
    print("   • Biological Realism - Gradual learning and consolidation benefits")
    print("   • Environmental Adaptation - Robustness across different scenarios")
    
    print("\\n🚀 Usage:")
    print("   python3 validation/micro_experiment_runner.py core")
    print("   python3 validation/micro_experiment_runner.py all")

def main():
    """Main micro-experiment runner."""
    parser = argparse.ArgumentParser(description='Brain Assumption Micro-Experiment Runner')
    parser.add_argument('suite', nargs='?', default='core',
                       choices=['core', 'all'],
                       help='Which experiment suite to run')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    parser.add_argument('--stop-on-failure', action='store_true',
                       help='Stop running experiments on first failure')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    print(f"🧪 Brain Assumption Validation")
    print(f"   Suite: {args.suite}")
    print(f"   Stop on failure: {args.stop_on_failure}")
    
    try:
        if args.suite == 'core':
            summary = run_core_experiments()
        elif args.suite == 'all':
            summary = run_all_experiments()
        
        # Print final assessment
        print(f"\\n🎯 Validation Assessment")
        print("=" * 60)
        
        if args.suite == 'core':
            success_rate = summary['success_rate']
            avg_confidence = summary['avg_confidence']
            
            if success_rate >= 0.75:
                print("✅ VALIDATION PASSED: Core assumptions are well-supported")
                print("   → Proceed with full validation experiments")
            elif success_rate >= 0.5:
                print("⚠️ VALIDATION PARTIAL: Some assumptions need refinement")
                print("   → Fix failing assumptions before full experiments")
            else:
                print("❌ VALIDATION FAILED: Major assumptions are not supported")
                print("   → Significant architecture changes needed")
            
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Average confidence: {avg_confidence:.3f}")
        
        return 0 if summary.get('success_rate', 0) >= 0.75 else 1
        
    except KeyboardInterrupt:
        print("\\n⏹️ Experiments interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Experiment suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
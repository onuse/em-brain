#!/usr/bin/env python3
"""
Validation Runner

Execute scientific validation studies for the embodied AI brain system.
This runner manages long-running experiments, data collection, and analysis.

Usage:
  python3 validation_runner.py embodied_learning.biological_embodied_learning
  python3 validation_runner.py --list
  python3 validation_runner.py --all
"""

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional

def get_available_studies() -> Dict[str, Dict]:
    """Get all available validation studies."""
    
    studies = {
        'embodied_learning': {
            'biological_embodied_learning': {
                'path': 'validation/embodied_learning/experiments/biological_embodied_learning.py',
                'description': 'Biological timescale embodied learning with sensory-motor coordination',
                'duration': '2-8 hours',
                'scientific_value': 'High - Tests core embodied intelligence hypotheses'
            }
        },
        'legacy_tests': {
            'biological_timescales': {
                'path': 'tests/test_biological_timescales.py',
                'description': 'Simple pattern learning over biological timescales',
                'duration': '1-8 hours',
                'scientific_value': 'Low - Trivial pattern learning'
            }
        }
    }
    
    return studies

def list_studies():
    """List all available validation studies."""
    studies = get_available_studies()
    
    print("🔬 Available Validation Studies:")
    print("=" * 60)
    
    for category, category_studies in studies.items():
        print(f"\n📂 {category.replace('_', ' ').title()}:")
        
        for study_name, study_info in category_studies.items():
            print(f"   🧪 {study_name}")
            print(f"      Description: {study_info['description']}")
            print(f"      Duration: {study_info['duration']}")
            print(f"      Scientific Value: {study_info['scientific_value']}")
            print()

def run_study(study_path: str, args: List[str] = None) -> bool:
    """Run a validation study."""
    
    if not Path(study_path).exists():
        print(f"❌ Study not found: {study_path}")
        return False
    
    print(f"🚀 Running validation study: {study_path}")
    print(f"   Arguments: {' '.join(args) if args else 'None'}")
    print(f"   Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Build command
        cmd = ['python3', study_path]
        if args:
            cmd.extend(args)
        
        # Run study
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ Study completed successfully")
            return True
        else:
            print(f"❌ Study failed with return code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\\n⏹️ Study interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Study failed with error: {e}")
        return False

def run_all_studies():
    """Run all validation studies."""
    studies = get_available_studies()
    
    print("🔬 Running All Validation Studies")
    print("=" * 60)
    
    results = {}
    
    for category, category_studies in studies.items():
        print(f"\\n📂 Category: {category}")
        
        for study_name, study_info in category_studies.items():
            print(f"\\n🧪 Running {study_name}...")
            
            # Skip legacy tests in full run
            if category == 'legacy_tests':
                print("   ⏭️ Skipping legacy test")
                results[f"{category}.{study_name}"] = 'skipped'
                continue
            
            success = run_study(study_info['path'])
            results[f"{category}.{study_name}"] = 'success' if success else 'failed'
    
    # Print summary
    print(f"\\n📊 Validation Results Summary:")
    print("=" * 40)
    
    for study, result in results.items():
        status_icon = {'success': '✅', 'failed': '❌', 'skipped': '⏭️'}[result]
        print(f"   {status_icon} {study}: {result}")
    
    # Overall statistics
    successes = sum(1 for r in results.values() if r == 'success')
    failures = sum(1 for r in results.values() if r == 'failed')
    skipped = sum(1 for r in results.values() if r == 'skipped')
    
    print(f"\\n📈 Overall: {successes} passed, {failures} failed, {skipped} skipped")
    
    return failures == 0

def parse_study_identifier(identifier: str) -> Optional[str]:
    """Parse study identifier and return path."""
    studies = get_available_studies()
    
    # Handle full identifiers like "embodied_learning.biological_embodied_learning"
    if '.' in identifier:
        category, study_name = identifier.split('.', 1)
        if category in studies and study_name in studies[category]:
            return studies[category][study_name]['path']
    
    # Handle study names directly
    for category, category_studies in studies.items():
        if identifier in category_studies:
            return category_studies[identifier]['path']
    
    return None

def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(
        description='Scientific Validation Runner for Embodied AI Brain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 validation_runner.py embodied_learning.biological_embodied_learning
  python3 validation_runner.py biological_embodied_learning --hours 4
  python3 validation_runner.py --list
  python3 validation_runner.py --all
        """
    )
    
    parser.add_argument('study', nargs='?', 
                       help='Study identifier (e.g., embodied_learning.biological_embodied_learning)')
    parser.add_argument('--list', action='store_true',
                       help='List all available validation studies')
    parser.add_argument('--all', action='store_true',
                       help='Run all validation studies')
    parser.add_argument('--hours', type=float,
                       help='Duration in hours (for applicable studies)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args, unknown_args = parser.parse_known_args()
    
    if args.list:
        list_studies()
        return
    
    if args.all:
        success = run_all_studies()
        sys.exit(0 if success else 1)
    
    if not args.study:
        parser.print_help()
        return
    
    # Parse study identifier
    study_path = parse_study_identifier(args.study)
    
    if not study_path:
        print(f"❌ Unknown study: {args.study}")
        print("\\nAvailable studies:")
        list_studies()
        sys.exit(1)
    
    # Build arguments for study
    study_args = unknown_args.copy()
    
    if args.hours:
        study_args.extend(['--hours', str(args.hours)])
    
    if args.seed:
        study_args.extend(['--seed', str(args.seed)])
    
    # Run the study
    success = run_study(study_path, study_args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
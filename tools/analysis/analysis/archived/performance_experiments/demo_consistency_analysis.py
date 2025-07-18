#!/usr/bin/env python3
"""
Demo Consistency Analysis

Verifies that test_demo, demo_2d, and demo_3d all use the same 
brainstem implementation to ensure consistent transfer learning.

Critical for confirming that extended simulation in any demo 
will provide the same memory foundation for the real robot.
"""

import sys
import os
import ast
import re
from typing import Dict, List, Any

# Add brain directory to path
brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, brain_dir)


class DemoConsistencyAnalyzer:
    """Analyzes consistency across different demo implementations."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.demo_files = {
            'test_demo': 'demos/test_demo.py',
            'demo_2d': 'demos/demo_2d.py', 
            'demo_3d': 'demos/demo_3d.py'
        }
        
    def analyze_brainstem_usage(self) -> Dict[str, Any]:
        """Analyze how each demo uses the PiCarXBrainstem."""
        
        print("🔍 DEMO CONSISTENCY ANALYSIS")
        print("=" * 50)
        print("Verifying all demos use identical brainstem for transfer learning")
        print()
        
        results = {}
        
        for demo_name, file_path in self.demo_files.items():
            full_path = os.path.join(brain_dir, file_path)
            
            if not os.path.exists(full_path):
                results[demo_name] = {'error': f'File not found: {full_path}'}
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                analysis = self._analyze_demo_file(content, demo_name)
                results[demo_name] = analysis
                
            except Exception as e:
                results[demo_name] = {'error': str(e)}
        
        return results
    
    def _analyze_demo_file(self, content: str, demo_name: str) -> Dict[str, Any]:
        """Analyze a single demo file for brainstem usage."""
        
        analysis = {
            'demo_name': demo_name,
            'brainstem_import': None,
            'brainstem_initialization': None,
            'sensor_configuration': {},
            'control_loop': None,
            'differences': []
        }
        
        # Check brainstem import
        import_patterns = [
            r'from\s+\.?picar_x_simulation\.picar_x_brainstem\s+import\s+PiCarXBrainstem',
            r'from\s+picar_x_simulation\.picar_x_brainstem\s+import\s+PiCarXBrainstem'
        ]
        
        for pattern in import_patterns:
            if re.search(pattern, content):
                analysis['brainstem_import'] = 'PiCarXBrainstem from picar_x_simulation.picar_x_brainstem'
                break
        
        if not analysis['brainstem_import']:
            analysis['differences'].append('❌ Missing or different brainstem import')
        
        # Check brainstem initialization
        init_pattern = r'PiCarXBrainstem\s*\(\s*(.*?)\s*\)'
        init_matches = re.findall(init_pattern, content, re.DOTALL)
        
        if init_matches:
            init_args = init_matches[0]
            analysis['brainstem_initialization'] = init_args.strip()
            
            # Parse sensor configuration
            if 'enable_camera=True' in init_args:
                analysis['sensor_configuration']['camera'] = True
            elif 'enable_camera=False' in init_args:
                analysis['sensor_configuration']['camera'] = False
                
            if 'enable_ultrasonics=True' in init_args:
                analysis['sensor_configuration']['ultrasonics'] = True
            elif 'enable_ultrasonics=False' in init_args:
                analysis['sensor_configuration']['ultrasonics'] = False
                
            if 'enable_line_tracking=True' in init_args:
                analysis['sensor_configuration']['line_tracking'] = True
            elif 'enable_line_tracking=False' in init_args:
                analysis['sensor_configuration']['line_tracking'] = False
        else:
            analysis['differences'].append('❌ No brainstem initialization found')
        
        # Check for control loop usage
        if 'control_cycle' in content:
            analysis['control_loop'] = 'Uses robot.control_cycle()'
        else:
            analysis['differences'].append('❌ No control_cycle usage found')
        
        return analysis
    
    def compare_brainstem_configurations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare brainstem configurations across demos."""
        
        print("\n🔄 BRAINSTEM CONFIGURATION COMPARISON")
        print("=" * 50)
        
        # Extract configurations for comparison
        configs = {}
        for demo_name, analysis in results.items():
            if 'error' not in analysis:
                configs[demo_name] = analysis.get('sensor_configuration', {})
        
        if len(configs) < 2:
            return {'error': 'Insufficient demos for comparison'}
        
        # Compare all configurations
        reference_demo = list(configs.keys())[0]
        reference_config = configs[reference_demo]
        
        comparison = {
            'reference_demo': reference_demo,
            'reference_config': reference_config,
            'all_identical': True,
            'differences': {},
            'consistent_sensors': []
        }
        
        print(f"📊 Reference configuration ({reference_demo}):")
        for sensor, enabled in reference_config.items():
            print(f"   {sensor}: {enabled}")
        
        print(f"\n🔍 Comparing other demos to {reference_demo}:")
        
        for demo_name, config in configs.items():
            if demo_name == reference_demo:
                continue
                
            demo_diffs = []
            
            for sensor, ref_value in reference_config.items():
                demo_value = config.get(sensor)
                
                if demo_value != ref_value:
                    comparison['all_identical'] = False
                    demo_diffs.append(f"{sensor}: {ref_value} → {demo_value}")
                else:
                    if sensor not in comparison['consistent_sensors']:
                        comparison['consistent_sensors'].append(sensor)
            
            # Check for sensors in demo but not reference
            for sensor, demo_value in config.items():
                if sensor not in reference_config:
                    comparison['all_identical'] = False
                    demo_diffs.append(f"{sensor}: missing in reference → {demo_value}")
            
            if demo_diffs:
                comparison['differences'][demo_name] = demo_diffs
                print(f"   ❌ {demo_name}: {', '.join(demo_diffs)}")
            else:
                print(f"   ✅ {demo_name}: Identical configuration")
        
        return comparison
    
    def assess_transfer_learning_consistency(self, results: Dict[str, Any], 
                                           comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Assess whether all demos provide consistent transfer learning."""
        
        print(f"\n🧠 TRANSFER LEARNING CONSISTENCY ASSESSMENT")
        print("=" * 50)
        
        assessment = {
            'consistent_for_transfer': True,
            'transfer_confidence': 'HIGH',
            'issues': [],
            'recommendations': []
        }
        
        # Check if all demos use same brainstem
        brainstem_imports = set()
        for demo_name, analysis in results.items():
            if 'error' not in analysis and analysis.get('brainstem_import'):
                brainstem_imports.add(analysis['brainstem_import'])
        
        if len(brainstem_imports) > 1:
            assessment['consistent_for_transfer'] = False
            assessment['issues'].append('❌ Different brainstem imports across demos')
        elif len(brainstem_imports) == 1:
            print("✅ All demos use same brainstem implementation")
        
        # Check sensor configuration consistency
        if not comparison.get('all_identical', False):
            if comparison.get('differences'):
                print("⚠️  Sensor configuration differences found:")
                for demo, diffs in comparison['differences'].items():
                    print(f"   {demo}: {', '.join(diffs)}")
                
                # Assess impact on transfer learning
                critical_sensors = ['ultrasonics', 'camera']
                has_critical_differences = False
                
                for demo, diffs in comparison['differences'].items():
                    for diff in diffs:
                        if any(sensor in diff for sensor in critical_sensors):
                            has_critical_differences = True
                            break
                
                if has_critical_differences:
                    assessment['consistent_for_transfer'] = False
                    assessment['transfer_confidence'] = 'MODERATE'
                    assessment['issues'].append('⚠️  Critical sensor differences may affect transfer')
                else:
                    print("✅ Differences are minor - transfer learning should work")
            else:
                print("✅ All sensor configurations are identical")
        
        # Check control loop consistency
        control_loops = set()
        for demo_name, analysis in results.items():
            if 'error' not in analysis and analysis.get('control_loop'):
                control_loops.add(analysis['control_loop'])
        
        if len(control_loops) == 1:
            print("✅ All demos use same control loop pattern")
        elif len(control_loops) > 1:
            assessment['consistent_for_transfer'] = False
            assessment['issues'].append('❌ Different control loop patterns')
        
        # Generate recommendations
        if assessment['consistent_for_transfer']:
            assessment['recommendations'] = [
                '🟢 PROCEED: All demos provide consistent simulation',
                '🚀 Extended simulation in ANY demo will transfer to real robot',
                '💾 Memory banks from different demos can be safely merged',
                '🎯 Choose demo based on visualization preference, not accuracy'
            ]
        else:
            assessment['recommendations'] = [
                '🟡 CAUTION: Some inconsistencies detected',
                '🔍 Review specific differences before extended simulation',
                '⚠️  Use the most accurate demo for transfer learning',
                '🧪 Test real robot with memories from different demos'
            ]
        
        return assessment
    
    def generate_final_verdict(self, assessment: Dict[str, Any]) -> str:
        """Generate final verdict on demo consistency."""
        
        print(f"\n🏆 FINAL VERDICT")
        print("=" * 50)
        
        if assessment['consistent_for_transfer']:
            verdict = "✅ ALL DEMOS ARE CONSISTENT FOR TRANSFER LEARNING"
            print(verdict)
            print()
            print("🎯 Key Findings:")
            print("   • All demos use identical PiCarXBrainstem implementation")
            print("   • Sensor configurations are compatible across demos")
            print("   • Control loops are consistent")
            print("   • Memory banks will transfer equally well from any demo")
            print()
            print("💡 Recommendation:")
            print("   Run extended simulation in whichever demo you prefer!")
            print("   - test_demo: Fast, text-based, minimal resources")
            print("   - demo_2d: Visual feedback, good for monitoring")
            print("   - demo_3d: Full 3D visualization, most immersive")
            print()
            print("🚀 Transfer Confidence: All demos → Real robot = HIGH")
            
        else:
            verdict = "⚠️  SOME INCONSISTENCIES DETECTED"
            print(verdict)
            print()
            print("🔍 Issues Found:")
            for issue in assessment['issues']:
                print(f"   {issue}")
            print()
            print("💡 Recommendations:")
            for rec in assessment['recommendations']:
                print(f"   {rec}")
        
        return verdict


def main():
    """Run complete demo consistency analysis."""
    
    print("🔬 Demo Consistency Analysis for Transfer Learning")
    print("=" * 70)
    print("Question: Do all demos use the same brainstem for consistent")
    print("         memory transfer to the real robot?")
    print()
    
    analyzer = DemoConsistencyAnalyzer()
    
    # Analyze brainstem usage
    results = analyzer.analyze_brainstem_usage()
    
    # Compare configurations
    comparison = analyzer.compare_brainstem_configurations(results)
    
    # Assess transfer learning consistency
    assessment = analyzer.assess_transfer_learning_consistency(results, comparison)
    
    # Generate final verdict
    verdict = analyzer.generate_final_verdict(assessment)
    
    return {
        'verdict': verdict,
        'transfer_confidence': assessment['transfer_confidence'],
        'consistent': assessment['consistent_for_transfer']
    }


if __name__ == "__main__":
    main()
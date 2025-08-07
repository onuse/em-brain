#!/usr/bin/env python3
"""
Brain Architecture Analysis
===========================

Static analysis of three brain implementations to understand their
architectural differences and predict deployment characteristics.

Since we can't run the full benchmark without PyTorch, this provides
architectural insights and theoretical performance analysis.
"""

import os
import sys
import re
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

class BrainArchitectureAnalyzer:
    """
    Analyzes brain architectures statically to understand their complexity,
    design patterns, and predicted performance characteristics.
    """
    
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.brain_files = {
            'unified': 'src/brains/field/unified_field_brain.py',
            'minimal': 'src/brains/field/minimal_field_brain.py', 
            'pure': 'src/brains/field/pure_field_brain.py'
        }
        
    def analyze_code_complexity(self, filepath: str) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        try:
            with open(os.path.join(self.server_path, filepath), 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': 'File not found'}
        
        lines = content.split('\n')
        
        # Basic metrics
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        blank_lines = len([l for l in lines if not l.strip()])
        
        # Import complexity
        import_lines = [l for l in lines if l.strip().startswith('from ') or l.strip().startswith('import ')]
        imports = len(import_lines)
        
        # Class and function analysis
        class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        function_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
        method_count = len(re.findall(r'^\s+def\s+\w+', content, re.MULTILINE))
        
        # Complexity indicators
        torch_ops = len(re.findall(r'torch\.\w+', content))
        tensor_ops = len(re.findall(r'\.tensor|\.cuda|\.cpu|\.float|\.double', content))
        gpu_ops = len(re.findall(r'cuda|device|gpu', content, re.IGNORECASE))
        
        # Control flow complexity
        if_statements = len(re.findall(r'\bif\s+', content))
        for_loops = len(re.findall(r'\bfor\s+', content))
        while_loops = len(re.findall(r'\bwhile\s+', content))
        try_blocks = len(re.findall(r'\btry:', content))
        
        # Mathematical operations
        math_ops = len(re.findall(r'[+\-*/]|torch\.|np\.|math\.', content))
        
        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines,
            'imports': imports,
            'classes': class_count,
            'functions': function_count,
            'methods': method_count,
            'torch_operations': torch_ops,
            'tensor_operations': tensor_ops,
            'gpu_operations': gpu_ops,
            'if_statements': if_statements,
            'for_loops': for_loops,
            'while_loops': while_loops,
            'try_blocks': try_blocks,
            'math_operations': math_ops,
            'complexity_score': self._calculate_complexity_score(
                code_lines, imports, function_count + method_count,
                if_statements + for_loops + while_loops, torch_ops
            )
        }
    
    def _calculate_complexity_score(self, code_lines: int, imports: int, 
                                   functions: int, control_flow: int, 
                                   torch_ops: int) -> float:
        """Calculate relative complexity score (0-100)"""
        # Weighted complexity
        score = (
            code_lines * 0.1 +      # Lines of code
            imports * 2.0 +         # Import overhead
            functions * 1.5 +       # Function complexity
            control_flow * 1.0 +    # Control flow
            torch_ops * 0.5         # GPU operations
        ) / 10.0
        
        return min(100.0, score)
    
    def analyze_dependencies(self, filepath: str) -> Dict[str, List[str]]:
        """Analyze import dependencies"""
        try:
            with open(os.path.join(self.server_path, filepath), 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': 'File not found'}
        
        lines = content.split('\n')
        
        # Extract imports
        standard_libs = []
        third_party = []
        local_imports = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                if 'torch' in line or 'numpy' in line:
                    third_party.append(line)
                elif line.startswith('from .') or line.startswith('from ...'):
                    local_imports.append(line)
                elif any(lib in line for lib in ['os', 'sys', 'time', 'json', 're', 'typing', 'collections', 'dataclasses', 'math', 'logging']):
                    standard_libs.append(line)
                else:
                    third_party.append(line)
        
        return {
            'standard_library': standard_libs,
            'third_party': third_party,
            'local_imports': local_imports,
            'dependency_count': len(standard_libs) + len(third_party) + len(local_imports)
        }
    
    def analyze_field_operations(self, filepath: str) -> Dict[str, Any]:
        """Analyze field-specific operations and patterns"""
        try:
            with open(os.path.join(self.server_path, filepath), 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': 'File not found'}
        
        # Field-specific patterns
        field_creation = len(re.findall(r'field.*=.*torch\.|self\.field', content))
        convolution_ops = len(re.findall(r'conv3d|conv2d|F\.conv', content))
        activation_funcs = len(re.findall(r'tanh|relu|sigmoid|softmax', content))
        gradient_ops = len(re.findall(r'grad|gradient|backward', content))
        diffusion_ops = len(re.findall(r'diffus|blur|smooth', content))
        
        # Learning patterns
        learning_refs = len(re.findall(r'learn|adapt|error|predict', content))
        reward_refs = len(re.findall(r'reward|reinforce', content))
        
        # Subsystem complexity
        subsystems = []
        if 'MotorCortex' in content:
            subsystems.append('MotorCortex')
        if 'SensoryMapping' in content:
            subsystems.append('SensoryMapping')
        if 'PatternSystem' in content:
            subsystems.append('PatternSystem')
        if 'PredictiveSystem' in content:
            subsystems.append('PredictiveSystem')
        if 'StrategicPlanner' in content:
            subsystems.append('StrategicPlanner')
        
        return {
            'field_operations': field_creation,
            'convolution_operations': convolution_ops,
            'activation_functions': activation_funcs,
            'gradient_operations': gradient_ops,
            'diffusion_operations': diffusion_ops,
            'learning_references': learning_refs,
            'reward_references': reward_refs,
            'subsystems': subsystems,
            'subsystem_count': len(subsystems),
            'field_sophistication': field_creation + convolution_ops + activation_funcs + gradient_ops
        }
    
    def predict_performance_characteristics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance characteristics based on code analysis"""
        
        # Performance prediction (higher complexity = slower)
        complexity = analysis['complexity']['complexity_score']
        performance_score = max(0, 100 - complexity) / 100
        
        # Memory prediction (more operations = more memory)
        operations = (analysis['complexity']['torch_operations'] + 
                     analysis['complexity']['tensor_operations'])
        memory_score = max(0, 100 - operations) / 100
        
        # Learning capability (more learning refs = better learning)
        learning_refs = analysis['field_ops']['learning_references']
        learning_score = min(1.0, learning_refs / 20.0)
        
        # Stability prediction (fewer control structures = more stable)
        control_complexity = (analysis['complexity']['if_statements'] +
                            analysis['complexity']['for_loops'] +
                            analysis['complexity']['while_loops'])
        stability_score = max(0, 100 - control_complexity * 2) / 100
        
        # GPU efficiency (more GPU ops but not too complex)
        gpu_ops = analysis['complexity']['gpu_operations']
        gpu_efficiency = min(1.0, gpu_ops / 10.0) * (1.0 - complexity / 200.0)
        
        # Deployment readiness
        subsystem_penalty = analysis['field_ops']['subsystem_count'] * 0.1
        deployment_score = (performance_score * 0.3 + 
                          memory_score * 0.2 +
                          learning_score * 0.2 + 
                          stability_score * 0.3) - subsystem_penalty
        
        return {
            'predicted_performance': performance_score,
            'predicted_memory_efficiency': memory_score,
            'predicted_learning_capability': learning_score,
            'predicted_stability': stability_score,
            'predicted_gpu_efficiency': gpu_efficiency,
            'predicted_deployment_score': max(0, deployment_score)
        }
    
    def analyze_brain(self, brain_name: str, filepath: str) -> Dict[str, Any]:
        """Complete analysis of one brain"""
        
        print(f"\n🧠 Analyzing {brain_name.upper()} brain...")
        
        # Basic analysis
        complexity = self.analyze_code_complexity(filepath)
        dependencies = self.analyze_dependencies(filepath)
        field_ops = self.analyze_field_operations(filepath)
        
        # Combined analysis
        analysis = {
            'name': brain_name,
            'complexity': complexity,
            'dependencies': dependencies,
            'field_ops': field_ops
        }
        
        # Performance predictions
        predictions = self.predict_performance_characteristics(analysis)
        analysis['predictions'] = predictions
        
        # Print summary
        if 'error' not in complexity:
            print(f"   Lines of code: {complexity['code_lines']}")
            print(f"   Dependencies: {dependencies['dependency_count']}")
            print(f"   Subsystems: {field_ops['subsystem_count']}")
            print(f"   Complexity score: {complexity['complexity_score']:.1f}/100")
            print(f"   Predicted deployment score: {predictions['predicted_deployment_score']:.1%}")
        
        return analysis
    
    def run_analysis(self) -> List[Dict[str, Any]]:
        """Run complete analysis on all brains"""
        
        print("🔍 Brain Architecture Analysis")
        print("=" * 60)
        print("Analyzing UnifiedFieldBrain vs MinimalFieldBrain vs PureFieldBrain")
        print("Static code analysis to predict performance characteristics\n")
        
        results = []
        
        for brain_name, filepath in self.brain_files.items():
            try:
                analysis = self.analyze_brain(brain_name, filepath)
                results.append(analysis)
            except Exception as e:
                print(f"❌ Failed to analyze {brain_name}: {e}")
        
        return results
    
    def print_comparison(self, results: List[Dict[str, Any]]):
        """Print detailed comparison of all brains"""
        
        print("\n" + "=" * 80)
        print("📊 ARCHITECTURAL COMPARISON")
        print("=" * 80)
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r.get('complexity', {})]
        
        if not valid_results:
            print("❌ No valid results to compare")
            return
        
        # Comparison table
        print(f"\n{'Metric':<25} {'Unified':<12} {'Minimal':<12} {'Pure':<12}")
        print("-" * 65)
        
        metrics = [
            ('Lines of Code', 'complexity', 'code_lines'),
            ('Dependencies', 'dependencies', 'dependency_count'),
            ('Subsystems', 'field_ops', 'subsystem_count'),
            ('Torch Operations', 'complexity', 'torch_operations'),
            ('Complexity Score', 'complexity', 'complexity_score'),
            ('Learning References', 'field_ops', 'learning_references'),
        ]
        
        for metric_name, category, key in metrics:
            values = []
            for result in valid_results:
                if category in result and key in result[category]:
                    val = result[category][key]
                    if isinstance(val, float):
                        values.append(f"{val:.1f}")
                    else:
                        values.append(str(val))
                else:
                    values.append("N/A")
            
            # Pad values to match brain order
            while len(values) < 3:
                values.append("N/A")
            
            print(f"{metric_name:<25} {values[0]:<12} {values[1]:<12} {values[2]:<12}")
        
        # Predictions table
        print(f"\n{'Prediction':<25} {'Unified':<12} {'Minimal':<12} {'Pure':<12}")
        print("-" * 65)
        
        prediction_metrics = [
            ('Performance', 'predicted_performance'),
            ('Memory Efficiency', 'predicted_memory_efficiency'),
            ('Learning Capability', 'predicted_learning_capability'),
            ('Stability', 'predicted_stability'),
            ('GPU Efficiency', 'predicted_gpu_efficiency'),
            ('Deployment Score', 'predicted_deployment_score'),
        ]
        
        for metric_name, key in prediction_metrics:
            values = []
            for result in valid_results:
                if 'predictions' in result and key in result['predictions']:
                    val = result['predictions'][key]
                    values.append(f"{val:.1%}")
                else:
                    values.append("N/A")
                    
            while len(values) < 3:
                values.append("N/A")
            
            print(f"{metric_name:<25} {values[0]:<12} {values[1]:<12} {values[2]:<12}")
        
        # Winner analysis
        best_brain = None
        best_score = -1
        
        for result in valid_results:
            if 'predictions' in result:
                score = result['predictions'].get('predicted_deployment_score', 0)
                if score > best_score:
                    best_score = score
                    best_brain = result
        
        if best_brain:
            print(f"\n🏆 PREDICTED WINNER: {best_brain['name'].upper()}")
            print(f"   Predicted Deployment Score: {best_score:.1%}")
            
            complexity_score = best_brain['complexity'].get('complexity_score', 0)
            if complexity_score < 30:
                print("   ✅ Low complexity - should be fast and stable")
            elif complexity_score < 60:
                print("   ⚠️  Medium complexity - balanced performance")
            else:
                print("   ❌ High complexity - may be slow or unstable")
                
            subsystems = best_brain['field_ops'].get('subsystem_count', 0)
            if subsystems == 0:
                print("   ✅ No subsystems - minimal architecture")
            elif subsystems < 5:
                print(f"   ⚠️  {subsystems} subsystems - moderate complexity")
            else:
                print(f"   ❌ {subsystems} subsystems - high integration complexity")
        
        # Architectural insights
        print(f"\n🔍 ARCHITECTURAL INSIGHTS:")
        
        for result in valid_results:
            name = result['name']
            complexity = result.get('complexity', {})
            field_ops = result.get('field_ops', {})
            
            print(f"\n   {name.upper()}:")
            
            if complexity:
                lines = complexity.get('code_lines', 0)
                if lines < 300:
                    print(f"     • Minimal implementation ({lines} lines)")
                elif lines < 800:
                    print(f"     • Moderate implementation ({lines} lines)")
                else:
                    print(f"     • Complex implementation ({lines} lines)")
                
                torch_ops = complexity.get('torch_operations', 0)
                if torch_ops < 20:
                    print(f"     • Low PyTorch usage ({torch_ops} ops)")
                elif torch_ops < 50:
                    print(f"     • Moderate PyTorch usage ({torch_ops} ops)")
                else:
                    print(f"     • Heavy PyTorch usage ({torch_ops} ops)")
            
            if field_ops:
                subsystem_count = field_ops.get('subsystem_count', 0)
                if subsystem_count == 0:
                    print("     • Monolithic design - all in one")
                else:
                    subsystems = field_ops.get('subsystems', [])
                    print(f"     • Modular design: {', '.join(subsystems)}")
                
                sophistication = field_ops.get('field_sophistication', 0)
                if sophistication < 10:
                    print("     • Simple field operations")
                elif sophistication < 30:
                    print("     • Moderate field sophistication")
                else:
                    print("     • Advanced field processing")
        
        # Deployment recommendation
        print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
        
        if best_brain:
            name = best_brain['name']
            print(f"   For real robot deployment, choose: {name}")
            
            # Specific recommendations based on analysis
            complexity_score = best_brain['complexity'].get('complexity_score', 0)
            subsystems = best_brain['field_ops'].get('subsystem_count', 0)
            
            if name == 'pure':
                print("   Rationale: Single tensor operation, GPU-optimal, minimal complexity")
                print("   Pros: Fastest, most efficient, least likely to crash")
                print("   Cons: May need more tuning for specific behaviors")
            elif name == 'minimal':
                print("   Rationale: Simple but effective, good balance")
                print("   Pros: Easy to understand, reasonable performance")
                print("   Cons: May not scale to complex behaviors")
            else:  # unified
                print("   Rationale: Full-featured but complex")
                print("   Pros: Rich behaviors, well-tested subsystems")
                print("   Cons: Higher memory, more failure modes")
        
    def save_analysis(self, results: List[Dict[str, Any]], filename: str = None):
        """Save analysis results to JSON"""
        if filename is None:
            filename = f"brain_architecture_analysis.json"
        
        analysis_data = {
            'analysis_type': 'static_code_analysis',
            'brains_analyzed': len(results),
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n💾 Analysis saved to: {filename}")


def main():
    """Run the architectural analysis"""
    server_path = "/mnt/c/Users/glimm/Documents/Projects/em-brain/server"
    
    if not os.path.exists(server_path):
        print(f"❌ Server path not found: {server_path}")
        return
    
    analyzer = BrainArchitectureAnalyzer(server_path)
    results = analyzer.run_analysis()
    analyzer.print_comparison(results)
    analyzer.save_analysis(results)
    
    return results


if __name__ == "__main__":
    main()
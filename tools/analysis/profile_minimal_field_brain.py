#!/usr/bin/env python3
"""
Profile Minimal Field Brain - Deep Dive

Now that we've eliminated the threading bottleneck, let's see exactly
what's consuming the remaining 5.568 seconds of processing time.
"""

import sys
import os
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def profile_minimal_field_brain():
    """Profile the field brain with minimal infrastructure to see pure computation cost."""
    print("🔍 PROFILING MINIMAL FIELD BRAIN")
    print("=" * 50)
    print("Question: What's consuming the 5.568 seconds?")
    print("Expected: Pure field mathematics and justified function calls")
    print()
    
    try:
        from src.brain import MinimalBrain
        
        # Same minimal config that gave us 5.568s
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 10,  # Small field
                "field_temporal_window": 5.0,   # Short window
                "field_evolution_rate": 0.05,   
                "constraint_discovery_rate": 0.05
            },
            "memory": {"enable_persistence": False},
            "logging": {
                "log_brain_cycles": False,
                "log_pattern_storage": False,
                "log_performance": False
            }
        }
        
        print("🧠 Creating minimal field brain...")
        brain = MinimalBrain(config=config, quiet_mode=True, enable_logging=False)
        
        print("🔬 Starting detailed profiling...")
        
        # Profile a single processing cycle
        sensory_input = [0.1] * 16
        
        # Use cProfile for detailed function-by-function analysis
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        action, brain_state = brain.process_sensory_input(sensory_input)
        
        profiler.disable()
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.3f}s")
        
        # Analyze the profile in detail
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(50)  # Top 50 functions
        
        profile_data = s.getvalue()
        
        # Save detailed profile
        profile_file = Path(__file__).parent.parent.parent / "logs" / "minimal_field_brain_profile.txt"
        profile_file.parent.mkdir(exist_ok=True)
        with open(profile_file, 'w') as f:
            f.write(profile_data)
        
        print(f"💾 Detailed profile saved to: {profile_file}")
        
        # Analyze the profile data
        analyze_profile_data(profile_data, processing_time)
        
        brain.finalize_session()
        return processing_time
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_profile_data(profile_data: str, total_time: float):
    """Analyze profile data to identify what's consuming time."""
    print(f"\n📊 PROFILE ANALYSIS")
    print("=" * 30)
    
    lines = profile_data.split('\n')
    
    # Look for the top time consumers
    print("🎯 Top Time Consumers:")
    
    # Parse the profile data
    function_times = []
    parsing_data = False
    
    for line in lines:
        if 'ncalls' in line and 'tottime' in line:
            parsing_data = True
            continue
        
        if parsing_data and line.strip():
            parts = line.split()
            if len(parts) >= 6 and parts[0].replace('.', '').isdigit():
                try:
                    ncalls = parts[0]
                    tottime = float(parts[1])
                    cumtime = float(parts[3])
                    function_name = ' '.join(parts[5:])
                    
                    function_times.append({
                        'ncalls': ncalls,
                        'tottime': tottime,
                        'cumtime': cumtime,
                        'function': function_name,
                        'percentage': (cumtime / total_time) * 100
                    })
                except (ValueError, IndexError):
                    continue
    
    # Show top time consumers
    print(f"\n⏱️ Functions consuming >1% of total time:")
    for i, func in enumerate(function_times[:20]):
        if func['percentage'] > 1.0:
            print(f"   {i+1:2d}. {func['cumtime']:6.3f}s ({func['percentage']:5.1f}%) - {func['function'][:80]}")
    
    # Categorize the function calls
    categorize_functions(function_times, total_time)

def categorize_functions(function_times: list, total_time: float):
    """Categorize functions to understand what types of operations are slow."""
    print(f"\n🏷️ OPERATION CATEGORIES:")
    
    categories = {
        'field_operations': [],
        'torch_operations': [],
        'math_operations': [],
        'system_operations': [],
        'unknown_operations': []
    }
    
    for func in function_times:
        func_name = func['function'].lower()
        
        if any(keyword in func_name for keyword in ['field', 'dynamics', 'unified', 'constraint', 'evolution']):
            categories['field_operations'].append(func)
        elif any(keyword in func_name for keyword in ['torch', 'tensor', 'cuda', 'gpu']):
            categories['torch_operations'].append(func)
        elif any(keyword in func_name for keyword in ['math', 'numpy', 'sqrt', 'pow', 'mean']):
            categories['math_operations'].append(func)
        elif any(keyword in func_name for keyword in ['system', 'built-in', 'method', 'thread', 'lock', 'queue']):
            categories['system_operations'].append(func)
        else:
            categories['unknown_operations'].append(func)
    
    # Show category summaries
    for category, funcs in categories.items():
        if funcs:
            total_category_time = sum(f['cumtime'] for f in funcs)
            percentage = (total_category_time / total_time) * 100
            count = len(funcs)
            
            print(f"\n📁 {category.replace('_', ' ').title()}:")
            print(f"   Total time: {total_category_time:.3f}s ({percentage:.1f}%)")
            print(f"   Function count: {count}")
            
            # Show top 3 in this category
            sorted_funcs = sorted(funcs, key=lambda x: x['cumtime'], reverse=True)
            for i, func in enumerate(sorted_funcs[:3]):
                print(f"   {i+1}. {func['cumtime']:6.3f}s - {func['function'][:60]}...")

def check_for_hidden_synchronization():
    """Check if there's any hidden synchronization we missed."""
    print(f"\n🔍 CHECKING FOR HIDDEN SYNCHRONIZATION")
    print("=" * 45)
    
    try:
        # Check if any modules are importing threading/async stuff
        import src.brains.field.generic_brain as field_brain_module
        
        # Check module attributes for threading/async
        suspicious_attrs = []
        for attr_name in dir(field_brain_module):
            attr = getattr(field_brain_module, attr_name)
            if hasattr(attr, '__module__'):
                module_name = getattr(attr, '__module__', '')
                if any(keyword in module_name for keyword in ['thread', 'async', 'queue', 'lock']):
                    suspicious_attrs.append((attr_name, module_name))
        
        if suspicious_attrs:
            print("⚠️ Found potential synchronization:")
            for attr_name, module_name in suspicious_attrs:
                print(f"   {attr_name} from {module_name}")
        else:
            print("✅ No obvious synchronization imports in field brain")
        
        # Check for time.sleep calls (common hidden delay)
        import inspect
        source = inspect.getsource(field_brain_module)
        if 'time.sleep' in source:
            print("⚠️ Found time.sleep calls in field brain!")
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'time.sleep' in line:
                    print(f"   Line {i+1}: {line.strip()}")
        else:
            print("✅ No time.sleep calls found")
            
    except Exception as e:
        print(f"❌ Synchronization check failed: {e}")

def main():
    """Run detailed profiling of minimal field brain."""
    print("🚀 MINIMAL FIELD BRAIN DEEP PROFILER")
    print("=" * 60)
    print("Goal: Understand what's consuming 5.568 seconds")
    print("Expected: Pure mathematics and justified computations")
    print()
    
    # Profile the minimal brain
    processing_time = profile_minimal_field_brain()
    
    # Check for hidden synchronization
    check_for_hidden_synchronization()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("🎯 PROFILING SUMMARY")
    print("=" * 60)
    
    if processing_time:
        print(f"📊 Processing time: {processing_time:.3f}s")
        
        if processing_time > 5.0:
            print("🔍 Analysis: Still significant computation time")
            print("   Likely causes:")
            print("   1. Heavy field mathematics (torch operations)")
            print("   2. Large field dimensions creating O(n²) or O(n³) operations")
            print("   3. Constraint discovery algorithms")
            print("   4. Hidden complexity in field evolution")
        elif processing_time > 1.0:
            print("🔧 Analysis: Moderate computation time")
            print("   Field brain doing substantial work but might be optimizable")
        else:
            print("✅ Analysis: Reasonable computation time")
    
    print(f"\n🧠 Next steps:")
    print("   1. Review detailed profile in logs/minimal_field_brain_profile.txt")
    print("   2. Identify the heaviest mathematical operations")
    print("   3. Implement sparse field updates for O(n) complexity reduction")
    print("   4. Consider approximation algorithms for heavy computations")

if __name__ == "__main__":
    main()
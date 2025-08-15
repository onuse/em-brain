#!/usr/bin/env python3
"""
Field Brain Performance Profiler

Deep profiling of field brain processing to identify bottlenecks and guide
implementation of biological optimization shortcuts.

This will help us understand where the 62+ second processing time is coming from
and identify opportunities for "biological hacks" that real brains use.
"""

import sys
import os
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

class FieldBrainProfiler:
    """Comprehensive profiler for field brain performance analysis."""
    
    def __init__(self):
        self.results = {}
        self.config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 20,
                "field_temporal_window": 10.0,
                "field_evolution_rate": 0.1,
                "constraint_discovery_rate": 0.15
            },
            "memory": {"enable_persistence": False}  # Disable for clean profiling
        }
    
    def profile_brain_creation(self):
        """Profile brain instantiation time."""
        print("🔍 Profiling brain creation...")
        
        from src.brain import MinimalBrain
        
        # Profile the creation process
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        brain = MinimalBrain(config=self.config, quiet_mode=True)
        
        profiler.disable()
        creation_time = time.time() - start_time
        
        # Analyze profiling results
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(20)  # Top 20 functions
        
        self.results['brain_creation'] = {
            'total_time': creation_time,
            'profile_data': s.getvalue()
        }
        
        print(f"   ✅ Brain creation: {creation_time:.3f}s")
        
        # Cleanup
        brain.finalize_session()
        return brain
    
    def profile_single_processing(self):
        """Profile a single sensory processing cycle in detail."""
        print("\n🔄 Profiling single processing cycle...")
        
        from src.brain import MinimalBrain
        
        # Create brain
        brain = MinimalBrain(config=self.config, quiet_mode=True)
        
        # Test sensory input
        sensory_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        
        # Profile the processing
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        action, brain_state = brain.process_sensory_input(sensory_input)
        
        profiler.disable()
        processing_time = time.time() - start_time
        
        # Analyze profiling results
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(30)  # Top 30 functions
        
        self.results['single_processing'] = {
            'total_time': processing_time,
            'output_size': len(action),
            'profile_data': s.getvalue()
        }
        
        print(f"   ✅ Single processing: {processing_time:.3f}s")
        print(f"   Output: {len(action)}D action vector")
        
        # Cleanup
        brain.finalize_session()
    
    def profile_field_components(self):
        """Profile individual field brain components."""
        print("\n🧩 Profiling field brain components...")
        
        try:
            # Direct access to field brain components
            from src.brains.field.generic_field_brain import GenericFieldBrain
            from src.brains.field.field_brain_config import FieldBrainConfig
            
            # Create field brain configuration
            field_config = FieldBrainConfig(
                sensory_dimensions=16,
                motor_dimensions=4,
                spatial_resolution=20,
                temporal_window=10.0,
                field_evolution_rate=0.1,
                constraint_discovery_rate=0.15,
                quiet_mode=True
            )
            
            # Profile field brain creation
            print("   🏗️ Profiling field brain instantiation...")
            start_time = time.time()
            field_brain = GenericFieldBrain(field_config)
            field_creation_time = time.time() - start_time
            print(f"      ✅ Field brain created: {field_creation_time:.3f}s")
            
            # Profile capability negotiation
            print("   🤝 Profiling capability negotiation...")
            from src.brains.shared.stream_types import StreamCapabilities
            
            capabilities = StreamCapabilities(
                sensory_dimensions=16,
                motor_dimensions=4,
                supports_temporal=True,
                supports_batching=False,
                max_batch_size=1,
                processing_latency_ms=50.0
            )
            
            start_time = time.time()
            field_brain.negotiate_stream_capabilities(capabilities)
            negotiation_time = time.time() - start_time
            print(f"      ✅ Capability negotiation: {negotiation_time:.3f}s")
            
            # Profile stream processing (the main bottleneck candidate)
            print("   🌊 Profiling stream processing...")
            input_stream = [0.1] * 16
            
            # Use cProfile for detailed analysis
            profiler = cProfile.Profile()
            
            start_time = time.time()
            profiler.enable()
            
            output_stream, brain_state = field_brain.process_input_stream(input_stream)
            
            profiler.disable()
            stream_processing_time = time.time() - start_time
            
            print(f"      ✅ Stream processing: {stream_processing_time:.3f}s")
            
            # Analyze stream processing profile
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats(40)  # Top 40 functions for detailed analysis
            
            self.results['field_components'] = {
                'field_creation_time': field_creation_time,
                'negotiation_time': negotiation_time,
                'stream_processing_time': stream_processing_time,
                'stream_profile_data': s.getvalue()
            }
            
        except Exception as e:
            print(f"   ❌ Component profiling failed: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_bottlenecks(self):
        """Analyze profiling results to identify bottlenecks."""
        print("\n📊 Analyzing performance bottlenecks...")
        
        if 'single_processing' not in self.results:
            print("   ❌ No processing data to analyze")
            return
        
        processing_time = self.results['single_processing']['total_time']
        profile_data = self.results['single_processing']['profile_data']
        
        print(f"\n🎯 Key Performance Findings:")
        print(f"   Total processing time: {processing_time:.3f}s")
        
        # Analyze profile data for common bottlenecks
        lines = profile_data.split('\n')
        
        # Look for specific performance patterns
        field_operations = []
        torch_operations = []
        math_operations = []
        io_operations = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['field', 'dynamics', 'unified']):
                field_operations.append(line.strip())
            elif any(keyword in line.lower() for keyword in ['torch', 'tensor', 'cuda']):
                torch_operations.append(line.strip())
            elif any(keyword in line.lower() for keyword in ['math', 'numpy', 'calculation']):
                math_operations.append(line.strip())
            elif any(keyword in line.lower() for keyword in ['file', 'write', 'read', 'save']):
                io_operations.append(line.strip())
        
        print(f"\n🔍 Bottleneck Categories Found:")
        print(f"   Field operations: {len(field_operations)}")
        print(f"   Torch operations: {len(torch_operations)}")
        print(f"   Math operations: {len(math_operations)}")
        print(f"   I/O operations: {len(io_operations)}")
        
        # Show top field operations
        if field_operations:
            print(f"\n🌊 Top Field Operations:")
            for i, op in enumerate(field_operations[:5]):
                if op.strip():
                    print(f"   {i+1}. {op[:80]}...")
        
        # Biological optimization suggestions
        self.suggest_biological_optimizations(processing_time)
    
    def suggest_biological_optimizations(self, processing_time: float):
        """Suggest biological optimization strategies based on findings."""
        print(f"\n🧠 Biological Optimization Recommendations:")
        print(f"=" * 50)
        
        if processing_time > 30.0:
            print("🚨 CRITICAL PERFORMANCE ISSUE (>30s processing)")
            print("   Biological Reality: Real neurons process in ~1ms")
            print("   Required: Aggressive biological shortcuts")
            print()
            print("🎯 Priority Optimizations:")
            print("   1. **Sparse Field Updates**: Only update 'interesting' regions")
            print("      - Real brains don't update every neuron every cycle")
            print("      - Implement attention-based sparse updates")
            print("      - Target: 90% reduction in field computations")
            print()
            print("   2. **Hierarchical Processing**: Coarse-to-fine resolution")
            print("      - Start with low-resolution field approximation")
            print("      - Refine only high-activity regions")
            print("      - Target: 80% reduction in computation")
            print()
            print("   3. **Predictive Field States**: Cache common patterns")
            print("      - Pre-compute frequent field configurations")
            print("      - Use temporal momentum for predictions")
            print("      - Target: 70% cache hit rate")
            print()
            print("   4. **Regional Specialization**: Optimize by dynamics family")
            print("      - Different algorithms for oscillatory vs flow dynamics")
            print("      - Specialized hardware/computation paths")
            print("      - Target: 60% efficiency gain per family")
            
        elif processing_time > 5.0:
            print("⚠️ MODERATE PERFORMANCE ISSUE (5-30s processing)")
            print("   Focus on field computation optimizations")
            
        elif processing_time > 1.0:
            print("🔧 MINOR PERFORMANCE ISSUE (1-5s processing)")
            print("   Fine-tuning and caching optimizations")
            
        else:
            print("✅ ACCEPTABLE PERFORMANCE (<1s processing)")
            print("   Minor optimizations for production readiness")
    
    def save_results(self):
        """Save profiling results to file."""
        output_file = Path(__file__).parent.parent.parent / "logs" / "field_brain_profile_analysis.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {k: v for k, v in value.items() if k != 'profile_data'}
                # Save profile data separately as text
                if 'profile_data' in value:
                    profile_file = output_file.parent / f"profile_{key}.txt"
                    with open(profile_file, 'w') as f:
                        f.write(value['profile_data'])
                    json_results[key]['profile_file'] = str(profile_file)
            else:
                json_results[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")

def main():
    """Run comprehensive field brain profiling."""
    print("🚀 FIELD BRAIN PERFORMANCE PROFILER")
    print("=" * 60)
    print("Analyzing where the 62+ second processing time comes from...")
    print("This will guide implementation of biological optimization shortcuts.")
    print()
    
    profiler = FieldBrainProfiler()
    
    try:
        # Run profiling steps
        profiler.profile_brain_creation()
        profiler.profile_single_processing()
        profiler.profile_field_components()
        profiler.analyze_bottlenecks()
        profiler.save_results()
        
        print(f"\n{'=' * 60}")
        print("📋 PROFILING COMPLETE!")
        print("=" * 60)
        print("🎯 Next Steps:")
        print("   1. Review detailed profile data in logs/")
        print("   2. Implement biological optimization shortcuts")
        print("   3. Focus on sparse field updates and caching")
        print("   4. Test performance improvements")
        print()
        print("🧠 Remember: Real brains use 'quick n dirty' shortcuts")
        print("   These aren't just optimizations - they create intelligence!")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
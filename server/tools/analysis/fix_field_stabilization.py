#!/usr/bin/env python3
"""
Fix Field Stabilization Test

The current test fails because it expects variance to decrease from early to late.
But if early variance is 0, the test fails. We need a better stability metric.
"""

import numpy as np


def original_field_stabilization(energy_levels):
    """Original test that fails"""
    if len(energy_levels) >= 10:
        early_variance = np.var(energy_levels[:10])
        late_variance = np.var(energy_levels[-10:])
        
        if early_variance > 0:
            stability_improvement = 1.0 - (late_variance / early_variance)
            return max(0, min(1.0, stability_improvement))
    
    return 0.5


def improved_field_stabilization(energy_levels):
    """
    Improved test that measures actual stability.
    
    Stability is measured as:
    1. Low variance in the last portion (absolute stability)
    2. Convergence trend (energy settling)
    3. Reduced volatility over time
    """
    if len(energy_levels) < 20:
        return 0.5
    
    # Convert to numpy array
    energy = np.array(energy_levels)
    
    # 1. Absolute stability in the last quarter
    last_quarter = energy[-len(energy)//4:]
    mean_energy = np.mean(last_quarter)
    if mean_energy > 0:
        cv_last = np.std(last_quarter) / mean_energy  # Coefficient of variation
        stability_score = 1.0 / (1.0 + cv_last * 10)  # Lower CV = higher score
    else:
        stability_score = 0.5
    
    # 2. Convergence trend - compare quarters
    quarters = np.array_split(energy, 4)
    quarter_stds = [np.std(q) for q in quarters]
    
    # Check if volatility decreases
    if len(quarter_stds) >= 4:
        early_volatility = np.mean(quarter_stds[:2])
        late_volatility = np.mean(quarter_stds[-2:])
        
        if early_volatility > 0:
            convergence_score = max(0, 1.0 - late_volatility / early_volatility)
        else:
            # If started stable, check if it stayed stable
            convergence_score = 1.0 if late_volatility < 0.001 else 0.5
    else:
        convergence_score = 0.5
    
    # 3. Overall volatility reduction
    window = max(5, len(energy) // 10)
    rolling_std = [np.std(energy[i:i+window]) 
                   for i in range(0, len(energy) - window, window//2)]
    
    if len(rolling_std) >= 2:
        trend = np.polyfit(range(len(rolling_std)), rolling_std, 1)[0]
        # Negative trend (decreasing volatility) is good
        trend_score = 1.0 / (1.0 + np.exp(trend * 100))
    else:
        trend_score = 0.5
    
    # Combine scores with weights
    final_score = (
        0.4 * stability_score +      # Absolute stability matters most
        0.3 * convergence_score +    # Convergence is important
        0.3 * trend_score           # Overall trend
    )
    
    return final_score


def test_with_sample_data():
    """Test with the actual data from debug"""
    # Sample energy values from the debug output
    energy_data = [0.001003, 0.001003, 0.001003, 0.001003, 0.001003,
                   0.001003, 0.001003, 0.001003, 0.001003, 0.001003,  # First 10: all same
                   0.001122, 0.001150, 0.001175, 0.001200, 0.001215,
                   0.001225, 0.001230, 0.001235, 0.001240, 0.001225,  # Rising
                   0.001225, 0.001042, 0.000900, 0.000800, 0.000700,
                   0.000600, 0.000500, 0.000450, 0.000420, 0.000400,  # Dropping
                   0.000400, 0.000380, 0.000370, 0.000365, 0.000363,  # Stabilizing low
                   0.000363, 0.000363, 0.000363, 0.000363, 0.000363]  # Last 10: stable
    
    print("Testing field stabilization metrics:")
    print("-" * 50)
    
    # Original test
    orig_score = original_field_stabilization(energy_data)
    print(f"Original test score: {orig_score:.3f}")
    print(f"  (Fails because early variance = 0)")
    
    # Improved test
    improved_score = improved_field_stabilization(energy_data)
    print(f"\nImproved test score: {improved_score:.3f}")
    
    # Show components
    energy = np.array(energy_data)
    last_quarter = energy[-len(energy)//4:]
    cv_last = np.std(last_quarter) / np.mean(last_quarter)
    print(f"\nComponents:")
    print(f"  Last quarter CV: {cv_last:.4f}")
    print(f"  Energy range: {np.min(energy):.6f} to {np.max(energy):.6f}")
    print(f"  Final stability: std={np.std(last_quarter):.6f}")


if __name__ == "__main__":
    test_with_sample_data()
    
    print("\n\nRecommendation:")
    print("Replace the field_stabilization test with the improved version")
    print("that measures absolute stability rather than relative improvement.")
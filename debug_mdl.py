#!/usr/bin/env python3
"""
Debug script to understand why controls are passing
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.os_reality_protocol import OSRealityProtocol
from protocols.tier_implementations.t0_compression import T0CompressionTest
from core.primitives import DimensionalState


def debug_mdl_calculation():
    """Debug the MDL calculation to understand the issue"""
    
    print("="*60)
    print("MDL CALCULATION DEBUG")
    print("="*60)
    
    # Setup a simple test
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Get the T0 test
    t0 = T0CompressionTest(threshold=0.10)
    
    # Test 1: Real data
    print("\n1. REAL DATA TEST:")
    result_real = t0.run_test(protocol.domain_adapters, protocol.sequences)
    print(f"   MDL Advantage: {result_real['mdl_advantage']:.3f}")
    print(f"   Unified MDL: {result_real['unified_mdl']:.3f}")
    print(f"   Separate MDL: {result_real['separate_mdl']:.3f}")
    print(f"   Unified params: {result_real['details']['unified_params']}")
    print(f"   Separate params: {result_real['details']['separate_params']}")
    
    # Test 2: Random data
    print("\n2. RANDOM DATA TEST:")
    for domain_name, sequences in protocol.sequences.items():
        for i, seq in enumerate(sequences):
            protocol.sequences[domain_name][i] = np.random.randn(*seq.shape)
    
    result_random = t0.run_test(protocol.domain_adapters, protocol.sequences)
    print(f"   MDL Advantage: {result_random['mdl_advantage']:.3f}")
    print(f"   Unified MDL: {result_random['unified_mdl']:.3f}")
    print(f"   Separate MDL: {result_random['separate_mdl']:.3f}")
    
    # Test 3: Analyze what MDL is actually measuring
    print("\n3. MDL BREAKDOWN:")
    
    # Get a unified model
    unified_model = t0._train_unified_model(protocol.domain_adapters, protocol.sequences)
    print(f"\n   Unified Model:")
    print(f"   - Params: {unified_model['param_count']}")
    print(f"   - MSE: {unified_model['mse']:.6f}")
    
    # Train separate models
    print(f"\n   Separate Models:")
    for domain_name, adapter in protocol.domain_adapters.items():
        if domain_name in protocol.sequences:
            model = t0._train_domain_model(adapter, protocol.sequences[domain_name])
            print(f"   - {domain_name}: params={model['param_count']}, mse={model['mse']:.6f}")
    
    # Test 4: What happens with identical data across domains?
    print("\n4. IDENTICAL DATA TEST:")
    identical_seq = np.random.randn(900, 3)  # Same sequence
    for domain_name in protocol.sequences.keys():
        protocol.sequences[domain_name] = [identical_seq.copy() for _ in range(3)]
    
    result_identical = t0.run_test(protocol.domain_adapters, protocol.sequences)
    print(f"   MDL Advantage: {result_identical['mdl_advantage']:.3f}")
    print(f"   Unified MDL: {result_identical['unified_mdl']:.3f}")
    print(f"   Separate MDL: {result_identical['separate_mdl']:.3f}")
    
    # Test 5: Examine the actual MDL calculation
    print("\n5. MDL CALCULATION ANALYSIS:")
    
    # Create a simple test case
    simple_model = {'param_count': 3, 'mse': 0.1, 'params': np.array([1, 2, 3])}
    mdl = t0._calculate_mdl(simple_model, protocol.sequences)
    
    print(f"   Simple model MDL: {mdl:.3f}")
    print(f"   Param cost: {simple_model['param_count'] * np.log2(len(simple_model['params']) + 1):.3f}")
    print(f"   Data cost: {-np.log2(1.0 / (simple_model['mse'] + 1e-10)):.3f}")
    
    # The issue might be here!
    print("\n6. THE PROBLEM:")
    print("   MDL = param_cost + data_cost")
    print("   param_cost = param_count * log2(param_count + 1)")
    print("   data_cost = -log2(1 / (mse + epsilon))")
    print()
    print("   For unified (3 params):")
    print(f"     param_cost = 3 * log2(4) = {3 * np.log2(4):.3f}")
    print("   For separate (9 params total, 3 each):")
    print(f"     param_cost = 3 * 3 * log2(4) = {9 * np.log2(4):.3f}")
    print()
    print("   The param_cost difference dominates!")
    print(f"   Difference: {9 * np.log2(4) - 3 * np.log2(4):.3f}")
    print()
    print("   This means ANY unified model with 3 params will beat")
    print("   separate models with 9 params, regardless of data!")
    
    # Test our hypothesis
    print("\n7. TESTING THE HYPOTHESIS:")
    print("   If param count dominates, then:")
    print("   - Random data should show similar advantage ✓")
    print("   - Noise should not hurt advantage ✓")
    print("   - Controls should pass ✓")
    print()
    print("   This is EXACTLY what we're seeing!")
    
    return result_real, result_random, result_identical


def analyze_encoding():
    """Analyze how sequences are encoded"""
    print("\n" + "="*60)
    print("SEQUENCE ENCODING ANALYSIS")
    print("="*60)
    
    # Create a simple sequence
    test_seq = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    print("\nOriginal sequence shape:", test_seq.shape)
    print("Original sequence:\n", test_seq)
    
    # How T0 encodes it
    state = DimensionalState(test_seq.shape[1], seed=42)
    state.state_vector = np.mean(test_seq, axis=0)  # Just takes mean!
    
    print("\nEncoded as:", state.state_vector)
    print("Information lost: Everything except the mean!")
    
    # This means:
    print("\nIMPLICATIONS:")
    print("1. All temporal information is lost (just takes mean)")
    print("2. All sequences become 3 numbers (one per channel)")
    print("3. Model just predicts variance from 3 means")
    print("4. This is trivial regression, not sequence modeling!")


if __name__ == "__main__":
    # Run debugging
    real, random, identical = debug_mdl_calculation()
    analyze_encoding()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The T0 test is fundamentally flawed because:")
    print("1. It reduces sequences to their means (massive info loss)")
    print("2. MDL is dominated by parameter count, not actual compression")
    print("3. Unified always has fewer params (3 vs 9) by design")
    print("4. This is why controls pass - the math doesn't care about data!")
    print("\nThe test measures parameter efficiency, not domain unification!")
    print("="*60)
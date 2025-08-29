#!/usr/bin/env python3
"""
Test the fixed T0 compression implementation
Verify that controls properly fail while real patterns pass
"""

import numpy as np
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.os_reality_protocol import OSRealityProtocol
from protocols.tier_implementations.t0_compression_fixed import T0CompressionTestFixed, T0CompressionTestSimple
from protocols.tier_implementations.t0_compression import T0CompressionTest  # Original flawed version


def test_controls_with_fixed_implementation():
    """Test that controls properly fail with the fixed implementation"""
    
    print("="*80)
    print("TESTING FIXED T0 COMPRESSION")
    print("="*80)
    
    # Initialize protocol
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Test both old and new implementations
    t0_old = T0CompressionTest(threshold=0.10)
    t0_fixed = T0CompressionTestFixed(threshold=0.10)
    t0_simple = T0CompressionTestSimple(threshold=0.10)
    
    # Test 1: Real data
    print("\n1. REAL DATA TEST:")
    print("-" * 40)
    
    result_old = t0_old.run_test(protocol.domain_adapters, protocol.sequences)
    print(f"OLD Implementation:")
    print(f"  MDL Advantage: {result_old['mdl_advantage']:.3f}")
    print(f"  Passed: {result_old['passed']}")
    
    result_fixed = t0_fixed.run_test(protocol.domain_adapters, protocol.sequences)
    print(f"\nFIXED Implementation:")
    print(f"  MDL Advantage: {result_fixed['mdl_advantage']:.3f}")
    print(f"  Passed: {result_fixed['passed']}")
    
    result_simple = t0_simple.run_test(protocol.domain_adapters, protocol.sequences)
    print(f"\nSIMPLE Implementation:")
    print(f"  MDL Advantage: {result_simple['mdl_advantage']:.3f}")
    print(f"  Passed: {result_simple['passed']}")
    
    # Test 2: Shuffled sequences (should FAIL)
    print("\n2. SHUFFLED SEQUENCES (should FAIL):")
    print("-" * 40)
    
    shuffled_sequences = {}
    for domain_name, sequences in protocol.sequences.items():
        shuffled = []
        for seq in sequences:
            # Shuffle time points
            shuffled_seq = seq.copy()
            np.random.shuffle(shuffled_seq)
            shuffled.append(shuffled_seq)
        shuffled_sequences[domain_name] = shuffled
    
    result_old_shuffle = t0_old.run_test(protocol.domain_adapters, shuffled_sequences)
    print(f"OLD Implementation:")
    print(f"  MDL Advantage: {result_old_shuffle['mdl_advantage']:.3f}")
    print(f"  Passed: {result_old_shuffle['passed']} ❌ (should be False)")
    
    result_fixed_shuffle = t0_fixed.run_test(protocol.domain_adapters, shuffled_sequences)
    print(f"\nFIXED Implementation:")
    print(f"  MDL Advantage: {result_fixed_shuffle['mdl_advantage']:.3f}")
    print(f"  Passed: {result_fixed_shuffle['passed']} {'✓' if not result_fixed_shuffle['passed'] else '❌'}")
    
    result_simple_shuffle = t0_simple.run_test(protocol.domain_adapters, shuffled_sequences)
    print(f"\nSIMPLE Implementation:")
    print(f"  MDL Advantage: {result_simple_shuffle['mdl_advantage']:.3f}")
    print(f"  Passed: {result_simple_shuffle['passed']} {'✓' if not result_simple_shuffle['passed'] else '❌'}")
    
    # Test 3: Random noise (should FAIL)
    print("\n3. RANDOM NOISE (should FAIL):")
    print("-" * 40)
    
    random_sequences = {}
    for domain_name, sequences in protocol.sequences.items():
        random_seqs = []
        for seq in sequences:
            random_seq = np.random.randn(*seq.shape)
            random_seqs.append(random_seq)
        random_sequences[domain_name] = random_seqs
    
    result_old_random = t0_old.run_test(protocol.domain_adapters, random_sequences)
    print(f"OLD Implementation:")
    print(f"  MDL Advantage: {result_old_random['mdl_advantage']:.3f}")
    print(f"  Passed: {result_old_random['passed']} ❌ (should be False)")
    
    result_fixed_random = t0_fixed.run_test(protocol.domain_adapters, random_sequences)
    print(f"\nFIXED Implementation:")
    print(f"  MDL Advantage: {result_fixed_random['mdl_advantage']:.3f}")
    print(f"  Passed: {result_fixed_random['passed']} {'✓' if not result_fixed_random['passed'] else '❌'}")
    
    result_simple_random = t0_simple.run_test(protocol.domain_adapters, random_sequences)
    print(f"\nSIMPLE Implementation:")
    print(f"  MDL Advantage: {result_simple_random['mdl_advantage']:.3f}")
    print(f"  Passed: {result_simple_random['passed']} {'✓' if not result_simple_random['passed'] else '❌'}")
    
    # Test 4: Identical data across domains (should PASS strongly)
    print("\n4. IDENTICAL DATA ACROSS DOMAINS (should PASS strongly):")
    print("-" * 40)
    
    # Create shared pattern
    shared_pattern = np.sin(np.linspace(0, 10*np.pi, 900))[:, np.newaxis]
    shared_pattern = np.hstack([shared_pattern, 
                                np.cos(np.linspace(0, 10*np.pi, 900))[:, np.newaxis],
                                np.sin(np.linspace(0, 5*np.pi, 900))[:, np.newaxis]])
    
    identical_sequences = {}
    for domain_name in protocol.sequences.keys():
        # Add small domain-specific noise
        domain_seqs = []
        for i in range(3):
            seq = shared_pattern + np.random.randn(*shared_pattern.shape) * 0.01
            domain_seqs.append(seq)
        identical_sequences[domain_name] = domain_seqs
    
    result_old_identical = t0_old.run_test(protocol.domain_adapters, identical_sequences)
    print(f"OLD Implementation:")
    print(f"  MDL Advantage: {result_old_identical['mdl_advantage']:.3f}")
    print(f"  Passed: {result_old_identical['passed']}")
    
    result_fixed_identical = t0_fixed.run_test(protocol.domain_adapters, identical_sequences)
    print(f"\nFIXED Implementation:")
    print(f"  MDL Advantage: {result_fixed_identical['mdl_advantage']:.3f}")
    print(f"  Passed: {result_fixed_identical['passed']}")
    
    result_simple_identical = t0_simple.run_test(protocol.domain_adapters, identical_sequences)
    print(f"\nSIMPLE Implementation:")
    print(f"  MDL Advantage: {result_simple_identical['mdl_advantage']:.3f}")
    print(f"  Passed: {result_simple_identical['passed']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Check if fixed implementations properly discriminate
    old_discriminates = False  # Old always passes everything
    fixed_discriminates = (result_fixed_identical['passed'] and 
                          not result_fixed_shuffle['passed'] and 
                          not result_fixed_random['passed'])
    simple_discriminates = (result_simple_identical['passed'] and 
                           not result_simple_shuffle['passed'] and 
                           not result_simple_random['passed'])
    
    print(f"\nOLD Implementation discriminates properly: {old_discriminates} ❌")
    print(f"FIXED Implementation discriminates properly: {fixed_discriminates} {'✓' if fixed_discriminates else '❌'}")
    print(f"SIMPLE Implementation discriminates properly: {simple_discriminates} {'✓' if simple_discriminates else '❌'}")
    
    if fixed_discriminates or simple_discriminates:
        print("\n✓ SUCCESS: Fixed implementation(s) properly distinguish real patterns from noise!")
        print("  - Identical data passes (shared structure detected)")
        print("  - Random noise fails (no structure to compress)")
        print("  - Shuffled data fails (temporal structure destroyed)")
    else:
        print("\n❌ PROBLEM: Fixed implementations still not discriminating properly")
        print("  Need to investigate further...")
    
    return {
        'old': (result_old, result_old_shuffle, result_old_random, result_old_identical),
        'fixed': (result_fixed, result_fixed_shuffle, result_fixed_random, result_fixed_identical),
        'simple': (result_simple, result_simple_shuffle, result_simple_random, result_simple_identical)
    }


if __name__ == "__main__":
    results = test_controls_with_fixed_implementation()
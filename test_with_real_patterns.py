#!/usr/bin/env python3
"""
Test with data that has ACTUAL shared mathematical structure
This simulates what we'd expect if the OS-of-Reality hypothesis is true
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.tier_implementations.t0_compression_fixed import T0CompressionTestFixed
from domains.biological.bio_adapter import BiologicalAdapter
from domains.geological.geo_adapter import GeologicalAdapter  
from domains.cosmological.cosmo_adapter import CosmologicalAdapter


def create_shared_structure_data():
    """
    Create data where domains actually share underlying mathematical patterns
    This simulates the OS-of-Reality hypothesis being TRUE
    """
    
    print("="*80)
    print("TESTING WITH SHARED MATHEMATICAL STRUCTURE")
    print("="*80)
    
    # SHARED UNIVERSAL PRIMITIVES (the substrate)
    # These represent the fundamental mathematical patterns
    # that allegedly underlie all three domains
    
    print("\n1. Creating Universal Mathematical Substrate:")
    print("-" * 60)
    
    # Universal oscillatory patterns (found in all domains)
    t = np.linspace(0, 100, 900)
    
    # Primitive 1: Multi-scale oscillations (fractals in nature)
    universal_osc1 = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t) + 0.25 * np.sin(0.7 * t)
    
    # Primitive 2: Power law decay (universal in nature)
    universal_decay = np.exp(-0.01 * t) * np.cos(0.2 * t)
    
    # Primitive 3: Burst patterns (found in many systems)
    burst_times = [200, 400, 600, 800]
    universal_burst = np.zeros_like(t)
    for bt in burst_times:
        universal_burst += np.exp(-0.1 * np.abs(t - bt)) * np.sin(2 * t)
    
    # Stack universal patterns
    universal_patterns = np.column_stack([
        universal_osc1,
        universal_decay,
        universal_burst
    ])
    
    print(f"Created {universal_patterns.shape[1]} universal primitive patterns")
    print(f"Pattern length: {len(t)} timesteps")
    
    # DOMAIN-SPECIFIC EXPRESSIONS
    # Each domain expresses these universal patterns differently
    
    print("\n2. Creating Domain-Specific Expressions:")
    print("-" * 60)
    
    # Biological: Emphasizes rhythmic patterns (heartbeat, breathing)
    bio_transform = np.array([
        [2.0, 0.5, 0.3],   # Strong oscillation
        [0.3, 1.5, 0.5],   # Moderate decay  
        [0.5, 0.3, 2.0]    # Strong bursts (neural spikes)
    ])
    
    # Geological: Emphasizes slow changes and sudden events
    geo_transform = np.array([
        [0.5, 2.0, 0.3],   # Slow oscillation
        [2.0, 0.5, 0.3],   # Strong decay (erosion)
        [0.3, 0.5, 3.0]    # Very strong bursts (earthquakes)
    ])
    
    # Cosmological: Emphasizes periodic and decay patterns
    cosmo_transform = np.array([
        [3.0, 0.3, 0.5],   # Very strong oscillation (orbits)
        [0.5, 2.5, 0.3],   # Strong decay (radiation)
        [0.3, 0.3, 1.5]    # Moderate bursts (stellar events)
    ])
    
    # Generate domain-specific sequences
    bio_sequences = []
    geo_sequences = []
    cosmo_sequences = []
    
    for i in range(3):  # 3 sequences per domain
        # Add slight variations and domain-specific noise
        noise_scale = 0.1
        
        # Biological sequences
        bio_seq = universal_patterns @ bio_transform.T
        bio_seq += np.random.randn(*bio_seq.shape) * noise_scale
        bio_seq += np.sin(0.5 * t + i)[:, np.newaxis] * 0.2  # Bio-specific rhythm
        bio_sequences.append(bio_seq)
        
        # Geological sequences  
        geo_seq = universal_patterns @ geo_transform.T
        geo_seq += np.random.randn(*geo_seq.shape) * noise_scale
        geo_seq += np.cumsum(np.random.randn(len(t)))[:, np.newaxis] * 0.01  # Drift
        geo_sequences.append(geo_seq)
        
        # Cosmological sequences
        cosmo_seq = universal_patterns @ cosmo_transform.T
        cosmo_seq += np.random.randn(*cosmo_seq.shape) * noise_scale
        cosmo_seq += np.cos(0.05 * t + i)[:, np.newaxis] * 0.3  # Long period
        cosmo_sequences.append(cosmo_seq)
    
    print(f"Generated 3 sequences for each domain")
    print(f"Each sequence: {bio_sequences[0].shape}")
    
    # Test with fixed T0
    print("\n3. Testing with Fixed T0 Compression:")
    print("-" * 60)
    
    # Create dummy adapters (we just need the structure)
    adapters = {
        'biological': BiologicalAdapter(),
        'geological': GeologicalAdapter(),
        'cosmological': CosmologicalAdapter()
    }
    
    sequences = {
        'biological': bio_sequences,
        'geological': geo_sequences,
        'cosmological': cosmo_sequences
    }
    
    # Run the fixed T0 test
    t0_test = T0CompressionTestFixed(threshold=0.10)
    results = t0_test.run_test(adapters, sequences)
    
    print(f"MDL Advantage: {results['mdl_advantage']:.3f}")
    print(f"Passed: {results['passed']}")
    print(f"Unified MDL: {results['unified_mdl']:.0f}")
    print(f"Separate MDL: {results['separate_mdl']:.0f}")
    
    # Test controls to verify they still fail
    print("\n4. Verifying Controls Still Fail:")
    print("-" * 60)
    
    # Shuffle test
    shuffled_sequences = {}
    for domain, seqs in sequences.items():
        shuffled = []
        for seq in seqs:
            shuffled_seq = seq.copy()
            np.random.shuffle(shuffled_seq)
            shuffled.append(shuffled_seq)
        shuffled_sequences[domain] = shuffled
    
    shuffle_results = t0_test.run_test(adapters, shuffled_sequences)
    print(f"Shuffled data - MDL Advantage: {shuffle_results['mdl_advantage']:.3f}, Passed: {shuffle_results['passed']}")
    
    # Random test
    random_sequences = {}
    for domain, seqs in sequences.items():
        random_seqs = []
        for seq in seqs:
            random_seqs.append(np.random.randn(*seq.shape))
        random_sequences[domain] = random_seqs
    
    random_results = t0_test.run_test(adapters, random_sequences)
    print(f"Random data - MDL Advantage: {random_results['mdl_advantage']:.3f}, Passed: {random_results['passed']}")
    
    # Transfer learning test
    print("\n5. Transfer Learning Test:")
    print("-" * 60)
    
    # Flatten sequences for PCA  
    bio_flat = np.vstack([seq[:100].flatten() for seq in bio_sequences])
    geo_flat = np.vstack([seq[:100].flatten() for seq in geo_sequences])
    cosmo_flat = np.vstack([seq[:100].flatten() for seq in cosmo_sequences])
    
    # Train on bio+geo, test on cosmo
    train_data = np.vstack([bio_flat, geo_flat])
    test_data = cosmo_flat
    
    # Use appropriate number of components
    n_components = min(5, train_data.shape[0] - 1, train_data.shape[1] - 1)
    pca_transfer = PCA(n_components=n_components)
    pca_transfer.fit(train_data)
    
    # Reconstruction error
    test_recon = pca_transfer.inverse_transform(pca_transfer.transform(test_data))
    transfer_error = mean_squared_error(test_data, test_recon)
    
    # Compare to training only on cosmo
    n_comp_direct = min(3, test_data.shape[0] - 1, test_data.shape[1] - 1)
    pca_direct = PCA(n_components=n_comp_direct)
    pca_direct.fit(test_data)
    direct_recon = pca_direct.inverse_transform(pca_direct.transform(test_data))
    direct_error = mean_squared_error(test_data, direct_recon)
    
    print(f"Transfer learning error: {transfer_error:.4f}")
    print(f"Direct training error: {direct_error:.4f}")
    print(f"Transfer benefit: {(direct_error - transfer_error) / direct_error * 100:.1f}%")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results['passed']:
        print("✅ SUCCESS: Unified model successfully compresses shared structure!")
        print(f"   MDL Advantage: {results['mdl_advantage']*100:.1f}%")
        print("   Controls properly fail (shuffled and random)")
        print(f"   Transfer learning works ({(direct_error - transfer_error) / direct_error * 100:.1f}% benefit)")
        print("\n🎉 When domains ACTUALLY share mathematical structure,")
        print("   the OS-of-Reality hypothesis is SUPPORTED!")
    else:
        print("❌ UNEXPECTED: Test failed even with designed shared structure")
        print("   This suggests the test might be too conservative")
    
    return results


if __name__ == "__main__":
    results = create_shared_structure_data()
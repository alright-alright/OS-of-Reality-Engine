#!/usr/bin/env python3
"""
Test if domains actually share structure by measuring transfer learning
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.os_reality_protocol import OSRealityProtocol


def test_transfer_learning():
    """
    The RIGHT way to test unified substrate:
    Can patterns learned from one domain help predict another?
    """
    
    print("="*80)
    print("TRANSFER LEARNING TEST")
    print("="*80)
    print("\nHypothesis: If domains share mathematical substrate,")
    print("            patterns from one should help predict another")
    
    # Get real data
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Flatten sequences for analysis
    bio_data = []
    geo_data = []
    cosmo_data = []
    
    for seq in protocol.sequences['biological']:
        # Take windows of the sequence
        for i in range(0, len(seq) - 10, 10):
            bio_data.append(seq[i:i+10].flatten())
    
    for seq in protocol.sequences['geological']:
        for i in range(0, len(seq) - 10, 10):
            geo_data.append(seq[i:i+10].flatten())
            
    for seq in protocol.sequences['cosmological']:
        for i in range(0, len(seq) - 10, 10):
            cosmo_data.append(seq[i:i+10].flatten())
    
    bio_data = np.array(bio_data)
    geo_data = np.array(geo_data)
    cosmo_data = np.array(cosmo_data)
    
    print(f"\nData shapes:")
    print(f"  Biological: {bio_data.shape}")
    print(f"  Geological: {geo_data.shape}")
    print(f"  Cosmological: {cosmo_data.shape}")
    
    # Test 1: Train on bio+geo, test on cosmo
    print("\n1. Train on Biological + Geological, Test on Cosmological:")
    print("-" * 60)
    
    train_data = np.vstack([bio_data, geo_data])
    test_data = cosmo_data
    
    # Fit PCA on training domains
    pca_transfer = PCA(n_components=10)
    pca_transfer.fit(train_data)
    
    # Test on held-out domain
    test_transformed = pca_transfer.transform(test_data)
    test_reconstructed = pca_transfer.inverse_transform(test_transformed)
    transfer_error = mean_squared_error(test_data, test_reconstructed)
    
    # Compare to PCA trained only on test domain
    pca_direct = PCA(n_components=10)
    pca_direct.fit(test_data)
    direct_transformed = pca_direct.transform(test_data)
    direct_reconstructed = pca_direct.inverse_transform(direct_transformed)
    direct_error = mean_squared_error(test_data, direct_reconstructed)
    
    # Compare to random projection (baseline)
    random_projection = np.random.randn(test_data.shape[1], 10)
    random_transformed = test_data @ random_projection
    random_reconstructed = random_transformed @ random_projection.T
    random_error = mean_squared_error(test_data, random_reconstructed)
    
    print(f"Transfer learning error: {transfer_error:.4f}")
    print(f"Direct training error: {direct_error:.4f}")
    print(f"Random baseline error: {random_error:.4f}")
    
    transfer_advantage = (random_error - transfer_error) / random_error
    direct_advantage = (random_error - direct_error) / random_error
    
    print(f"\nTransfer improvement over random: {transfer_advantage:.2%}")
    print(f"Direct improvement over random: {direct_advantage:.2%}")
    
    # Test 2: Check if transfer beats domain-specific for small data
    print("\n2. Small Data Regime (where transfer should help most):")
    print("-" * 60)
    
    # Use only 10 samples from test domain
    small_test = test_data[:10]
    
    # PCA needs at least as many samples as components
    n_comp_small = min(5, small_test.shape[0] - 1)
    
    # Train small PCA on limited data
    pca_small = PCA(n_components=n_comp_small)
    pca_small.fit(small_test)
    small_recon = pca_small.inverse_transform(pca_small.transform(small_test))
    small_error = mean_squared_error(small_test, small_recon)
    
    # Use transfer model (already trained on other domains)
    transfer_small_recon = pca_transfer.inverse_transform(pca_transfer.transform(small_test))
    transfer_small_error = mean_squared_error(small_test, transfer_small_recon)
    
    print(f"Small domain-specific error: {small_error:.4f}")
    print(f"Transfer learning error: {transfer_small_error:.4f}")
    
    if transfer_small_error < small_error:
        print("✓ Transfer learning helps with small data!")
    else:
        print("✗ Transfer learning doesn't help")
    
    # Test 3: Measure actual compression using MDL properly
    print("\n3. Proper MDL Calculation:")
    print("-" * 60)
    
    def calculate_mdl(pca_model, data):
        """Calculate proper MDL: model_bits + residual_bits"""
        n_params = pca_model.n_components_ * (pca_model.n_features_in_ + 1)
        model_bits = n_params * 32  # 32 bits per float
        
        # Reconstruction error
        recon = pca_model.inverse_transform(pca_model.transform(data))
        mse = mean_squared_error(data, recon)
        
        # Residual coding cost (simplified: assume Gaussian)
        if mse > 0:
            residual_bits_per_value = 0.5 * np.log2(2 * np.pi * np.e * mse)
        else:
            residual_bits_per_value = 0
            
        total_values = data.shape[0] * data.shape[1]
        residual_bits = residual_bits_per_value * total_values
        
        return model_bits + residual_bits
    
    # Unified model MDL
    all_data = np.vstack([bio_data, geo_data, cosmo_data])
    pca_unified = PCA(n_components=10)
    pca_unified.fit(all_data)
    unified_mdl = calculate_mdl(pca_unified, all_data)
    
    # Separate models MDL
    pca_bio = PCA(n_components=10)
    pca_geo = PCA(n_components=10)
    pca_cosmo = PCA(n_components=10)
    
    pca_bio.fit(bio_data)
    pca_geo.fit(geo_data)
    pca_cosmo.fit(cosmo_data)
    
    bio_mdl = calculate_mdl(pca_bio, bio_data)
    geo_mdl = calculate_mdl(pca_geo, geo_data)
    cosmo_mdl = calculate_mdl(pca_cosmo, cosmo_data)
    
    separate_mdl = bio_mdl + geo_mdl + cosmo_mdl
    
    print(f"Unified MDL: {unified_mdl:.0f} bits")
    print(f"Separate MDL: {separate_mdl:.0f} bits")
    print(f"Compression advantage: {(separate_mdl - unified_mdl) / separate_mdl:.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if transfer_advantage > 0.1:
        print("✓ Evidence of shared structure: transfer learning works!")
    else:
        print("✗ No evidence of shared structure: transfer learning fails")
    
    print("\nThe issue with the original T0 test:")
    print("- It only counted parameters, not actual compression")
    print("- It reduced sequences to means, losing temporal info")
    print("- Unified models need MORE capacity when domains differ")
    print("\nA proper test should measure:")
    print("- Transfer learning between domains")
    print("- Shared latent representations")
    print("- Cross-domain prediction accuracy")


if __name__ == "__main__":
    test_transfer_learning()
#!/usr/bin/env python3
"""
Properly test compression: does unified model find shared structure?
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def test_real_compression():
    """Test if we can actually find shared structure across domains"""
    
    print("="*80)
    print("PROPER COMPRESSION TEST")
    print("="*80)
    
    # Create data with ACTUAL shared structure
    print("\n1. Creating data with shared latent structure:")
    print("-" * 40)
    
    # Shared latent factors (these are the universal primitives)
    n_samples = 100
    latent_dim = 2
    
    # Generate shared latent factors
    latent = np.random.randn(n_samples, latent_dim)
    
    # Domain-specific projections (how each domain expresses the latent factors)
    bio_projection = np.random.randn(latent_dim, 10)
    geo_projection = np.random.randn(latent_dim, 10) 
    cosmo_projection = np.random.randn(latent_dim, 10)
    
    # Generate domain data from shared latent + domain-specific noise
    bio_data = latent @ bio_projection + np.random.randn(n_samples, 10) * 0.1
    geo_data = latent @ geo_projection + np.random.randn(n_samples, 10) * 0.1
    cosmo_data = latent @ cosmo_projection + np.random.randn(n_samples, 10) * 0.1
    
    print(f"Created {n_samples} samples per domain")
    print(f"Shared latent dimensions: {latent_dim}")
    print(f"Observable dimensions per domain: 10")
    
    # Test 1: Unified model on structured data
    print("\n2. Testing unified model on structured data:")
    print("-" * 40)
    
    # Stack all data
    all_data = np.vstack([bio_data, geo_data, cosmo_data])
    
    # Fit PCA with different numbers of components
    for n_comp in [2, 5, 10]:
        pca = PCA(n_components=n_comp)
        pca.fit(all_data)
        reconstructed = pca.inverse_transform(pca.transform(all_data))
        error = mean_squared_error(all_data, reconstructed)
        var_explained = np.sum(pca.explained_variance_ratio_)
        
        print(f"Components: {n_comp}, Error: {error:.4f}, Variance explained: {var_explained:.2%}")
    
    # Test 2: Compare unified vs separate models
    print("\n3. Unified vs Separate models:")
    print("-" * 40)
    
    # Unified model
    pca_unified = PCA(n_components=5)
    pca_unified.fit(all_data)
    unified_recon = pca_unified.inverse_transform(pca_unified.transform(all_data))
    unified_error = mean_squared_error(all_data, unified_recon)
    
    # Separate models
    pca_bio = PCA(n_components=5)
    pca_geo = PCA(n_components=5)
    pca_cosmo = PCA(n_components=5)
    
    pca_bio.fit(bio_data)
    pca_geo.fit(geo_data)
    pca_cosmo.fit(cosmo_data)
    
    bio_recon = pca_bio.inverse_transform(pca_bio.transform(bio_data))
    geo_recon = pca_geo.inverse_transform(pca_geo.transform(geo_data))
    cosmo_recon = pca_cosmo.inverse_transform(pca_cosmo.transform(cosmo_data))
    
    separate_recon = np.vstack([bio_recon, geo_recon, cosmo_recon])
    separate_error = mean_squared_error(all_data, separate_recon)
    
    print(f"Unified model error: {unified_error:.4f}")
    print(f"Separate models error: {separate_error:.4f}")
    print(f"Advantage (lower is better): {(separate_error - unified_error) / separate_error:.2%}")
    
    # Test 3: Random data (no shared structure)
    print("\n4. Testing on random data (no shared structure):")
    print("-" * 40)
    
    # Completely random data
    random_bio = np.random.randn(n_samples, 10)
    random_geo = np.random.randn(n_samples, 10)
    random_cosmo = np.random.randn(n_samples, 10)
    random_all = np.vstack([random_bio, random_geo, random_cosmo])
    
    # Unified model on random
    pca_unified_random = PCA(n_components=5)
    pca_unified_random.fit(random_all)
    unified_random_recon = pca_unified_random.inverse_transform(pca_unified_random.transform(random_all))
    unified_random_error = mean_squared_error(random_all, unified_random_recon)
    
    # Separate models on random
    pca_bio_r = PCA(n_components=5)
    pca_geo_r = PCA(n_components=5)
    pca_cosmo_r = PCA(n_components=5)
    
    pca_bio_r.fit(random_bio)
    pca_geo_r.fit(random_geo)
    pca_cosmo_r.fit(random_cosmo)
    
    bio_r_recon = pca_bio_r.inverse_transform(pca_bio_r.transform(random_bio))
    geo_r_recon = pca_geo_r.inverse_transform(pca_geo_r.transform(random_geo))
    cosmo_r_recon = pca_cosmo_r.inverse_transform(pca_cosmo_r.transform(random_cosmo))
    
    separate_random_recon = np.vstack([bio_r_recon, geo_r_recon, cosmo_r_recon])
    separate_random_error = mean_squared_error(random_all, separate_random_recon)
    
    print(f"Unified model error: {unified_random_error:.4f}")
    print(f"Separate models error: {separate_random_error:.4f}")
    print(f"Advantage: {(separate_random_error - unified_random_error) / separate_random_error:.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    structured_advantage = (separate_error - unified_error) / separate_error
    random_advantage = (separate_random_error - unified_random_error) / separate_random_error
    
    print(f"\nStructured data (shared latent):")
    print(f"  Advantage: {structured_advantage:.2%}")
    print(f"  Interpretation: Unified model {'BETTER' if structured_advantage > 0 else 'WORSE'}")
    
    print(f"\nRandom data (no shared structure):")  
    print(f"  Advantage: {random_advantage:.2%}")
    print(f"  Interpretation: Unified model {'BETTER' if random_advantage > 0 else 'WORSE'}")
    
    print(f"\nDifference: {abs(structured_advantage - random_advantage):.2%}")
    
    if structured_advantage > random_advantage + 0.05:
        print("\n✓ SUCCESS: Unified model finds shared structure when it exists!")
    else:
        print("\n❌ PROBLEM: Unified model doesn't leverage shared structure")
        print("   This might be because:")
        print("   - PCA is too simple to find complex shared patterns")
        print("   - Need more sophisticated models (VAE, neural networks)")
        print("   - The shared structure needs to be stronger")


if __name__ == "__main__":
    test_real_compression()
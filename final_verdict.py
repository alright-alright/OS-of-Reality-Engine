#!/usr/bin/env python3
"""
Final verdict on OS-of-Reality hypothesis with proper testing
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.os_reality_protocol import OSRealityProtocol


def final_verdict():
    """
    Definitive test using proper cross-validation and realistic constraints
    """
    
    print("="*80)
    print("FINAL VERDICT: OS-OF-REALITY HYPOTHESIS")
    print("="*80)
    
    # Get data
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Create sliding windows from sequences
    def create_windows(sequences, window_size=30, stride=10):
        """Create overlapping windows from sequences"""
        windows = []
        for seq in sequences:
            for i in range(0, len(seq) - window_size, stride):
                windows.append(seq[i:i+window_size].flatten())
        return np.array(windows)
    
    bio_windows = create_windows(protocol.sequences['biological'])
    geo_windows = create_windows(protocol.sequences['geological'])
    cosmo_windows = create_windows(protocol.sequences['cosmological'])
    
    print(f"\nData prepared:")
    print(f"  Biological: {bio_windows.shape}")
    print(f"  Geological: {geo_windows.shape}")
    print(f"  Cosmological: {cosmo_windows.shape}")
    
    # Split data for proper testing
    def split_data(data, train_ratio=0.7):
        n_train = int(len(data) * train_ratio)
        indices = np.random.permutation(len(data))
        return data[indices[:n_train]], data[indices[n_train:]]
    
    bio_train, bio_test = split_data(bio_windows)
    geo_train, geo_test = split_data(geo_windows)
    cosmo_train, cosmo_test = split_data(cosmo_windows)
    
    print(f"\nTrain/Test split: 70/30")
    
    # TEST 1: Cross-domain prediction
    print("\n" + "="*60)
    print("TEST 1: CROSS-DOMAIN PREDICTION")
    print("="*60)
    print("Can patterns from two domains predict the third?")
    
    results = []
    
    # Leave-one-domain-out
    domains = {
        'biological': (bio_train, bio_test),
        'geological': (geo_train, geo_test),
        'cosmological': (cosmo_train, cosmo_test)
    }
    
    for test_domain, (_, test_data) in domains.items():
        print(f"\nHolding out: {test_domain}")
        
        # Train on other two domains
        train_domains = [d for d in domains if d != test_domain]
        train_data = np.vstack([domains[d][0] for d in train_domains])
        
        # Fit model on training domains
        n_components = min(20, train_data.shape[0] - 1, train_data.shape[1] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(train_data)
        
        # Test on held-out domain
        test_recon = pca.inverse_transform(pca.transform(test_data))
        test_error = mean_squared_error(test_data, test_recon)
        
        # Baseline: random projection
        random_proj = np.random.randn(test_data.shape[1], n_components)
        random_proj = random_proj / np.linalg.norm(random_proj, axis=0)
        random_recon = (test_data @ random_proj) @ random_proj.T
        random_error = mean_squared_error(test_data, random_recon)
        
        improvement = (random_error - test_error) / random_error * 100
        results.append(improvement)
        
        print(f"  Cross-domain error: {test_error:.4f}")
        print(f"  Random baseline: {random_error:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    avg_improvement = np.mean(results)
    print(f"\nAverage cross-domain improvement: {avg_improvement:.1f}%")
    
    # TEST 2: Shared vs Separate Components
    print("\n" + "="*60)
    print("TEST 2: SHARED VS SEPARATE COMPONENTS")
    print("="*60)
    print("Do domains share common principal components?")
    
    # Fit PCA to each domain
    pca_bio = PCA(n_components=10)
    pca_geo = PCA(n_components=10)
    pca_cosmo = PCA(n_components=10)
    
    pca_bio.fit(bio_train)
    pca_geo.fit(geo_train)
    pca_cosmo.fit(cosmo_train)
    
    # Compare principal components
    def subspace_angle(U, V):
        """Calculate principal angles between subspaces"""
        # Ensure orthonormal bases
        U, _ = np.linalg.qr(U)
        V, _ = np.linalg.qr(V)
        # Compute cosines of principal angles
        cosines = np.linalg.svd(U.T @ V, compute_uv=False)
        # Convert to angles in degrees
        angles = np.arccos(np.clip(cosines, -1, 1)) * 180 / np.pi
        return angles
    
    # Get top 5 components from each
    bio_components = pca_bio.components_[:5].T
    geo_components = pca_geo.components_[:5].T
    cosmo_components = pca_cosmo.components_[:5].T
    
    angles_bio_geo = subspace_angle(bio_components, geo_components)
    angles_bio_cosmo = subspace_angle(bio_components, cosmo_components)
    angles_geo_cosmo = subspace_angle(geo_components, cosmo_components)
    
    print(f"\nPrincipal angles between subspaces (degrees):")
    print(f"  Bio-Geo: {angles_bio_geo.mean():.1f}° ± {angles_bio_geo.std():.1f}°")
    print(f"  Bio-Cosmo: {angles_bio_cosmo.mean():.1f}° ± {angles_bio_cosmo.std():.1f}°")
    print(f"  Geo-Cosmo: {angles_geo_cosmo.mean():.1f}° ± {angles_geo_cosmo.std():.1f}°")
    
    # Random baseline
    random_components = np.random.randn(bio_components.shape[0], 5)
    random_components, _ = np.linalg.qr(random_components)
    angles_random = subspace_angle(bio_components, random_components)
    print(f"  Random baseline: {angles_random.mean():.1f}° ± {angles_random.std():.1f}°")
    
    # TEST 3: Information-Theoretic Test
    print("\n" + "="*60)
    print("TEST 3: INFORMATION-THEORETIC COMPRESSION")
    print("="*60)
    print("Does unified model achieve better compression?")
    
    # Combine all training data
    all_train = np.vstack([bio_train, geo_train, cosmo_train])
    all_test = np.vstack([bio_test, geo_test, cosmo_test])
    
    # Test different model complexities
    compression_results = []
    
    for n_comp in [5, 10, 15, 20]:
        if n_comp >= min(all_train.shape):
            continue
            
        # Unified model
        pca_unified = PCA(n_components=n_comp)
        pca_unified.fit(all_train)
        unified_recon = pca_unified.inverse_transform(pca_unified.transform(all_test))
        unified_mse = mean_squared_error(all_test, unified_recon)
        
        # Separate models (n_comp/3 each to match total parameters)
        n_comp_sep = max(1, n_comp // 3)
        
        pca_bio_sep = PCA(n_components=n_comp_sep)
        pca_geo_sep = PCA(n_components=n_comp_sep)
        pca_cosmo_sep = PCA(n_components=n_comp_sep)
        
        pca_bio_sep.fit(bio_train)
        pca_geo_sep.fit(geo_train)
        pca_cosmo_sep.fit(cosmo_train)
        
        bio_recon_sep = pca_bio_sep.inverse_transform(pca_bio_sep.transform(bio_test))
        geo_recon_sep = pca_geo_sep.inverse_transform(pca_geo_sep.transform(geo_test))
        cosmo_recon_sep = pca_cosmo_sep.inverse_transform(pca_cosmo_sep.transform(cosmo_test))
        
        sep_recon = np.vstack([bio_recon_sep, geo_recon_sep, cosmo_recon_sep])
        separate_mse = mean_squared_error(all_test, sep_recon)
        
        # Calculate compression ratio
        compression_ratio = separate_mse / unified_mse
        compression_results.append(compression_ratio)
        
        print(f"\nComponents: {n_comp} unified, {n_comp_sep} per domain")
        print(f"  Unified MSE: {unified_mse:.4f}")
        print(f"  Separate MSE: {separate_mse:.4f}")
        print(f"  Ratio (>1 means unified better): {compression_ratio:.2f}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    # Analyze results
    cross_domain_works = avg_improvement > 50
    shared_subspace = np.mean([angles_bio_geo.mean(), angles_bio_cosmo.mean(), angles_geo_cosmo.mean()]) < 60
    compression_works = np.mean(compression_results) > 1.0 if compression_results else False
    
    print(f"\n✓ Cross-domain prediction: {'PASS' if cross_domain_works else 'FAIL'} ({avg_improvement:.1f}% improvement)")
    print(f"✓ Shared subspace: {'PASS' if shared_subspace else 'FAIL'} (avg angle: {np.mean([angles_bio_geo.mean(), angles_bio_cosmo.mean(), angles_geo_cosmo.mean()]):.1f}°)")
    print(f"✓ Compression advantage: {'PASS' if compression_works else 'FAIL'} (ratio: {np.mean(compression_results) if compression_results else 0:.2f})")
    
    if cross_domain_works and shared_subspace:
        print("\n🎉 HYPOTHESIS SUPPORTED: Evidence for shared mathematical substrate!")
        print("   Domains appear to share common mathematical structure.")
    elif cross_domain_works or shared_subspace:
        print("\n⚠️  HYPOTHESIS PARTIALLY SUPPORTED: Some evidence of shared structure")
        print("   Further investigation with more sophisticated models recommended.")
    else:
        print("\n❌ HYPOTHESIS NOT SUPPORTED: No clear evidence of shared substrate")
        print("   Domains appear mathematically independent.")
    
    print("\nRECOMMENDATIONS:")
    print("1. The original T0 test was fundamentally flawed (parameter counting)")
    print("2. Use transfer learning and cross-validation for robust testing")
    print("3. Consider more sophisticated models (VAEs, neural networks)")
    print("4. Test on larger, more diverse datasets")
    print("5. Focus on predictive accuracy, not just compression")


if __name__ == "__main__":
    final_verdict()
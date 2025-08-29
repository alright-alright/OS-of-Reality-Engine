#!/usr/bin/env python3
"""
Integration of REAL Postojna Cave breathing data into OS-of-Reality testing
This uses actual cave airflow measurements from Gabrovsek PlosOne dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.tier_implementations.t0_compression_fixed import T0CompressionTestFixed
from protocols.os_reality_protocol import OSRealityProtocol


def load_cave_data():
    """
    Load real cave breathing data from Gabrovsek PlosOne dataset.
    """
    
    data_path = Path("Data-Gabrovsek-PlosOne")
    
    print("="*80)
    print("LOADING REAL POSTOJNA CAVE BREATHING DATA")
    print("="*80)
    print("Source: Gabrovsek, F. (2023) 'How do caves breathe: the airflow patterns in karst underground'")
    print()
    
    cave_sequences = {}
    
    # Load Pisani entrance data
    pisani_file = data_path / "Tout-Wind-Pisani.dat"
    print(f"Loading {pisani_file.name}...")
    
    try:
        # Load with specific encoding for special characters
        pisani_data = pd.read_csv(pisani_file, encoding='latin-1')
        print(f"  Loaded Pisani data: {pisani_data.shape}")
        print(f"  Columns: {list(pisani_data.columns)}")
        
        # Extract airflow velocity and temperature
        wind_col = [col for col in pisani_data.columns if 'Wind' in col][0]
        temp_col = [col for col in pisani_data.columns if 'Tout' in col or 'C' in col][0]
        
        wind_velocity = pisani_data[wind_col].values
        temperature = pisani_data[temp_col].values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(wind_velocity) | np.isnan(temperature))
        wind_velocity = wind_velocity[valid_idx]
        temperature = temperature[valid_idx]
        
        print(f"  Wind velocity range: [{wind_velocity.min():.2f}, {wind_velocity.max():.2f}] m/s")
        print(f"  Temperature range: [{temperature.min():.2f}, {temperature.max():.2f}] °C")
        
        # Convert to aperture dynamics format (3 channels like biological data)
        # Channel 1: Aperture area (derived from wind velocity)
        aperture_area = np.abs(wind_velocity) / wind_velocity.max() if wind_velocity.max() > 0 else wind_velocity
        
        # Channel 2: Pressure differential (use temperature gradient as proxy)
        pressure = (temperature - temperature.mean()) / (temperature.std() + 1e-10)
        
        # Channel 3: Flow rate (wind velocity normalized)
        flow_rate = wind_velocity / (np.abs(wind_velocity).max() + 1e-10)
        
        # Create sequence matrix
        pisani_sequence = np.column_stack([aperture_area, pressure, flow_rate])
        
        # Resample to standard length (900 points like our synthetic data)
        if len(pisani_sequence) > 900:
            indices = np.linspace(0, len(pisani_sequence)-1, 900, dtype=int)
            pisani_sequence = pisani_sequence[indices]
        
        cave_sequences['pisani'] = pisani_sequence
        print(f"  ✓ Created Pisani cave sequence: {pisani_sequence.shape}")
        
    except Exception as e:
        print(f"  Error loading Pisani data: {e}")
    
    # Load Brezimeni entrance data
    brezimeni_file = data_path / "Tout-Wind-Brezimeni.dat"
    print(f"\nLoading {brezimeni_file.name}...")
    
    try:
        brezimeni_data = pd.read_csv(brezimeni_file, encoding='latin-1')
        print(f"  Loaded Brezimeni data: {brezimeni_data.shape}")
        print(f"  Columns: {list(brezimeni_data.columns)}")
        
        # Extract airflow velocity and temperature
        wind_col = [col for col in brezimeni_data.columns if 'Wind' in col][0]
        temp_col = [col for col in brezimeni_data.columns if 'Tout' in col or 'C' in col][0]
        
        wind_velocity = brezimeni_data[wind_col].values
        temperature = brezimeni_data[temp_col].values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(wind_velocity) | np.isnan(temperature))
        wind_velocity = wind_velocity[valid_idx]
        temperature = temperature[valid_idx]
        
        print(f"  Wind velocity range: [{wind_velocity.min():.2f}, {wind_velocity.max():.2f}] m/s")
        print(f"  Temperature range: [{temperature.min():.2f}, {temperature.max():.2f}] °C")
        
        # Convert to aperture dynamics format
        aperture_area = np.abs(wind_velocity) / wind_velocity.max() if wind_velocity.max() > 0 else wind_velocity
        pressure = (temperature - temperature.mean()) / (temperature.std() + 1e-10)
        flow_rate = wind_velocity / (np.abs(wind_velocity).max() + 1e-10)
        
        # Create sequence matrix
        brezimeni_sequence = np.column_stack([aperture_area, pressure, flow_rate])
        
        # Resample to standard length
        if len(brezimeni_sequence) > 900:
            indices = np.linspace(0, len(brezimeni_sequence)-1, 900, dtype=int)
            brezimeni_sequence = brezimeni_sequence[indices]
        
        cave_sequences['brezimeni'] = brezimeni_sequence
        print(f"  ✓ Created Brezimeni cave sequence: {brezimeni_sequence.shape}")
        
    except Exception as e:
        print(f"  Error loading Brezimeni data: {e}")
    
    # Create a third sequence by combining both entrances (simulating whole cave system)
    if 'pisani' in cave_sequences and 'brezimeni' in cave_sequences:
        # Average the two entrances to simulate overall cave breathing
        combined_sequence = (cave_sequences['pisani'] + cave_sequences['brezimeni']) / 2
        cave_sequences['combined'] = combined_sequence
        print(f"\n  ✓ Created combined cave sequence: {combined_sequence.shape}")
    
    print(f"\n✓ Successfully loaded {len(cave_sequences)} real cave breathing sequences")
    return cave_sequences


def test_real_cave_data():
    """
    Test OS-of-Reality hypothesis with REAL cave breathing data
    """
    
    print("\n" + "="*80)
    print("TESTING OS-OF-REALITY WITH REAL CAVE DATA")
    print("="*80)
    
    # Load real cave data
    real_cave_sequences = load_cave_data()
    
    if not real_cave_sequences:
        print("❌ No cave data could be loaded")
        return
    
    # Initialize OS-of-Reality protocol
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Replace synthetic geological data with REAL cave data
    print("\n1. Replacing synthetic geological data with REAL cave breathing...")
    protocol.sequences['geological'] = list(real_cave_sequences.values())
    
    print(f"   Biological: {len(protocol.sequences['biological'])} synthetic throat sequences")
    print(f"   Geological: {len(protocol.sequences['geological'])} REAL cave sequences")
    print(f"   Cosmological: {len(protocol.sequences['cosmological'])} synthetic void sequences")
    
    # Run the fixed T0 compression test
    print("\n2. Running T0 Compression Test with REAL data...")
    print("-" * 60)
    
    t0_test = T0CompressionTestFixed(threshold=0.10)
    results = t0_test.run_test(protocol.domain_adapters, protocol.sequences)
    
    print(f"MDL Advantage: {results['mdl_advantage']:.3f}")
    print(f"Passed: {results['passed']}")
    print(f"Unified MDL: {results['unified_mdl']:.0f}")
    print(f"Separate MDL: {results['separate_mdl']:.0f}")
    
    if 'details' in results:
        print(f"\nDetails:")
        print(f"  Unified params: {results['details'].get('unified_params', 'N/A')}")
        print(f"  Unified reconstruction error: {results['details'].get('unified_reconstruction_error', 'N/A'):.6f}")
        variance = results['details'].get('unified_model', {}).get('total_variance_explained', None)
        if variance is not None:
            print(f"  Variance explained: {variance:.2%}")
        else:
            print(f"  Variance explained: N/A")
    
    # Test cross-domain prediction
    print("\n3. Testing Cross-Domain Prediction...")
    print("-" * 60)
    
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    
    # Prepare data
    bio_data = np.vstack([seq[:300, :].flatten() for seq in protocol.sequences['biological']])
    geo_data = np.vstack([seq[:300, :].flatten() for seq in protocol.sequences['geological']])
    cosmo_data = np.vstack([seq[:300, :].flatten() for seq in protocol.sequences['cosmological']])
    
    # Train on biological (synthetic) + geological (REAL), test on cosmological
    train_data = np.vstack([bio_data, geo_data])
    test_data = cosmo_data
    
    n_components = min(10, train_data.shape[0] - 1, train_data.shape[1] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    
    test_recon = pca.inverse_transform(pca.transform(test_data))
    transfer_error = mean_squared_error(test_data, test_recon)
    
    # Random baseline
    random_proj = np.random.randn(test_data.shape[1], n_components)
    random_proj = random_proj / np.linalg.norm(random_proj, axis=0)
    random_recon = (test_data @ random_proj) @ random_proj.T
    random_error = mean_squared_error(test_data, random_recon)
    
    improvement = (random_error - transfer_error) / random_error * 100
    
    print(f"Transfer learning error: {transfer_error:.4f}")
    print(f"Random baseline error: {random_error:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    
    # Visualize the data
    print("\n4. Visualizing Real Cave vs Synthetic Throat Dynamics...")
    print("-" * 60)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Biological (synthetic throat)
    bio_seq = protocol.sequences['biological'][0]
    axes[0, 0].plot(bio_seq[:300, 0], 'b-', alpha=0.7)
    axes[0, 0].set_title('Biological: Throat Aperture (Synthetic)')
    axes[0, 0].set_ylabel('Aperture')
    
    axes[0, 1].plot(bio_seq[:300, 1], 'b-', alpha=0.7)
    axes[0, 1].set_title('Biological: Pressure (Synthetic)')
    axes[0, 1].set_ylabel('Pressure')
    
    axes[0, 2].plot(bio_seq[:300, 2], 'b-', alpha=0.7)
    axes[0, 2].set_title('Biological: Flow Rate (Synthetic)')
    axes[0, 2].set_ylabel('Flow')
    
    # Row 2: Geological (REAL cave)
    geo_seq = protocol.sequences['geological'][0]
    axes[1, 0].plot(geo_seq[:300, 0], 'g-', alpha=0.7)
    axes[1, 0].set_title('Geological: Cave Aperture (REAL Postojna)')
    axes[1, 0].set_ylabel('Aperture')
    
    axes[1, 1].plot(geo_seq[:300, 1], 'g-', alpha=0.7)
    axes[1, 1].set_title('Geological: Temperature/Pressure (REAL)')
    axes[1, 1].set_ylabel('Pressure')
    
    axes[1, 2].plot(geo_seq[:300, 2], 'g-', alpha=0.7)
    axes[1, 2].set_title('Geological: Airflow (REAL)')
    axes[1, 2].set_ylabel('Flow')
    
    # Row 3: Cosmological (synthetic void)
    cosmo_seq = protocol.sequences['cosmological'][0]
    axes[2, 0].plot(cosmo_seq[:300, 0], 'r-', alpha=0.7)
    axes[2, 0].set_title('Cosmological: Void Aperture (Synthetic)')
    axes[2, 0].set_ylabel('Aperture')
    axes[2, 0].set_xlabel('Time')
    
    axes[2, 1].plot(cosmo_seq[:300, 1], 'r-', alpha=0.7)
    axes[2, 1].set_title('Cosmological: Dark Energy (Synthetic)')
    axes[2, 1].set_ylabel('Pressure')
    axes[2, 1].set_xlabel('Time')
    
    axes[2, 2].plot(cosmo_seq[:300, 2], 'r-', alpha=0.7)
    axes[2, 2].set_title('Cosmological: Expansion Rate (Synthetic)')
    axes[2, 2].set_ylabel('Flow')
    axes[2, 2].set_xlabel('Time')
    
    plt.suptitle('OS-of-Reality: REAL Cave Breathing vs Synthetic Patterns', fontsize=16)
    plt.tight_layout()
    plt.savefig('real_cave_vs_synthetic_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization as 'real_cave_vs_synthetic_comparison.png'")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT: REAL CAVE DATA TEST")
    print("="*80)
    
    if results['passed']:
        print("🎉 HYPOTHESIS SUPPORTED WITH REAL DATA!")
        print(f"   Real Postojna Cave breathing shares {results['mdl_advantage']*100:.1f}% compression")
        print("   with synthetic throat swallowing and cosmic void expansion!")
        print("\n   This suggests caves ACTUALLY breathe like throats swallow!")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED WITH CURRENT REAL DATA")
        print(f"   MDL Advantage: {results['mdl_advantage']*100:.1f}%")
        print("\n   Possible reasons:")
        print("   - Real cave data is more complex than synthetic patterns")
        print("   - Need more sophisticated models to find shared structure")
        print("   - Cave breathing may operate on different timescales")
    
    if improvement > 50:
        print(f"\n   However, cross-domain prediction shows {improvement:.1f}% improvement!")
        print("   This suggests some shared mathematical structure exists.")
    
    return results


if __name__ == "__main__":
    # Run the test with real cave data
    results = test_real_cave_data()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("This was a test of the OS-of-Reality hypothesis using:")
    print("  • REAL cave breathing data from Postojna Cave (Gabrovsek 2023)")
    print("  • Synthetic throat swallowing dynamics")
    print("  • Synthetic cosmic void expansion patterns")
    print("\nThe question: Do these three domains share mathematical structure?")
    print("="*80)
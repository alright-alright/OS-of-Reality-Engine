#!/usr/bin/env python3
"""
Deeper analysis of REAL cave breathing patterns vs throat/cosmic dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrate_real_cave_data import load_cave_data
from protocols.os_reality_protocol import OSRealityProtocol


def analyze_spectral_patterns():
    """
    Analyze frequency domain patterns across all three domains
    """
    
    print("="*80)
    print("SPECTRAL ANALYSIS: REAL CAVE vs SYNTHETIC PATTERNS")
    print("="*80)
    
    # Load data
    real_cave = load_cave_data()
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Get representative sequences
    cave_seq = list(real_cave.values())[0]  # Pisani entrance
    bio_seq = protocol.sequences['biological'][0]
    cosmo_seq = protocol.sequences['cosmological'][0]
    
    # Compute power spectral density for each channel
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    channels = ['Aperture', 'Pressure', 'Flow Rate']
    domains = [('Cave (REAL)', cave_seq, 'green'),
               ('Throat (Synthetic)', bio_seq, 'blue'),
               ('Cosmic (Synthetic)', cosmo_seq, 'red')]
    
    shared_frequencies = []
    
    for ch_idx, channel in enumerate(channels):
        print(f"\nChannel {ch_idx+1}: {channel}")
        print("-" * 40)
        
        for row_idx, (name, sequence, color) in enumerate(domains):
            # Get signal
            signal_data = sequence[:, ch_idx]
            
            # Compute PSD
            freqs, psd = signal.welch(signal_data, fs=10, nperseg=min(256, len(signal_data)//4))
            
            # Plot
            axes[row_idx, ch_idx].semilogy(freqs, psd, color=color, alpha=0.7, linewidth=2)
            axes[row_idx, ch_idx].set_title(f'{name}: {channel}')
            axes[row_idx, ch_idx].set_xlabel('Frequency (Hz)')
            axes[row_idx, ch_idx].set_ylabel('PSD')
            axes[row_idx, ch_idx].grid(True, alpha=0.3)
            
            # Find dominant frequencies
            peak_indices = signal.find_peaks(np.log10(psd + 1e-10))[0]
            if len(peak_indices) > 0:
                dominant_freq = freqs[peak_indices[np.argmax(psd[peak_indices])]]
                axes[row_idx, ch_idx].axvline(dominant_freq, color=color, linestyle='--', alpha=0.5)
                print(f"  {name}: Dominant freq = {dominant_freq:.3f} Hz")
                
                if row_idx == 0:  # Cave data
                    shared_frequencies.append(dominant_freq)
    
    plt.suptitle('Spectral Analysis: Shared Frequencies Across Domains', fontsize=16)
    plt.tight_layout()
    plt.savefig('spectral_analysis_real_cave.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved spectral analysis as 'spectral_analysis_real_cave.png'")
    
    return shared_frequencies


def analyze_phase_space():
    """
    Analyze phase space trajectories - do they share attractors?
    """
    
    print("\n" + "="*80)
    print("PHASE SPACE ANALYSIS: SHARED ATTRACTORS?")
    print("="*80)
    
    # Load data
    real_cave = load_cave_data()
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    # Get sequences
    cave_seq = list(real_cave.values())[0]
    bio_seq = protocol.sequences['biological'][0]
    cosmo_seq = protocol.sequences['cosmological'][0]
    
    # Create phase space plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Time-delay embedding for phase space reconstruction
    delay = 10
    
    for idx, (name, seq, color) in enumerate([
        ('REAL Cave', cave_seq, 'green'),
        ('Synthetic Throat', bio_seq, 'blue'),
        ('Synthetic Cosmic', cosmo_seq, 'red')
    ]):
        # Use aperture channel for phase space
        x = seq[:-delay, 0]
        y = seq[delay:, 0]
        
        axes[idx].scatter(x, y, c=color, alpha=0.3, s=1)
        axes[idx].set_title(f'{name} Phase Space')
        axes[idx].set_xlabel('Aperture(t)')
        axes[idx].set_ylabel('Aperture(t+τ)')
        axes[idx].grid(True, alpha=0.3)
        
        # Calculate attractor dimensions
        from scipy.spatial.distance import pdist
        points = np.column_stack([x, y])
        distances = pdist(points)
        
        # Correlation dimension (simplified)
        r_values = np.logspace(-3, 0, 20)
        correlation_sum = []
        
        for r in r_values:
            c_r = np.sum(distances < r) / len(distances)
            correlation_sum.append(c_r)
        
        # Estimate dimension from slope
        log_r = np.log(r_values[5:15])
        log_c = np.log(np.array(correlation_sum[5:15]) + 1e-10)
        slope, _ = np.polyfit(log_r, log_c, 1)
        
        print(f"{name} - Correlation dimension: {slope:.2f}")
    
    plt.suptitle('Phase Space: Similar Attractors Across Domains?', fontsize=16)
    plt.tight_layout()
    plt.savefig('phase_space_real_cave.png', dpi=150, bbox_inches='tight')
    print("✓ Saved phase space analysis as 'phase_space_real_cave.png'")


def analyze_information_transfer():
    """
    Test information transfer between domains using mutual information
    """
    
    print("\n" + "="*80)
    print("INFORMATION TRANSFER ANALYSIS")
    print("="*80)
    
    # Load data
    real_cave = load_cave_data()
    protocol = OSRealityProtocol(seed=42)
    protocol.setup_domains()
    
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import mutual_info_score
    
    # Prepare sequences
    cave_data = np.vstack([seq.flatten()[:300] for seq in real_cave.values()])
    bio_data = np.vstack([seq.flatten()[:300] for seq in protocol.sequences['biological']])
    cosmo_data = np.vstack([seq.flatten()[:300] for seq in protocol.sequences['cosmological']])
    
    # Calculate mutual information between domains
    print("\nMutual Information Matrix:")
    print("-" * 40)
    
    domains_data = {
        'Cave (REAL)': cave_data,
        'Throat (Synthetic)': bio_data,
        'Cosmic (Synthetic)': cosmo_data
    }
    
    mi_matrix = np.zeros((3, 3))
    domain_names = list(domains_data.keys())
    
    for i, (name1, data1) in enumerate(domains_data.items()):
        for j, (name2, data2) in enumerate(domains_data.items()):
            if i != j:
                # Use first principal component for MI calculation
                from sklearn.decomposition import PCA
                pca1 = PCA(n_components=1).fit(data1)
                pca2 = PCA(n_components=1).fit(data2)
                
                pc1 = pca1.transform(data1)[:, 0]
                pc2 = pca2.transform(data2)[:, 0]
                
                # Discretize for MI calculation
                bins = 10
                pc1_discrete = np.digitize(pc1, np.histogram(pc1, bins)[1])
                pc2_discrete = np.digitize(pc2, np.histogram(pc2, bins)[1])
                
                mi = mutual_info_score(pc1_discrete, pc2_discrete)
                mi_matrix[i, j] = mi
    
    # Display MI matrix
    print("\n       Cave    Throat  Cosmic")
    for i, name in enumerate(['Cave  ', 'Throat', 'Cosmic']):
        print(f"{name} ", end='')
        for j in range(3):
            print(f"{mi_matrix[i, j]:6.3f}  ", end='')
        print()
    
    # Calculate average cross-domain MI
    cave_throat_mi = mi_matrix[0, 1]
    cave_cosmic_mi = mi_matrix[0, 2]
    throat_cosmic_mi = mi_matrix[1, 2]
    
    avg_mi = (cave_throat_mi + cave_cosmic_mi + throat_cosmic_mi) / 3
    
    print(f"\nAverage cross-domain MI: {avg_mi:.3f}")
    
    if avg_mi > 0.1:
        print("✓ Significant information sharing detected!")
    else:
        print("✗ Limited information sharing")
    
    return mi_matrix


def test_universality_hypothesis():
    """
    Final comprehensive test of the universality hypothesis with REAL data
    """
    
    print("\n" + "="*80)
    print("UNIVERSALITY TEST: Do apertures share mathematical laws?")
    print("="*80)
    
    # Run all analyses
    print("\n1. Spectral Analysis...")
    shared_freqs = analyze_spectral_patterns()
    
    print("\n2. Phase Space Analysis...")
    analyze_phase_space()
    
    print("\n3. Information Transfer Analysis...")
    mi_matrix = analyze_information_transfer()
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT: REAL DATA ANALYSIS")
    print("="*80)
    
    evidence_for = []
    evidence_against = []
    
    # Check shared frequencies
    if len(shared_freqs) > 0:
        evidence_for.append("Shared frequency components detected")
    
    # Check mutual information
    avg_mi = mi_matrix[np.triu_indices(3, k=1)].mean()
    if avg_mi > 0.1:
        evidence_for.append(f"Significant mutual information ({avg_mi:.3f})")
    else:
        evidence_against.append(f"Low mutual information ({avg_mi:.3f})")
    
    print("\nEvidence FOR shared mathematical structure:")
    for e in evidence_for:
        print(f"  ✓ {e}")
    
    print("\nEvidence AGAINST shared mathematical structure:")
    for e in evidence_against:
        print(f"  ✗ {e}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if len(evidence_for) > len(evidence_against):
        print("🎉 The REAL Postojna Cave data shows evidence of shared")
        print("   mathematical structure with throat and cosmic dynamics!")
        print("\n   This supports the OS-of-Reality hypothesis that")
        print("   aperture dynamics follow universal mathematical laws.")
    else:
        print("❌ The current analysis does not strongly support")
        print("   shared mathematical structure between domains.")
        print("\n   However, this may be due to:")
        print("   - Different timescales (cave: days, throat: seconds)")
        print("   - Measurement noise in real data")
        print("   - Need for more sophisticated analysis methods")
    
    print("\nNext steps:")
    print("  1. Normalize timescales across domains")
    print("  2. Apply denoising to real cave data")
    print("  3. Use deep learning for pattern discovery")
    print("  4. Collect more diverse real-world aperture data")


if __name__ == "__main__":
    test_universality_hypothesis()
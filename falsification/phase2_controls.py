"""
PHASE 2: CONTROL GAUNTLET
Aggressive controls designed to kill false positives
"""

import numpy as np
import time
from typing import Dict, List, Any
import sys
import os
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocols.os_reality_protocol import OSRealityProtocol
from protocols.tier_implementations.t0_compression import T0CompressionTest


class Phase2ControlGauntlet:
    """Aggressive controls designed to kill false positives"""
    
    def __init__(self, launcher=None):
        self.launcher = launcher
        
        self.control_tests = [
            'shuffled_sequences',
            'shuffled_labels', 
            'pure_random_data',
            'white_noise',
            'structured_but_unrelated',
            'domain_reversed',
            'temporal_scrambled',
            'amplitude_randomized'
        ]
        
        self.results = {}
        
    def log(self, message: str):
        """Log through launcher or print"""
        if self.launcher:
            self.launcher.log(f"[Phase2] {message}")
        else:
            print(f"[Phase2] {message}")
    
    def execute_gauntlet(self) -> Dict:
        """Run every control test designed to break the hypothesis"""
        
        self.log("Starting Control Gauntlet")
        self.log(f"Running {len(self.control_tests)} control tests")
        
        start_time = time.time()
        
        for control_name in self.control_tests:
            self.log(f"Running control: {control_name}")
            
            try:
                # Get the test method
                test_method = getattr(self, f"_test_{control_name}")
                result = test_method()
                self.results[control_name] = result
                
                # Log result
                passed = result['passed']
                self.log(f"  {control_name}: {'PASSED' if passed else 'FAILED'} (expected: FAIL)")
                
                # Save checkpoint
                if self.launcher:
                    self.launcher.save_checkpoint(
                        'phase2_controls',
                        self.results,
                        iteration=len(self.results)
                    )
                    
            except Exception as e:
                self.log(f"  ERROR in {control_name}: {e}")
                self.results[control_name] = {
                    'control_type': control_name,
                    'passed': False,
                    'error': str(e)
                }
        
        # Calculate verdict
        control_passes = sum(1 for r in self.results.values() if r.get('passed', False))
        
        duration = time.time() - start_time
        
        final_results = {
            'phase': 'Control Gauntlet',
            'verdict': 'CLEAN' if control_passes == 0 else 'CONTAMINATED',
            'control_passes': control_passes,
            'total_controls': len(self.results),
            'duration_seconds': duration,
            'results': self.results,
            'summary': {
                'all_controls_failed': control_passes == 0,
                'contamination_rate': control_passes / len(self.results) if self.results else 0
            }
        }
        
        # Log summary
        self.log("="*60)
        self.log("PHASE 2 SUMMARY")
        self.log(f"Verdict: {final_results['verdict']}")
        self.log(f"Controls passed: {control_passes}/{len(self.results)} (should be 0)")
        self.log(f"Duration: {duration:.1f}s")
        self.log("="*60)
        
        return final_results
    
    def _test_shuffled_sequences(self) -> Dict:
        """Shuffle time points within sequences"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Shuffle all sequences
        for domain_name, sequences in protocol.sequences.items():
            for seq in sequences:
                for channel in range(seq.shape[1]):
                    np.random.shuffle(seq[:, channel])
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'shuffled_sequences',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']  # Correct if it fails
        }
    
    def _test_shuffled_labels(self) -> Dict:
        """Keep sequences intact but randomly assign to wrong domains"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Shuffle domain labels
        original_sequences = deepcopy(protocol.sequences)
        domain_names = list(protocol.sequences.keys())
        np.random.shuffle(domain_names)
        
        shuffled_sequences = {}
        for i, domain in enumerate(protocol.sequences.keys()):
            shuffled_sequences[domain] = original_sequences[domain_names[i]]
        
        protocol.sequences = shuffled_sequences
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'shuffled_labels',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }
    
    def _test_pure_random_data(self) -> Dict:
        """Completely synthetic Gaussian noise"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Replace with random data
        for domain_name, sequences in protocol.sequences.items():
            for i, seq in enumerate(sequences):
                protocol.sequences[domain_name][i] = np.random.randn(*seq.shape)
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'pure_random_data',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }
    
    def _test_white_noise(self) -> Dict:
        """Uncorrelated noise across all channels"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Replace with white noise
        for domain_name, sequences in protocol.sequences.items():
            for i, seq in enumerate(sequences):
                # White noise: mean=0, std=1, uncorrelated
                protocol.sequences[domain_name][i] = np.random.normal(0, 1, seq.shape)
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'white_noise',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }
    
    def _test_structured_but_unrelated(self) -> Dict:
        """Sine waves with no cross-domain relationship"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Replace with unrelated sine waves
        for domain_name, sequences in protocol.sequences.items():
            for i, seq in enumerate(sequences):
                t = np.linspace(0, 10, len(seq))
                for channel in range(seq.shape[1]):
                    # Different frequency for each channel, no relationship
                    freq = np.random.uniform(0.1, 5.0)
                    phase = np.random.uniform(0, 2*np.pi)
                    seq[:, channel] = np.sin(freq * t + phase)
                protocol.sequences[domain_name][i] = seq
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'structured_but_unrelated',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }
    
    def _test_domain_reversed(self) -> Dict:
        """Use bio data for geo, geo for cosmo, etc."""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Rotate domain data
        original_sequences = deepcopy(protocol.sequences)
        domain_names = list(protocol.sequences.keys())
        
        for i, domain in enumerate(domain_names):
            next_domain = domain_names[(i + 1) % len(domain_names)]
            protocol.sequences[domain] = original_sequences[next_domain]
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'domain_reversed',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }
    
    def _test_temporal_scrambled(self) -> Dict:
        """Keep spatial structure but destroy temporal relationships"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Scramble temporal order but keep each time slice intact
        for domain_name, sequences in protocol.sequences.items():
            for seq in sequences:
                # Randomly permute time indices
                time_indices = np.arange(len(seq))
                np.random.shuffle(time_indices)
                seq[:] = seq[time_indices]
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'temporal_scrambled',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }
    
    def _test_amplitude_randomized(self) -> Dict:
        """Keep patterns but randomize all amplitudes"""
        protocol = OSRealityProtocol(seed=42)
        protocol.setup_domains()
        
        # Randomize amplitudes while keeping zero-crossings
        for domain_name, sequences in protocol.sequences.items():
            for seq in sequences:
                for channel in range(seq.shape[1]):
                    # Keep sign but randomize magnitude
                    signs = np.sign(seq[:, channel])
                    magnitudes = np.abs(np.random.randn(len(seq)))
                    seq[:, channel] = signs * magnitudes
        
        # Run test - should FAIL
        t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
        result = t0.run_test(protocol.domain_adapters, protocol.sequences)
        
        return {
            'control_type': 'amplitude_randomized',
            'passed': result['passed'],
            'mdl_advantage': result['mdl_advantage'],
            'expected_result': 'FAIL',
            'correct': not result['passed']
        }


if __name__ == "__main__":
    # Test phase 2 independently
    print("Testing Phase 2: Control Gauntlet")
    phase2 = Phase2ControlGauntlet()
    results = phase2.execute_gauntlet()
    
    print(f"\nFinal verdict: {results['verdict']}")
    print(f"Controls passed: {results['control_passes']}/{results['total_controls']}")
    print("Individual results:")
    for control, result in results['results'].items():
        print(f"  {control}: {'PASSED' if result.get('passed') else 'FAILED'}")
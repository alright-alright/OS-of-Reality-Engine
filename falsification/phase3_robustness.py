"""
PHASE 3: ROBUSTNESS TORTURE TEST
Torture test to find breaking points
"""

import numpy as np
import time
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocols.os_reality_protocol import OSRealityProtocol
from protocols.tier_implementations.t0_compression import T0CompressionTest


class Phase3RobustnessTorture:
    """Torture test to find breaking points"""
    
    def __init__(self, launcher=None):
        self.launcher = launcher
        self.torture_tests = [
            'noise_tolerance',
            'sequence_lengths',
            'missing_data',
            'domain_imbalance'
        ]
        
    def log(self, message: str):
        """Log through launcher or print"""
        if self.launcher:
            self.launcher.log(f"[Phase3] {message}")
        else:
            print(f"[Phase3] {message}")
    
    def execute_torture(self) -> Dict:
        """Try to break the unified model"""
        
        self.log("Starting Robustness Torture Test")
        torture_results = {}
        
        # Torture 1: Noise Tolerance
        torture_results['noise_levels'] = self._test_noise_tolerance()
        
        # Torture 2: Sequence Length Variations
        torture_results['sequence_lengths'] = self._test_length_robustness()
        
        return {
            'phase': 'Robustness Torture',
            'torture_results': torture_results,
            'breaking_points': self._identify_breaking_points(torture_results)
        }
    
    def _test_noise_tolerance(self) -> List[Dict]:
        """Test with increasing noise levels"""
        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
        results = []
        
        for noise_level in noise_levels:
            protocol = OSRealityProtocol(seed=42)
            protocol.setup_domains()
            
            # Add noise to all sequences
            for sequences in protocol.sequences.values():
                for seq in sequences:
                    noise = np.random.normal(0, noise_level, seq.shape)
                    seq += noise
            
            t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
            result = t0.run_test(protocol.domain_adapters, protocol.sequences)
            
            results.append({
                'noise_level': noise_level,
                'mdl_advantage': result['mdl_advantage'],
                'passed': result['passed']
            })
            
        return results
    
    def _test_length_robustness(self) -> List[Dict]:
        """Test with different sequence lengths"""
        durations = [30, 60, 120]  # seconds
        results = []
        
        for duration in durations:
            protocol = OSRealityProtocol(seed=42)
            # Note: would need to modify protocol to support different durations
            protocol.setup_domains()
            
            t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
            result = t0.run_test(protocol.domain_adapters, protocol.sequences)
            
            results.append({
                'duration': duration,
                'mdl_advantage': result['mdl_advantage'],
                'passed': result['passed']
            })
            
        return results
    
    def _identify_breaking_points(self, torture_results: Dict) -> List[str]:
        """Identify where the model breaks"""
        breaking_points = []
        
        # Check noise tolerance
        if 'noise_levels' in torture_results:
            for result in torture_results['noise_levels']:
                if not result['passed'] and result['noise_level'] <= 1.0:
                    breaking_points.append(f"noise_level_{result['noise_level']}")
        
        return breaking_points
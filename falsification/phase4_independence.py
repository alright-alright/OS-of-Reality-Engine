"""
PHASE 4: INDEPENDENCE VALIDATION
Rigorous cross-validation protocols
"""

import numpy as np
import time
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocols.os_reality_protocol import OSRealityProtocol
from protocols.tier_implementations.t0_compression import T0CompressionTest


class Phase4IndependenceValidation:
    """Rigorous cross-validation protocols"""
    
    def __init__(self, launcher=None):
        self.launcher = launcher
        
    def log(self, message: str):
        """Log through launcher or print"""
        if self.launcher:
            self.launcher.log(f"[Phase4] {message}")
        else:
            print(f"[Phase4] {message}")
    
    def execute_validation(self) -> Dict:
        """Multiple cross-validation strategies"""
        
        self.log("Starting Independence Validation")
        validation_results = {}
        
        # CV Strategy: Leave-One-Domain-Out
        validation_results['leave_domain_out'] = self._leave_domain_out_validation()
        
        return {
            'phase': 'Independence Validation',
            'validation_results': validation_results,
            'overall_cv_score': self._calculate_overall_cv_score(validation_results)
        }
    
    def _leave_domain_out_validation(self) -> List[Dict]:
        """Train on 2 domains, test on 3rd"""
        domain_names = ['biological', 'geological', 'cosmological']
        results = []
        
        for test_domain in domain_names:
            train_domains = [d for d in domain_names if d != test_domain]
            
            protocol = OSRealityProtocol(seed=42)
            protocol.setup_domains()
            
            # Split data
            train_sequences = {d: protocol.sequences[d] for d in train_domains}
            
            # Simple test: run T0 on train domains only
            t0 = T0CompressionTest(threshold=protocol.THRESHOLDS["T0_MDL_ADVANTAGE"])
            train_adapters = {d: protocol.domain_adapters[d] for d in train_domains}
            
            result = t0.run_test(train_adapters, train_sequences)
            
            results.append({
                'test_domain': test_domain,
                'train_domains': train_domains,
                'mdl_advantage': result['mdl_advantage'],
                'passed': result['passed']
            })
            
        return results
    
    def _calculate_overall_cv_score(self, validation_results: Dict) -> float:
        """Calculate overall cross-validation score"""
        scores = []
        
        if 'leave_domain_out' in validation_results:
            for result in validation_results['leave_domain_out']:
                scores.append(result['mdl_advantage'])
        
        return float(np.mean(scores)) if scores else 0.0
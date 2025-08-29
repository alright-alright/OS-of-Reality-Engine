"""
PHASE 5: PUBLICATION-GRADE STATISTICS
Generate bulletproof statistical evidence
"""

import numpy as np
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase5PublicationStats:
    """Generate publication-ready statistical analysis"""
    
    def __init__(self, launcher=None):
        self.launcher = launcher
        
    def log(self, message: str):
        """Log through launcher or print"""
        if self.launcher:
            self.launcher.log(f"[Phase5] {message}")
        else:
            print(f"[Phase5] {message}")
    
    def generate_final_report(self, all_phase_results: Dict) -> Dict:
        """Comprehensive statistical report"""
        
        self.log("Generating Publication-Grade Statistics")
        
        report = {
            'executive_summary': self._generate_executive_summary(all_phase_results),
            'statistical_significance': self._calculate_statistical_significance(all_phase_results),
            'effect_sizes': self._calculate_effect_sizes(all_phase_results),
            'confidence_intervals': self._calculate_confidence_intervals(all_phase_results),
            'publication_verdict': self._generate_publication_verdict(all_phase_results)
        }
        
        return report
    
    def _generate_executive_summary(self, results: Dict) -> Dict:
        """Generate executive summary"""
        phase1 = results.get('phase1', {})
        phase2 = results.get('phase2', {})
        
        return {
            'total_tests_run': phase1.get('stats', {}).get('total_runs', 0),
            'phase1_verdict': phase1.get('verdict', 'UNKNOWN'),
            'phase2_verdict': phase2.get('verdict', 'UNKNOWN'),
            'mean_mdl_advantage': phase1.get('stats', {}).get('mean_advantage', 0),
            'control_contamination': phase2.get('control_passes', 0)
        }
    
    def _calculate_statistical_significance(self, results: Dict) -> Dict:
        """Calculate statistical significance"""
        phase1 = results.get('phase1', {})
        stats = phase1.get('stats', {})
        
        if not stats:
            return {'error': 'No statistics available'}
        
        # Simple significance test: is CI above threshold?
        ci_99 = stats.get('confidence_interval_99', [0, 0])
        significant = ci_99[0] > 0.10  # Above 10% threshold
        
        return {
            'significant': significant,
            'confidence_level': 0.99,
            'ci_lower': ci_99[0],
            'ci_upper': ci_99[1]
        }
    
    def _calculate_effect_sizes(self, results: Dict) -> Dict:
        """Calculate effect sizes"""
        phase1 = results.get('phase1', {})
        stats = phase1.get('stats', {})
        
        if not stats:
            return {'error': 'No statistics available'}
        
        # Cohen's d effect size
        mean_advantage = stats.get('mean_advantage', 0)
        std_advantage = stats.get('std_advantage', 1)
        
        cohens_d = mean_advantage / (std_advantage + 1e-10)
        
        # Interpret effect size
        if cohens_d < 0.2:
            interpretation = 'negligible'
        elif cohens_d < 0.5:
            interpretation = 'small'
        elif cohens_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'cohens_d': cohens_d,
            'interpretation': interpretation
        }
    
    def _calculate_confidence_intervals(self, results: Dict) -> Dict:
        """Calculate various confidence intervals"""
        phase1 = results.get('phase1', {})
        stats = phase1.get('stats', {})
        
        if not stats:
            return {'error': 'No statistics available'}
        
        return {
            'ci_99': stats.get('confidence_interval_99', [0, 0]),
            'ci_95': stats.get('confidence_interval_95', [0, 0]),
            'mean': stats.get('mean_advantage', 0),
            'median': stats.get('median_advantage', 0)
        }
    
    def _generate_publication_verdict(self, results: Dict) -> Dict:
        """Generate final scientific verdict"""
        
        phase1 = results.get('phase1', {})
        phase2 = results.get('phase2', {})
        phase3 = results.get('phase3', {})
        phase4 = results.get('phase4', {})
        
        # Criteria for publication-grade evidence
        criteria_met = {
            'statistical_robustness': (
                phase1.get('verdict') == 'ROBUST' and
                phase1.get('stats', {}).get('mean_advantage', 0) > 0.50
            ),
            'control_cleanliness': (
                phase2.get('verdict') == 'CLEAN' and
                phase2.get('control_passes', 1) == 0
            ),
            'torture_resistance': (
                len(phase3.get('breaking_points', [1,2,3,4])) < 3
            ),
            'cross_validation_success': (
                phase4.get('overall_cv_score', 0) > 0.30
            )
        }
        
        all_criteria_met = all(criteria_met.values())
        criteria_count = sum(criteria_met.values())
        
        if all_criteria_met:
            verdict = "HYPOTHESIS STRONGLY SUPPORTED"
            confidence = "HIGH"
            recommendation = "READY FOR PUBLICATION"
        elif criteria_count >= 3:
            verdict = "HYPOTHESIS SUPPORTED" 
            confidence = "MODERATE"
            recommendation = "ADDITIONAL VALIDATION RECOMMENDED"
        else:
            verdict = "HYPOTHESIS UNSUPPORTED"
            confidence = "LOW" 
            recommendation = "MAJOR REVISION REQUIRED"
            
        return {
            'verdict': verdict,
            'confidence': confidence,
            'recommendation': recommendation,
            'criteria_met': criteria_met,
            'overall_score': criteria_count / len(criteria_met)
        }
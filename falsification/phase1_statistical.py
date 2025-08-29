"""
PHASE 1: STATISTICAL BOMBARDMENT
Run 1000+ independent trials with different seeds to verify consistency
"""

import numpy as np
import time
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocols.os_reality_protocol import OSRealityProtocol


class Phase1StatisticalBombardment:
    """Run 1000+ independent trials with different seeds"""
    
    def __init__(self, launcher=None, quick_mode=False):
        self.launcher = launcher
        self.quick_mode = quick_mode
        
        # Configuration
        self.target_runs = 100 if quick_mode else 1000  # 1000 for full test
        self.confidence_level = 0.99  # 99% confidence required
        self.min_advantage_threshold = 0.10  # Original 10% threshold
        
        # Use all available cores
        self.n_workers = mp.cpu_count() if not quick_mode else 2
        
        self.results = []
        self.checkpoints_saved = 0
        
    def log(self, message: str):
        """Log through launcher or print"""
        if self.launcher:
            self.launcher.log(f"[Phase1] {message}")
        else:
            print(f"[Phase1] {message}")
    
    def run_single_trial(self, seed: int) -> Dict:
        """Run a single trial with given seed"""
        try:
            # Create protocol with specific seed
            protocol = OSRealityProtocol(seed=seed)
            
            # Setup domains
            protocol.setup_domains()
            
            # Run T0 test only (for speed)
            t0_test = protocol.THRESHOLDS["T0_MDL_ADVANTAGE"]
            from protocols.tier_implementations.t0_compression import T0CompressionTest
            
            t0 = T0CompressionTest(threshold=t0_test)
            result = t0.run_test(protocol.domain_adapters, protocol.sequences)
            
            return {
                'seed': seed,
                'mdl_advantage': result['mdl_advantage'],
                'unified_mdl': result['unified_mdl'],
                'separate_mdl': result['separate_mdl'],
                'passed': result['passed'],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            # Return error result
            return {
                'seed': seed,
                'mdl_advantage': 0.0,
                'unified_mdl': 0.0,
                'separate_mdl': 0.0,
                'passed': False,
                'success': False,
                'error': str(e)
            }
    
    def execute_bombardment(self) -> Dict:
        """Run massive statistical validation"""
        self.log(f"Starting statistical bombardment: {self.target_runs} trials")
        self.log(f"Using {self.n_workers} parallel workers")
        
        start_time = time.time()
        seeds = range(42, 42 + self.target_runs)
        
        # Run trials in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_seed = {
                executor.submit(self.run_single_trial, seed): seed 
                for seed in seeds
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                
                try:
                    result = future.result(timeout=60)  # 60 second timeout per trial
                    self.results.append(result)
                    
                    completed += 1
                    
                    # Update launcher
                    if self.launcher:
                        self.launcher.status['total_tests_run'] += 1
                    
                    # Log progress
                    if completed % 10 == 0:
                        self.log(f"Completed {completed}/{self.target_runs} trials")
                        
                        # Save checkpoint every 50 trials
                        if completed % 50 == 0 and self.launcher:
                            self.launcher.save_checkpoint(
                                'phase1_statistical',
                                self.results,
                                iteration=completed
                            )
                            self.checkpoints_saved += 1
                    
                except Exception as e:
                    self.log(f"Trial {seed} failed: {e}")
                    self.results.append({
                        'seed': seed,
                        'mdl_advantage': 0.0,
                        'passed': False,
                        'success': False,
                        'error': str(e)
                    })
        
        # Calculate statistics
        successful_results = [r for r in self.results if r['success']]
        
        if len(successful_results) < 10:
            self.log(f"WARNING: Only {len(successful_results)} successful trials!")
            return {
                'phase': 'Statistical Bombardment',
                'verdict': 'INSUFFICIENT_DATA',
                'successful_trials': len(successful_results),
                'total_trials': len(self.results),
                'stats': None,
                'error': 'Too few successful trials'
            }
        
        advantages = [r['mdl_advantage'] for r in successful_results]
        
        stats = {
            'mean_advantage': float(np.mean(advantages)),
            'std_advantage': float(np.std(advantages)),
            'median_advantage': float(np.median(advantages)),
            'min_advantage': float(np.min(advantages)),
            'max_advantage': float(np.max(advantages)),
            'pass_rate': float(np.mean([r['passed'] for r in successful_results])),
            'confidence_interval_99': [
                float(np.percentile(advantages, 0.5)),
                float(np.percentile(advantages, 99.5))
            ],
            'confidence_interval_95': [
                float(np.percentile(advantages, 2.5)),
                float(np.percentile(advantages, 97.5))
            ],
            'effect_size': float(np.mean(advantages) / (np.std(advantages) + 1e-10)),
            'total_runs': len(self.results),
            'successful_runs': len(successful_results),
            'failed_runs': len(self.results) - len(successful_results)
        }
        
        # Determine verdict
        robust_success = (
            stats['mean_advantage'] > self.min_advantage_threshold and
            stats['confidence_interval_99'][0] > self.min_advantage_threshold and
            stats['pass_rate'] > 0.95
        )
        
        duration = time.time() - start_time
        
        # Final results
        final_results = {
            'phase': 'Statistical Bombardment',
            'verdict': 'ROBUST' if robust_success else 'FRAGILE',
            'stats': stats,
            'duration_seconds': duration,
            'trials_per_second': len(self.results) / duration,
            'checkpoints_saved': self.checkpoints_saved,
            'configuration': {
                'target_runs': self.target_runs,
                'confidence_level': self.confidence_level,
                'min_advantage_threshold': self.min_advantage_threshold,
                'n_workers': self.n_workers
            },
            'raw_results': self.results[:10]  # Sample of raw results
        }
        
        # Log summary
        self.log("="*60)
        self.log("PHASE 1 SUMMARY")
        self.log(f"Verdict: {final_results['verdict']}")
        self.log(f"Mean advantage: {stats['mean_advantage']:.3f}")
        self.log(f"99% CI: [{stats['confidence_interval_99'][0]:.3f}, {stats['confidence_interval_99'][1]:.3f}]")
        self.log(f"Pass rate: {stats['pass_rate']:.1%}")
        self.log(f"Duration: {duration:.1f}s")
        self.log("="*60)
        
        return final_results


if __name__ == "__main__":
    # Test phase 1 independently
    print("Testing Phase 1: Statistical Bombardment")
    phase1 = Phase1StatisticalBombardment(quick_mode=True)
    results = phase1.execute_bombardment()
    
    print(f"\nFinal verdict: {results['verdict']}")
    if results['stats']:
        print(f"Mean advantage: {results['stats']['mean_advantage']:.3f}")
        print(f"Pass rate: {results['stats']['pass_rate']:.1%}")
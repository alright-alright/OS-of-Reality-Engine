#!/usr/bin/env python3
"""
AGGRESSIVE FALSIFICATION LAUNCHER
OS-of-Reality Universal Substrate Validation

Mission: Either PROVE the 95.1% MDL advantage is real and robust, 
or DESTROY the hypothesis through exhaustive testing.
"""

import os
import sys
import json
import time
import pickle
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from protocols.os_reality_protocol import OSRealityProtocol


class FalsificationLauncher:
    """
    Robust launcher with checkpoint system, frequent saves, and git integration
    """
    
    def __init__(self, base_dir: str = "falsification_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create session directory with timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Checkpoint paths
        self.checkpoint_dir = self.session_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Results paths
        self.results_dir = self.session_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Logs
        self.log_file = self.session_dir / "falsification.log"
        
        # Status tracking
        self.status = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'phases_completed': [],
            'current_phase': None,
            'total_tests_run': 0,
            'last_checkpoint': None,
            'errors': []
        }
        
        self.log(f"Falsification Launcher initialized - Session: {self.session_id}")
        
    def log(self, message: str, level: str = "INFO"):
        """Thread-safe logging with timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Print to console
        print(log_entry.strip())
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def save_checkpoint(self, phase: str, data: Any, iteration: Optional[int] = None):
        """Save checkpoint with atomic write and backup"""
        checkpoint_name = f"{phase}_{iteration if iteration else 'final'}_{int(time.time())}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        try:
            # Write to temp file first (atomic operation)
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'phase': phase,
                    'iteration': iteration,
                    'timestamp': time.time(),
                    'data': data
                }, f)
            
            # Move temp to final (atomic on most filesystems)
            temp_path.rename(checkpoint_path)
            
            self.status['last_checkpoint'] = str(checkpoint_path)
            self.log(f"Checkpoint saved: {checkpoint_name}")
            
            # Auto-commit to git every 10 checkpoints
            if self.status['total_tests_run'] % 10 == 0:
                self.git_commit_results(f"Checkpoint: {phase} iteration {iteration}")
                
            return checkpoint_path
            
        except Exception as e:
            self.log(f"ERROR saving checkpoint: {e}", "ERROR")
            self.status['errors'].append(str(e))
            return None
    
    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict]:
        """Load checkpoint with error recovery"""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            self.log(f"Checkpoint loaded: {checkpoint_path.name}")
            return data
        except Exception as e:
            self.log(f"ERROR loading checkpoint: {e}", "ERROR")
            return None
    
    def git_commit_results(self, message: str):
        """Auto-commit results to git for safety"""
        try:
            # Add all results
            subprocess.run(['git', 'add', str(self.session_dir)], 
                         check=True, capture_output=True)
            
            # Commit with message
            commit_message = f"[AUTO] {message} - Session {self.session_id}"
            subprocess.run(['git', 'commit', '-m', commit_message], 
                         check=True, capture_output=True)
            
            self.log(f"Git commit successful: {message}")
            
        except subprocess.CalledProcessError as e:
            # Git commit failed, but don't crash the test
            self.log(f"Git commit failed (non-fatal): {e}", "WARNING")
    
    def save_results(self, phase: str, results: Dict):
        """Save results in multiple formats for redundancy"""
        timestamp = int(time.time())
        base_name = f"{phase}_{timestamp}"
        
        # Save as JSON (human-readable)
        json_path = self.results_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as pickle (preserves numpy arrays)
        pkl_path = self.results_dir / f"{base_name}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Update status
        self.status['phases_completed'].append(phase)
        self.save_status()
        
        self.log(f"Results saved: {base_name}")
        return json_path, pkl_path
    
    def save_status(self):
        """Save current launcher status"""
        status_path = self.session_dir / "status.json"
        with open(status_path, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)
    
    def run_with_recovery(self, func, *args, **kwargs):
        """Run function with error recovery and retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.log(f"Attempt {attempt+1} failed: {e}", "ERROR")
                self.status['errors'].append({
                    'function': func.__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'attempt': attempt + 1
                })
                
                if attempt < max_retries - 1:
                    self.log(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.log(f"All attempts failed for {func.__name__}", "ERROR")
                    raise
    
    def execute_full_falsification(self):
        """
        Execute all 5 phases of aggressive falsification
        """
        self.log("="*80)
        self.log("STARTING AGGRESSIVE FALSIFICATION PROTOCOL")
        self.log(f"Session ID: {self.session_id}")
        self.log(f"Results directory: {self.session_dir}")
        self.log("="*80)
        
        all_results = {}
        
        try:
            # Phase 1: Statistical Bombardment
            self.status['current_phase'] = 'phase1'
            self.log("\n" + "="*60)
            self.log("PHASE 1: STATISTICAL BOMBARDMENT")
            self.log("="*60)
            phase1_results = self.run_with_recovery(self.run_phase1_statistical_bombardment)
            all_results['phase1'] = phase1_results
            self.save_results('phase1', phase1_results)
            
            # Phase 2: Control Gauntlet
            self.status['current_phase'] = 'phase2'
            self.log("\n" + "="*60)
            self.log("PHASE 2: CONTROL GAUNTLET")
            self.log("="*60)
            phase2_results = self.run_with_recovery(self.run_phase2_control_gauntlet)
            all_results['phase2'] = phase2_results
            self.save_results('phase2', phase2_results)
            
            # Phase 3: Robustness Torture
            self.status['current_phase'] = 'phase3'
            self.log("\n" + "="*60)
            self.log("PHASE 3: ROBUSTNESS TORTURE TEST")
            self.log("="*60)
            phase3_results = self.run_with_recovery(self.run_phase3_robustness_torture)
            all_results['phase3'] = phase3_results
            self.save_results('phase3', phase3_results)
            
            # Phase 4: Independence Validation
            self.status['current_phase'] = 'phase4'
            self.log("\n" + "="*60)
            self.log("PHASE 4: INDEPENDENCE VALIDATION")
            self.log("="*60)
            phase4_results = self.run_with_recovery(self.run_phase4_independence_validation)
            all_results['phase4'] = phase4_results
            self.save_results('phase4', phase4_results)
            
            # Phase 5: Publication-Grade Statistics
            self.status['current_phase'] = 'phase5'
            self.log("\n" + "="*60)
            self.log("PHASE 5: PUBLICATION-GRADE STATISTICS")
            self.log("="*60)
            phase5_results = self.run_with_recovery(
                self.run_phase5_publication_stats, all_results
            )
            all_results['phase5'] = phase5_results
            self.save_results('phase5', phase5_results)
            
            # Final report
            self.generate_final_report(all_results)
            
            # Final git commit
            self.git_commit_results("COMPLETE: All falsification phases finished")
            
        except Exception as e:
            self.log(f"FATAL ERROR: {e}", "CRITICAL")
            self.log(traceback.format_exc(), "CRITICAL")
            self.save_status()
            raise
        
        finally:
            # Always save final status
            self.status['end_time'] = time.time()
            self.status['duration'] = self.status['end_time'] - self.status['start_time']
            self.save_status()
            
        return all_results
    
    def run_phase1_statistical_bombardment(self):
        """Phase 1: Run 1000+ independent trials"""
        from falsification.phase1_statistical import Phase1StatisticalBombardment
        
        phase1 = Phase1StatisticalBombardment(launcher=self)
        results = phase1.execute_bombardment()
        
        return results
    
    def run_phase2_control_gauntlet(self):
        """Phase 2: Aggressive control tests"""
        from falsification.phase2_controls import Phase2ControlGauntlet
        
        phase2 = Phase2ControlGauntlet(launcher=self)
        results = phase2.execute_gauntlet()
        
        return results
    
    def run_phase3_robustness_torture(self):
        """Phase 3: Torture tests"""
        from falsification.phase3_robustness import Phase3RobustnessTorture
        
        phase3 = Phase3RobustnessTorture(launcher=self)
        results = phase3.execute_torture()
        
        return results
    
    def run_phase4_independence_validation(self):
        """Phase 4: Cross-validation"""
        from falsification.phase4_independence import Phase4IndependenceValidation
        
        phase4 = Phase4IndependenceValidation(launcher=self)
        results = phase4.execute_validation()
        
        return results
    
    def run_phase5_publication_stats(self, all_phase_results):
        """Phase 5: Publication-grade statistics"""
        from falsification.phase5_publication import Phase5PublicationStats
        
        phase5 = Phase5PublicationStats(launcher=self)
        report = phase5.generate_final_report(all_phase_results)
        
        return report
    
    def generate_final_report(self, all_results):
        """Generate comprehensive final report"""
        self.log("\n" + "="*80)
        self.log("FINAL FALSIFICATION REPORT")
        self.log("="*80)
        
        # Extract key metrics
        phase1_verdict = all_results.get('phase1', {}).get('verdict', 'UNKNOWN')
        phase2_verdict = all_results.get('phase2', {}).get('verdict', 'UNKNOWN')
        phase5_verdict = all_results.get('phase5', {}).get('publication_verdict', {}).get('verdict', 'UNKNOWN')
        
        self.log(f"Phase 1 (Statistical): {phase1_verdict}")
        self.log(f"Phase 2 (Controls): {phase2_verdict}")
        self.log(f"Phase 5 (Publication): {phase5_verdict}")
        
        # Save comprehensive report
        report_path = self.session_dir / "FINAL_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'duration': self.status.get('duration', 0),
                'total_tests': self.status['total_tests_run'],
                'phases_completed': self.status['phases_completed'],
                'errors_encountered': len(self.status['errors']),
                'results_summary': {
                    'phase1_verdict': phase1_verdict,
                    'phase2_verdict': phase2_verdict,
                    'phase5_verdict': phase5_verdict
                },
                'full_results': all_results
            }, f, indent=2, default=str)
        
        self.log(f"Final report saved: {report_path}")
        
        return report_path


def main():
    """Main entry point for falsification launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OS-of-Reality Falsification Launcher')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--phase', type=str, help='Run specific phase only')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer iterations)')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = FalsificationLauncher()
    
    try:
        if args.resume:
            launcher.log(f"Resuming from checkpoint: {args.resume}")
            # TODO: Implement resume logic
            
        if args.phase:
            launcher.log(f"Running phase: {args.phase}")
            # TODO: Implement single phase execution
            
        # Run full falsification
        results = launcher.execute_full_falsification()
        
        # Print final verdict
        verdict = results.get('phase5', {}).get('publication_verdict', {}).get('verdict', 'UNKNOWN')
        print(f"\n{'='*80}")
        print(f"FINAL VERDICT: {verdict}")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        launcher.log("Interrupted by user", "WARNING")
        launcher.save_status()
        sys.exit(1)
        
    except Exception as e:
        launcher.log(f"Fatal error: {e}", "CRITICAL")
        launcher.save_status()
        raise


if __name__ == "__main__":
    main()
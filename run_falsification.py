#!/usr/bin/env python3
"""
Run aggressive falsification protocols for OS-of-Reality
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from falsification.launcher import FalsificationLauncher


def main():
    parser = argparse.ArgumentParser(description='OS-of-Reality Falsification Suite')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test mode (fewer iterations)')
    parser.add_argument('--phase', type=str, 
                       help='Run specific phase only (1-5)')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OS-OF-REALITY AGGRESSIVE FALSIFICATION SUITE")
    print("="*80)
    print()
    print("Mission: Either PROVE the 95.1% MDL advantage is real and robust,")
    print("         or DESTROY the hypothesis through exhaustive testing.")
    print()
    print("This will run:")
    print("  Phase 1: Statistical Bombardment (1000+ trials)")
    print("  Phase 2: Control Gauntlet (8 hostile controls)")
    print("  Phase 3: Robustness Torture Test")
    print("  Phase 4: Independence Validation")
    print("  Phase 5: Publication-Grade Statistics")
    print()
    
    if args.quick or args.phase:
        print("*** AUTO MODE: Skipping confirmation prompt ***")
        print("Auto-continuing...")
        print()
    else:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create launcher
    launcher = FalsificationLauncher()
    
    try:
        if args.phase:
            # Run specific phase
            phase_num = int(args.phase)
            print(f"\nRunning Phase {phase_num} only...")
            
            if phase_num == 1:
                from falsification.phase1_statistical import Phase1StatisticalBombardment
                phase1 = Phase1StatisticalBombardment(launcher=launcher, quick_mode=args.quick)
                results = phase1.execute_bombardment()
                launcher.save_results('phase1', results)
                
            elif phase_num == 2:
                from falsification.phase2_controls import Phase2ControlGauntlet
                phase2 = Phase2ControlGauntlet(launcher=launcher)
                results = phase2.execute_gauntlet()
                launcher.save_results('phase2', results)
                
            elif phase_num == 3:
                from falsification.phase3_robustness import Phase3RobustnessTorture
                phase3 = Phase3RobustnessTorture(launcher=launcher)
                results = phase3.execute_torture()
                launcher.save_results('phase3', results)
                
            elif phase_num == 4:
                from falsification.phase4_independence import Phase4IndependenceValidation
                phase4 = Phase4IndependenceValidation(launcher=launcher)
                results = phase4.execute_validation()
                launcher.save_results('phase4', results)
                
            elif phase_num == 5:
                print("Phase 5 requires results from all previous phases")
                return
                
            else:
                print(f"Invalid phase number: {phase_num}")
                return
                
            print(f"\nPhase {phase_num} complete!")
            print(f"Results saved in: {launcher.session_dir}")
            
        else:
            # Run full falsification suite
            print("\nStarting full falsification protocol...")
            print(f"Session directory: {launcher.session_dir}")
            print()
            
            # Modify launcher for quick mode
            if args.quick:
                # Override phase configurations for quick testing
                import falsification.phase1_statistical as phase1_module
                original_init = phase1_module.Phase1StatisticalBombardment.__init__
                
                def quick_init(self, launcher=None, quick_mode=True):
                    original_init(self, launcher, quick_mode=True)
                
                phase1_module.Phase1StatisticalBombardment.__init__ = quick_init
            
            # Execute full protocol
            results = launcher.execute_full_falsification()
            
            # Print final verdict
            verdict = results.get('phase5', {}).get('publication_verdict', {}).get('verdict', 'UNKNOWN')
            
            print("\n" + "="*80)
            print("FALSIFICATION COMPLETE")
            print("="*80)
            print(f"Final Verdict: {verdict}")
            print(f"Full results in: {launcher.session_dir}")
            print("="*80)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        launcher.save_status()
        print(f"Progress saved in: {launcher.session_dir}")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        launcher.save_status()
        print(f"Error details saved in: {launcher.session_dir}")
        raise


if __name__ == "__main__":
    main()
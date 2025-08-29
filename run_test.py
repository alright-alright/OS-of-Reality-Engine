#!/usr/bin/env python3
"""
OS-of-Reality Test Runner
Run the T0 compression test to verify universal substrate hypothesis
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocols.os_reality_protocol import OSRealityProtocol
import json

if __name__ == "__main__":
    print("Starting OS-of-Reality Falsification Protocol...")
    
    # Run the protocol
    protocol = OSRealityProtocol(seed=42)
    results = protocol.run_full_protocol()
    
    # Save results
    output_file = "os_reality_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    if results['protocol_passed']:
        print("\n🎉 SUCCESS: Universal substrate hypothesis supported!")
        print(f"   MDL Advantage: {results['tier_results']['T0']['mdl_advantage']*100:.1f}%")
    else:
        print("\n⚠️ FAILED: Universal substrate hypothesis not supported")
        print("   Check results file for details")
# protocols/tier_implementations/t0_compression.py
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error
import json

from core.primitives import DimensionalState

class T0CompressionTest:
    """MDL compression test - T0 of falsification protocol"""
    
    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold  # 10% advantage required
        
    def run_test(self, domain_adapters: Dict, sequences: Dict) -> Dict:
        """
        Test if unified model compresses better than separate models
        Returns: {'passed': bool, 'mdl_advantage': float, 'details': dict}
        """
        
        results = {
            'passed': False,
            'mdl_advantage': 0.0,
            'unified_mdl': 0.0,
            'separate_mdl': 0.0,
            'details': {}
        }
        
        try:
            # 1. Train unified model on ALL domains simultaneously
            unified_model = self._train_unified_model(domain_adapters, sequences)
            unified_mdl = self._calculate_mdl(unified_model, sequences)
            
            # 2. Train separate models for each domain
            separate_models = {}
            separate_mdls = []
            
            for domain_name, adapter in domain_adapters.items():
                if domain_name in sequences:
                    model = self._train_domain_model(adapter, sequences[domain_name])
                    separate_models[domain_name] = model
                    domain_mdl = self._calculate_mdl(model, {domain_name: sequences[domain_name]})
                    separate_mdls.append(domain_mdl)
            
            total_separate_mdl = sum(separate_mdls)
            
            # 3. Calculate advantage
            mdl_advantage = (total_separate_mdl - unified_mdl) / total_separate_mdl
            
            # 4. Check if passes threshold
            passed = mdl_advantage >= self.threshold
            
            results.update({
                'passed': passed,
                'mdl_advantage': mdl_advantage,
                'unified_mdl': unified_mdl,
                'separate_mdl': total_separate_mdl,
                'details': {
                    'unified_params': unified_model['param_count'],
                    'separate_params': sum(m['param_count'] for m in separate_models.values()),
                    'domain_mdls': {name: mdl for name, mdl in zip(domain_adapters.keys(), separate_mdls)},
                    'threshold_met': passed,
                    'advantage_pct': mdl_advantage * 100
                }
            })
            
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def _train_unified_model(self, adapters: Dict, sequences: Dict) -> Dict:
        """Train single model on all domain sequences"""
        
        # Combine all sequences using UMST mathematical primitives
        all_data = []
        all_targets = []
        
        for domain_name, domain_sequences in sequences.items():
            adapter = adapters[domain_name]
            
            # Convert sequences to unified mathematical representation
            for seq in domain_sequences:
                # Use DimensionalState to encode sequence
                state = DimensionalState(seq.shape[1], seed=42)
                state.state_vector = np.mean(seq, axis=0)  # Aggregate features
                
                # Flatten for model input
                features = state.state_vector
                target = np.var(seq, axis=0)[0]  # Predict variance of first channel
                
                all_data.append(features)
                all_targets.append(target)
        
        # Simple linear model (can be made more sophisticated)
        X = np.array(all_data)
        y = np.array(all_targets)
        
        # Least squares solution
        params = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate predictions and error
        predictions = X @ params
        mse = mean_squared_error(y, predictions)
        
        return {
            'params': params,
            'param_count': len(params),
            'mse': mse,
            'type': 'unified'
        }
    
    def _train_domain_model(self, adapter, domain_sequences: List[np.ndarray]) -> Dict:
        """Train model on single domain"""
        
        data = []
        targets = []
        
        for seq in domain_sequences:
            # Same encoding as unified model
            state = DimensionalState(seq.shape[1], seed=42)
            state.state_vector = np.mean(seq, axis=0)
            
            features = state.state_vector
            target = np.var(seq, axis=0)[0]
            
            data.append(features)
            targets.append(target)
        
        X = np.array(data)
        y = np.array(targets)
        
        params = np.linalg.lstsq(X, y, rcond=None)[0]
        predictions = X @ params
        mse = mean_squared_error(y, predictions)
        
        return {
            'params': params,
            'param_count': len(params),
            'mse': mse,
            'type': 'domain_specific'
        }
    
    def _calculate_mdl(self, model: Dict, sequences: Dict) -> float:
        """Calculate Minimum Description Length"""
        
        # MDL = Model complexity + Data encoding cost
        param_cost = model['param_count'] * np.log2(len(model['params']) + 1)
        data_cost = -np.log2(1.0 / (model['mse'] + 1e-10))  # Encoding cost based on error
        
        mdl = param_cost + data_cost
        return mdl
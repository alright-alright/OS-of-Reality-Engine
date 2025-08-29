# protocols/tier_implementations/t0_compression_fixed.py
"""
FIXED T0 Compression Test
Actually measures sequence compression, not just parameter count
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import json

from core.primitives import DimensionalState, TemporalDynamics


class T0CompressionTestFixed:
    """Fixed MDL compression test that actually measures compression"""
    
    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        
    def run_test(self, domain_adapters: Dict, sequences: Dict) -> Dict:
        """
        Test if unified model compresses better than separate models
        This time, we actually model the SEQUENCES, not just their means!
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
            unified_model = self._train_unified_sequence_model(domain_adapters, sequences)
            unified_mdl = self._calculate_sequence_mdl(unified_model, sequences)
            
            # 2. Train separate models for each domain
            separate_models = {}
            separate_mdls = []
            
            for domain_name, adapter in domain_adapters.items():
                if domain_name in sequences:
                    model = self._train_domain_sequence_model(adapter, sequences[domain_name])
                    separate_models[domain_name] = model
                    domain_mdl = self._calculate_sequence_mdl(model, {domain_name: sequences[domain_name]})
                    separate_mdls.append(domain_mdl)
            
            total_separate_mdl = sum(separate_mdls)
            
            # 3. Calculate advantage
            # Note: Lower MDL is better, so advantage = (separate - unified) / separate
            mdl_advantage = (total_separate_mdl - unified_mdl) / abs(total_separate_mdl) if total_separate_mdl != 0 else 0
            
            # 4. Check if passes threshold
            passed = mdl_advantage >= self.threshold
            
            results.update({
                'passed': passed,
                'mdl_advantage': mdl_advantage,
                'unified_mdl': unified_mdl,
                'separate_mdl': total_separate_mdl,
                'details': {
                    'unified_params': unified_model['param_count'],
                    'unified_reconstruction_error': unified_model['reconstruction_error'],
                    'separate_params': sum(m['param_count'] for m in separate_models.values()),
                    'separate_reconstruction_errors': {
                        name: m['reconstruction_error'] 
                        for name, m in separate_models.items()
                    },
                    'domain_mdls': {name: mdl for name, mdl in zip(domain_adapters.keys(), separate_mdls)},
                    'threshold_met': passed,
                    'advantage_pct': mdl_advantage * 100
                }
            })
            
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def _train_unified_sequence_model(self, adapters: Dict, sequences: Dict) -> Dict:
        """
        Train a unified model that actually models SEQUENCES, not just means
        Uses PCA to find shared dynamics across all domains
        """
        
        # Collect all sequences
        all_sequences = []
        sequence_lengths = []
        
        for domain_name, domain_sequences in sequences.items():
            for seq in domain_sequences:
                all_sequences.append(seq)
                sequence_lengths.append(len(seq))
        
        # Stack all sequences
        max_len = max(sequence_lengths)
        n_features = all_sequences[0].shape[1]
        
        # Pad sequences to same length
        padded_sequences = []
        for seq in all_sequences:
            if len(seq) < max_len:
                padding = np.zeros((max_len - len(seq), n_features))
                padded = np.vstack([seq, padding])
            else:
                padded = seq[:max_len]
            padded_sequences.append(padded)
        
        # Convert to matrix (n_sequences, max_len * n_features)
        X = np.array([seq.flatten() for seq in padded_sequences])
        
        # Fit PCA to find shared components
        n_components = min(10, X.shape[0], X.shape[1])  # Use 10 components or less
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        
        # Transform and reconstruct
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction error
        reconstruction_error = mean_squared_error(X, X_reconstructed)
        
        # Calculate effective parameters
        # PCA params = n_components * (input_dim + 1)
        param_count = n_components * (X.shape[1] + 1)
        
        return {
            'type': 'unified_pca',
            'n_components': n_components,
            'param_count': param_count,
            'reconstruction_error': reconstruction_error,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': np.sum(pca.explained_variance_ratio_),
            'model': pca
        }
    
    def _train_domain_sequence_model(self, adapter, domain_sequences: List[np.ndarray]) -> Dict:
        """
        Train a model on a single domain's sequences
        """
        
        # Process sequences for this domain
        sequence_lengths = [len(seq) for seq in domain_sequences]
        max_len = max(sequence_lengths)
        n_features = domain_sequences[0].shape[1]
        
        # Pad sequences
        padded_sequences = []
        for seq in domain_sequences:
            if len(seq) < max_len:
                padding = np.zeros((max_len - len(seq), n_features))
                padded = np.vstack([seq, padding])
            else:
                padded = seq[:max_len]
            padded_sequences.append(padded)
        
        # Convert to matrix
        X = np.array([seq.flatten() for seq in padded_sequences])
        
        # Fit PCA for this domain
        n_components = min(10, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        
        # Transform and reconstruct
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction error
        reconstruction_error = mean_squared_error(X, X_reconstructed)
        
        # Calculate parameters
        param_count = n_components * (X.shape[1] + 1)
        
        return {
            'type': 'domain_pca',
            'n_components': n_components,
            'param_count': param_count,
            'reconstruction_error': reconstruction_error,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': np.sum(pca.explained_variance_ratio_),
            'model': pca
        }
    
    def _calculate_sequence_mdl(self, model: Dict, sequences: Dict) -> float:
        """
        Calculate MDL properly:
        MDL = log(model_complexity) + log(data_given_model)
        
        Where:
        - model_complexity = number of parameters * bits per parameter
        - data_given_model = reconstruction error in bits
        """
        
        # Model complexity in bits
        # Assume 32 bits per parameter (float32)
        model_bits = model['param_count'] * 32
        
        # Data complexity given model
        # Use reconstruction error as proxy for compression
        # Convert MSE to bits using entropy estimation
        reconstruction_error = model['reconstruction_error']
        
        # Estimate bits needed to encode residuals
        # Using Gaussian assumption: bits ≈ 0.5 * log2(2πe * variance)
        if reconstruction_error > 0:
            residual_bits = 0.5 * np.log2(2 * np.pi * np.e * reconstruction_error)
        else:
            residual_bits = 0  # Perfect reconstruction
        
        # Total sequence length
        total_length = sum(len(seq) * seq.shape[1] for seqs in sequences.values() for seq in seqs)
        
        # Total data bits
        data_bits = residual_bits * total_length
        
        # Total MDL
        mdl = model_bits + data_bits
        
        return mdl


class T0CompressionTestSimple:
    """Even simpler fix: just use reconstruction error directly"""
    
    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        
    def run_test(self, domain_adapters: Dict, sequences: Dict) -> Dict:
        """
        Simple test: Can a unified model reconstruct all domains better than separate models?
        """
        
        # For unified: train on all, test on all
        all_data = []
        for domain_seqs in sequences.values():
            for seq in domain_seqs:
                all_data.append(seq.flatten()[:100])  # Take first 100 points
        
        all_data = np.array(all_data)
        
        # Simple unified model: PCA with limited components
        n_components = min(5, all_data.shape[0] - 1, all_data.shape[1] - 1)
        pca_unified = PCA(n_components=n_components, random_state=42)
        pca_unified.fit(all_data)
        unified_reconstructed = pca_unified.inverse_transform(pca_unified.transform(all_data))
        unified_error = mean_squared_error(all_data, unified_reconstructed)
        
        # Separate models
        separate_errors = []
        for domain_name, domain_seqs in sequences.items():
            domain_data = []
            for seq in domain_seqs:
                domain_data.append(seq.flatten()[:100])
            domain_data = np.array(domain_data)
            
            n_comp_sep = min(5, domain_data.shape[0] - 1, domain_data.shape[1] - 1)
            pca_separate = PCA(n_components=n_comp_sep, random_state=42)
            pca_separate.fit(domain_data)
            separate_reconstructed = pca_separate.inverse_transform(pca_separate.transform(domain_data))
            separate_error = mean_squared_error(domain_data, separate_reconstructed)
            separate_errors.append(separate_error)
        
        avg_separate_error = np.mean(separate_errors)
        
        # Advantage: if unified has HIGHER error, it's actually worse!
        # We want unified to have LOWER error
        advantage = (avg_separate_error - unified_error) / avg_separate_error if avg_separate_error > 0 else 0
        
        return {
            'passed': advantage > self.threshold,
            'mdl_advantage': advantage,
            'unified_error': unified_error,
            'separate_error': avg_separate_error,
            'interpretation': 'Positive advantage means unified is better (lower error)'
        }
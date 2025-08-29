"""
Core Mathematical Primitives - UMST Foundation
These 5 classes are used IDENTICALLY by biology, cognition, and all other domains
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from enum import Enum


class DimensionalState:
    """N-dimensional state representation used by both biological and cognitive systems."""
    
    def __init__(self, dimensions: int, seed: Optional[int] = None):
        self.dimensions = dimensions
        self.rng = np.random.RandomState(seed)
        self.state_vector = np.zeros(dimensions)
        self.bounds = np.array([(-np.inf, np.inf)] * dimensions)
        self.metadata = {}
        
    def initialize(self, init_type: str = "gaussian", **kwargs) -> np.ndarray:
        """Initialize state with various distributions."""
        if init_type == "gaussian":
            mean = kwargs.get("mean", 0)
            std = kwargs.get("std", 1)
            self.state_vector = self.rng.normal(mean, std, self.dimensions)
        elif init_type == "uniform":
            low = kwargs.get("low", -1)
            high = kwargs.get("high", 1)
            self.state_vector = self.rng.uniform(low, high, self.dimensions)
        elif init_type == "sparse":
            sparsity = kwargs.get("sparsity", 0.1)
            mask = self.rng.random(self.dimensions) < sparsity
            self.state_vector = mask * self.rng.randn(self.dimensions)
        elif init_type == "custom":
            self.state_vector = np.array(kwargs.get("values", np.zeros(self.dimensions)))
        
        self._apply_bounds()
        return self.state_vector
    
    def _apply_bounds(self):
        """Apply dimensional bounds to state vector."""
        for i, (low, high) in enumerate(self.bounds):
            self.state_vector[i] = np.clip(self.state_vector[i], low, high)
    
    def set_bounds(self, dimension: int, low: float, high: float):
        """Set bounds for specific dimension."""
        self.bounds[dimension] = (low, high)
        self._apply_bounds()
    
    def project(self, target_dims: int) -> 'DimensionalState':
        """Project state to different dimensional space."""
        new_state = DimensionalState(target_dims)
        if target_dims <= self.dimensions:
            new_state.state_vector = self.state_vector[:target_dims].copy()
        else:
            new_state.state_vector[:self.dimensions] = self.state_vector
        return new_state
    
    def to_dict(self) -> Dict:
        return {
            "dimensions": self.dimensions,
            "state_sample": self.state_vector[:10].tolist() if len(self.state_vector) >= 10 else self.state_vector.tolist(),
            "bounds_sample": self.bounds[:5].tolist() if len(self.bounds) >= 5 else self.bounds.tolist(),
            "metadata": self.metadata
        }


class ConstraintType(Enum):
    CONSERVATION = "conservation"
    HOMEOSTASIS = "homeostasis" 
    VIABILITY = "viability"
    COHERENCE = "coherence"


@dataclass
class Constraint:
    name: str
    constraint_type: ConstraintType
    function: Callable
    target_value: Optional[float] = None
    tolerance: float = 0.1
    weight: float = 1.0
    active: bool = True


class ConstraintEngine:
    """Unified constraint satisfaction used by both biological and cognitive systems."""
    
    def __init__(self):
        self.constraints: Dict[str, Constraint] = {}
        
    def add_conservation_constraint(self, name: str, func: Callable, target_value: float, tolerance: float = 0.1):
        """Add conservation constraint (e.g., mass, energy, information)."""
        constraint = Constraint(
            name=name,
            constraint_type=ConstraintType.CONSERVATION,
            function=func,
            target_value=target_value,
            tolerance=tolerance
        )
        self.constraints[name] = constraint
    
    def add_homeostasis_constraint(self, name: str, dimension: int, low: float, high: float):
        """Add homeostasis constraint (e.g., temperature, pH, coherence)."""
        def bounds_check(state_vector):
            return low <= state_vector[dimension] <= high
            
        constraint = Constraint(
            name=name,
            constraint_type=ConstraintType.HOMEOSTASIS,
            function=bounds_check,
            target_value=None,
            tolerance=0.0
        )
        self.constraints[name] = constraint
    
    def validate(self, state: DimensionalState) -> Tuple[bool, List[str]]:
        """Validate state against all active constraints."""
        violations = []
        
        for name, constraint in self.constraints.items():
            if not constraint.active:
                continue
                
            try:
                if constraint.constraint_type == ConstraintType.CONSERVATION:
                    actual_value = constraint.function(state.state_vector)
                    if abs(actual_value - constraint.target_value) > constraint.tolerance:
                        violations.append(f"{name}: {actual_value:.3f} != {constraint.target_value:.3f}")
                        
                elif constraint.constraint_type == ConstraintType.HOMEOSTASIS:
                    if not constraint.function(state.state_vector):
                        violations.append(f"{name}: bounds violation")
                        
            except Exception as e:
                violations.append(f"{name}: evaluation error - {str(e)}")
        
        return len(violations) == 0, violations
    
    def get_constraint_space(self) -> Dict:
        return {
            name: {
                "type": constraint.constraint_type.value,
                "target": constraint.target_value,
                "tolerance": constraint.tolerance,
                "active": constraint.active
            }
            for name, constraint in self.constraints.items()
        }


class ConnectivityGraph:
    """Graph structures used by both biological and cognitive systems."""
    
    def __init__(self, num_nodes: int, seed: Optional[int] = None):
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(seed)
        self.adjacency = np.zeros((num_nodes, num_nodes))
        self.node_features = np.zeros((num_nodes, 10))  # Default feature dim
        
    def small_world_init(self, k: int, p: float):
        """Initialize as small-world network (brain-like)."""
        # Ring lattice
        for i in range(self.num_nodes):
            for j in range(1, k//2 + 1):
                self.adjacency[i, (i+j) % self.num_nodes] = 1
                self.adjacency[i, (i-j) % self.num_nodes] = 1
        
        # Rewiring
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if self.adjacency[i, j] == 1 and self.rng.random() < p:
                    k_new = self.rng.randint(0, self.num_nodes)
                    if k_new != i and self.adjacency[i, k_new] == 0:
                        self.adjacency[i, j] = 0
                        self.adjacency[j, i] = 0
                        self.adjacency[i, k_new] = 1
                        self.adjacency[k_new, i] = 1
                        
    def scale_free_init(self, m: int):
        """Initialize as scale-free network (semantic-like)."""
        # Start with m0 = m nodes fully connected
        for i in range(m):
            for j in range(i+1, m):
                self.adjacency[i, j] = 1
                self.adjacency[j, i] = 1
        
        # Add remaining nodes with preferential attachment
        for i in range(m, self.num_nodes):
            degrees = np.sum(self.adjacency, axis=1)
            if np.sum(degrees[:i]) > 0:
                probabilities = degrees[:i] / np.sum(degrees[:i])
                targets = self.rng.choice(i, size=min(m, i), replace=False, p=probabilities)
            else:
                targets = self.rng.choice(i, size=min(m, i), replace=False)
                
            for target in targets:
                self.adjacency[i, target] = 1
                self.adjacency[target, i] = 1
    
    def clustering_coefficient(self) -> float:
        """Calculate clustering coefficient."""
        clustering = 0.0
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency[i] > 0)[0]
            if len(neighbors) < 2:
                continue
                
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = 0
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    if self.adjacency[neighbors[j], neighbors[k]] > 0:
                        actual_edges += 1
            
            clustering += actual_edges / possible_edges if possible_edges > 0 else 0
            
        return clustering / self.num_nodes if self.num_nodes > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": int(np.sum(self.adjacency > 0) / 2),
            "clustering": self.clustering_coefficient(),
            "density": np.sum(self.adjacency > 0) / (self.num_nodes * (self.num_nodes - 1)) if self.num_nodes > 1 else 0
        }


class TemporalDynamics:
    """Temporal evolution used by both biological and cognitive systems."""
    
    def __init__(self, state_dim: int, seed: Optional[int] = None):
        self.state_dim = state_dim
        self.rng = np.random.RandomState(seed)
        self.dynamics_matrix = np.eye(state_dim)
        self.dynamics_type = "identity"
        
    def linear_dynamics(self, A: np.ndarray):
        """Set linear dynamics: dx/dt = Ax."""
        assert A.shape == (self.state_dim, self.state_dim)
        self.dynamics_matrix = A.copy()
        self.dynamics_type = "linear"
        
    def oscillatory_dynamics(self, frequencies: np.ndarray, amplitudes: np.ndarray):
        """Set oscillatory dynamics with given frequencies and amplitudes."""
        assert len(frequencies) == len(amplitudes)
        # Create block diagonal oscillator matrix
        n_oscs = len(frequencies)
        A = np.zeros((self.state_dim, self.state_dim))
        
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            if 2*i+1 < self.state_dim:
                # 2x2 oscillator block
                A[2*i, 2*i+1] = freq * amp
                A[2*i+1, 2*i] = -freq * amp
                
        self.dynamics_matrix = A
        self.dynamics_type = "oscillatory"
    
    def simulate(self, initial_state: np.ndarray, duration: float, dt: float = 0.01) -> np.ndarray:
        """Simulate temporal evolution."""
        steps = int(duration / dt)
        trajectory = np.zeros((steps, len(initial_state)))
        state = initial_state.copy()
        
        for i in range(steps):
            trajectory[i] = state
            # Simple Euler integration
            if len(state) == self.state_dim:
                state = state + dt * (self.dynamics_matrix @ state)
            else:
                # Handle dimension mismatch by using subset of dynamics matrix
                dim = min(len(state), self.state_dim)
                state[:dim] = state[:dim] + dt * (self.dynamics_matrix[:dim, :dim] @ state[:dim])
            
        return trajectory
    
    def to_dict(self) -> Dict:
        eigenvals = np.linalg.eigvals(self.dynamics_matrix)
        return {
            "state_dim": self.state_dim,
            "dynamics_type": self.dynamics_type,
            "matrix_norm": float(np.linalg.norm(self.dynamics_matrix)),
            "eigenvalue_real_parts": np.real(eigenvals)[:5].tolist() if len(eigenvals) >= 5 else np.real(eigenvals).tolist()
        }


class AdaptationEngine:
    """Optimization and adaptation used by both biological and cognitive systems."""
    
    def __init__(self, param_dim: int, seed: Optional[int] = None):
        self.param_dim = param_dim
        self.rng = np.random.RandomState(seed)
        self.objective_func = None
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.history = []
        
    def define_objective(self, objective_func: Callable[[np.ndarray], float]):
        """Define objective function to optimize."""
        self.objective_func = objective_func
        
    def gradient_descent(self, initial_params: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """Perform gradient descent optimization."""
        if self.objective_func is None:
            raise ValueError("Objective function not defined")
            
        params = initial_params.copy()
        velocity = np.zeros_like(params)
        self.history = []
        
        for i in range(max_iterations):
            # Numerical gradient
            grad = self._numerical_gradient(params)
            
            # Momentum update
            velocity = self.momentum * velocity - self.learning_rate * grad
            params += velocity
            
            # Record objective value
            obj_val = self.objective_func(params)
            self.history.append(obj_val)
            
            # Check convergence
            if np.linalg.norm(grad) < tolerance:
                break
                
        return params
    
    def _numerical_gradient(self, params: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Calculate numerical gradient."""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            grad[i] = (self.objective_func(params_plus) - self.objective_func(params_minus)) / (2 * epsilon)
            
        return grad
    
    def to_dict(self) -> Dict:
        return {
            "param_dim": self.param_dim,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "optimization_history_length": len(self.history),
            "final_objective": self.history[-1] if self.history else None
        }
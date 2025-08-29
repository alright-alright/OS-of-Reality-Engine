"""
Domain Adapters for OS-of-Reality
Each domain uses IDENTICAL mathematical primitives from UMST
"""

from .biological.bio_adapter import BiologicalAdapter
from .geological.geo_adapter import GeologicalAdapter
from .cosmological.cosmo_adapter import CosmologicalAdapter

__all__ = [
    'BiologicalAdapter',
    'GeologicalAdapter',
    'CosmologicalAdapter'
]
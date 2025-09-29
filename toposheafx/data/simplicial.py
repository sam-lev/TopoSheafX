"""
Data structures for simplicial complex neural networks.
"""

import torch
from typing import Dict, Optional


class SimplicialFeatures:
    """
    Container for features across simplex dimensions.

    Maintains separate feature tensors for vertices (0-simplices),
    edges (1-simplices), triangles (2-simplices), etc.

    Args:
        vertices: Vertex features [num_vertices, dim_0]
        edges: Edge features [num_edges, dim_1]
        triangles: Triangle features [num_triangles, dim_2]
        tetrahedra: Tetrahedron features [num_tetrahedra, dim_3]

    Example:
        >>> vertices = torch.randn(6, 8)
        >>> edges = torch.randn(8, 16)
        >>> features = SimplicialFeatures(vertices=vertices, edges=edges)
        >>> features[0].shape
        torch.Size([6, 8])
    """

    def __init__(self, vertices=None, edges=None, triangles=None, tetrahedra=None, **kwargs):
        """
        Initialize with either named arguments or dimension-indexed kwargs.

        Args:
            vertices: 0-simplex features
            edges: 1-simplex features
            triangles: 2-simplex features
            tetrahedra: 3-simplex features
            **kwargs: Can also pass dim_0, dim_1, dim_2, etc.
        """
        self.features = {}

        # Handle named arguments
        if vertices is not None:
            self.features[0] = vertices
        if edges is not None:
            self.features[1] = edges
        if triangles is not None:
            self.features[2] = triangles
        if tetrahedra is not None:
            self.features[3] = tetrahedra

        # Handle dimension-indexed arguments (dim_0, dim_1, etc.)
        for key, value in kwargs.items():
            if key.startswith('dim_') and value is not None:
                dim = int(key.split('_')[1])
                self.features[dim] = value

        self.dims = {k: v.shape[-1] for k, v in self.features.items()}
        self.max_dim = max(self.features.keys()) if self.features else 0

    @classmethod
    def from_dict(cls, feature_dict):
        """
        Create SimplicialFeatures from dictionary with integer keys.

        Args:
            feature_dict: dict with integer keys {0: vertices, 1: edges, ...}
        """
        kwargs = {}
        if 0 in feature_dict:
            kwargs['vertices'] = feature_dict[0]
        if 1 in feature_dict:
            kwargs['edges'] = feature_dict[1]
        if 2 in feature_dict:
            kwargs['triangles'] = feature_dict[2]
        if 3 in feature_dict:
            kwargs['tetrahedra'] = feature_dict[3]

        return cls(**kwargs)

    def __getitem__(self, dim: int) -> Optional[torch.Tensor]:
        """Get features for a specific dimension."""
        return self.features.get(dim)

    def __setitem__(self, dim: int, features: torch.Tensor):
        """Set features for a specific dimension."""
        self.features[dim] = features
        self.dims[dim] = features.shape[-1]
        self.max_dim = max(self.features.keys())

    def to(self, device):
        """Move all features to device."""
        self.features = {k: v.to(device) for k, v in self.features.items()}
        return self

    def __repr__(self):
        dims_str = ", ".join([f"{k}-simplices: {v.shape}"
                              for k, v in self.features.items()])
        return f"SimplicialFeatures({dims_str})"
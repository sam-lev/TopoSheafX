# message_passing.py
"""
Final implementation of copresheaf message passing classes implementing
"Copresheaf Topological Neural Networks" (Hajij et al.).

This module provides the core message passing framework for copresheaf-based neural networks,
integrating the centralized Aggregation framework for optimal flexibility and mathematical rigor.

THEORETICAL FOUNDATION:
======================
These classes implement the mathematical framework from Propositions 1 and 3:

Proposition 1 (Single Neighborhood):
    h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

Proposition 3 (Multiple Neighborhoods):
    h_x^{l+1} = β(h_x^l, ⋀_{k=1}^n ⊕_{y∈N_k(x)} α_k(h_x^l, ρ_k_{y→x}(h_y^l)))

The Aggregation class handles the ⊕ operators with various permutation-invariant strategies.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing as PyGMessagePassing

from toposheafx.base.aggregation import Aggregation, AggregationFactory
from toposheafx.utils.guards import _check_index, _same_device


class MessagePassing(PyGMessagePassing):
    """
    Copresheaf Message Passing implementing Proposition 1 with centralized aggregation.

    MATHEMATICAL FRAMEWORK (Proposition 1):
    =======================================
    This class implements the fundamental copresheaf message passing update:

        h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

    COMPONENT BREAKDOWN:
    ===================

    1. COPRESHEAF MORPHISM ρ_{y→x}: F(y) → F(x)
       - Mathematical: Linear map between feature spaces (stalks) of cells
       - Implementation: self.copresheaf_map (learnable neural network)
       - Purpose: Transport features from source stalk F(y) to target stalk F(x)
       - Input: [h_y, h_x, edge_attr] (source + target features + edge attributes)
       - Output: ρ_{y→x}(h_y) ∈ F(x) (transported features in target space)

    2. MESSAGE FUNCTION α(h_x, ρ_{y→x}(h_y))
       - Mathematical: Combines target features with transported source features
       - Implementation: self.message_fn (learnable neural network)
       - Purpose: Create messages that respect both local and transported information
       - Input: [h_x, ρ_{y→x}(h_y)] (target + transported source)
       - Output: message_{y→x} (message from y to x)

    3. AGGREGATION ⊕_{(y→x)∈E} messages
       - Mathematical: Permutation-invariant combination of all messages to x
       - Implementation: self.aggregation (Aggregation class instance)
       - Purpose: Combine multiple incoming messages while preserving structure
       - Methods: sum, mean, max, attention, set2set, multi, etc.
       - Input: All messages directed to node x
       - Output: Aggregated message for x

    4. UPDATE FUNCTION β(h_x, aggregated_messages)
       - Mathematical: Final update combining old and new information
       - Implementation: self.update_fn (learnable neural network)
       - Purpose: Integrate aggregated messages with current node features
       - Input: [h_x^l, ⊕ messages] (current features + aggregated messages)
       - Output: h_x^{l+1} (updated features for next layer)

    COPRESHEAF ADVANTAGES:
    =====================
    - HETEROGENEOUS SPACES: Each cell can have different feature dimension
    - DIRECTIONAL FLOW: Asymmetric information transport via ρ_{y→x}
    - FLEXIBLE AGGREGATION: Various strategies for combining neighborhood info
    - TOPOLOGICAL AWARENESS: Respects combinatorial complex structure

    Args:
        in_dim (int): Dimension of input node features F(x)
        out_dim (int): Dimension of output node features F'(x)
        edge_attr_dim (int): Dimension of edge attributes (default: 0)
        aggr (Union[str, Aggregation]): Aggregation specification
        copresheaf_map_factory (Callable): Factory for ρ_{y→x} morphisms
        message_factory (Callable): Factory for α message functions
        update_factory (Callable): Factory for β update functions
        aggregation_config (dict): Configuration for aggregation methods
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            edge_attr_dim: int = 0,
            aggr: Union[str, Aggregation] = 'sum',
            copresheaf_map_factory: Optional[Callable] = None,
            message_factory: Optional[Callable] = None,
            update_factory: Optional[Callable] = None,
            aggregation_config: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(aggr=None, **kwargs)  # Disable PyG's built-in aggregation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim

        # Initialize aggregation framework (implements ⊕ from Proposition 1)
        self._init_aggregation(aggr, aggregation_config)

        # Initialize copresheaf components with default factories if not provided
        if copresheaf_map_factory is None:
            copresheaf_map_factory = self._default_copresheaf_factory
        if message_factory is None:
            message_factory = self._default_message_factory
        if update_factory is None:
            # For multi-aggregation, we need to adjust the update factory
            if hasattr(self, 'num_aggregation_methods'):
                update_factory = lambda in_d, out_d: self._multi_update_factory(
                    in_d, out_d, self.num_aggregation_methods
                )
            else:
                update_factory = self._default_update_factory

        # Core copresheaf components following Proposition 1
        self.copresheaf_map = copresheaf_map_factory(in_dim, out_dim, edge_attr_dim)  # ρ
        self.message_fn = message_factory(in_dim, out_dim)  # α
        self.update_fn = update_factory(in_dim, out_dim)  # β

    def _multi_update_factory(self, in_dim: int, out_dim: int, num_methods: int):
        """
        Update factory for multi-aggregation that handles concatenated aggregation outputs.
        """
        aggregated_dim = out_dim * num_methods
        return nn.Sequential(
            nn.Linear(in_dim + aggregated_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def _init_aggregation(self, aggr: Union[str, Aggregation], config: Optional[dict]):
        """
        Initialize the aggregation component implementing ⊕ from Proposition 1.

        MATHEMATICAL ROLE OF AGGREGATION:
        ================================
        The aggregation operator ⊕ is fundamental to copresheaf message passing:

        ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l))

        It must satisfy:
        - PERMUTATION INVARIANCE: f({m1, m2, m3}) = f({m3, m1, m2})
        - DIFFERENTIABILITY: Supports gradient-based learning
        - EXPRESSIVENESS: Can distinguish different message distributions

        AGGREGATION STRATEGIES FOR COPRESHEAVES:
        =======================================
        Different strategies suit different copresheaf contexts:

        - 'sum': Preserves total information, good for additive features
        - 'mean': Normalizes by neighborhood size, handles variable degree
        - 'attention': Learns importance weights, good for heterogeneous neighbors
        - 'set2set': Handles variable-size sets optimally
        - 'multi': Combines multiple strategies for rich representations
        - 'copresheaf_adaptive': Automatically chooses based on context

        Args:
            aggr: Aggregation specification (string or Aggregation instance)
            config: Additional configuration parameters
        """

        # Store the aggregation config for later use
        self.aggregation_config = config or {}

        if isinstance(aggr, Aggregation):
            # Use provided Aggregation instance directly
            self.aggregation = aggr
        elif isinstance(aggr, str):
            # Create Aggregation instance from string specification
            config = config or {}

            if aggr == 'copresheaf_adaptive':
                # Automatically choose appropriate aggregation for copresheaf context
                complexity = config.get('complexity', 'simple')
                self.aggregation = AggregationFactory.create_for_copresheaf(
                    stalk_dim=self.out_dim,
                    neighborhood_complexity=complexity,
                    dropout=config.get('dropout', 0.0)
                )
                # Track if this creates a multi-aggregation
                if complexity == 'complex':
                    self.num_aggregation_methods = 3  # sum, mean, std
            elif aggr in {'attention', 'set2set', 'lstm'}:
                # Learnable aggregation methods require feature dimension
                self.aggregation = Aggregation(
                    method=aggr,
                    dim=self.out_dim,
                    num_heads=config.get('num_heads', 1),
                    dropout=config.get('dropout', 0.0),
                    **{k: v for k, v in config.items() if k not in ['num_heads', 'dropout']}
                )
            elif aggr == 'multi':
                # Multi-aggregation combines several methods
                methods = config.get('methods', ['sum', 'mean', 'max'])
                self.aggregation = AggregationFactory.create_multi(methods, self.out_dim)
                # Store the number of methods for update function dimension calculation
                self.num_aggregation_methods = len(methods)
            else:
                # Basic aggregation methods (sum, mean, max, min, std, var)
                self.aggregation = Aggregation(method=aggr, dim=self.out_dim, **config)
        else:
            raise TypeError(f"aggr must be str or Aggregation instance, got {type(aggr)}")

    def _default_copresheaf_factory(self, in_dim: int, out_dim: int, edge_attr_dim: int):
        """
        Default factory for copresheaf morphism ρ_{y→x}: F(y) → F(x).

        MATHEMATICAL SPECIFICATION:
        ==========================
        The copresheaf morphism is the core innovation of the framework:

        ρ_{y→x}: F(y) → F(x)

        Unlike standard GNNs where all nodes share the same feature space,
        copresheaves allow heterogeneous feature spaces F(y) ≠ F(x).

        IMPLEMENTATION DESIGN:
        =====================
        Input: [h_y, h_x, edge_attr] - source + target + edge information
        Architecture: Linear → Tanh (nonlinear transport map)
        Output: Features in target space F(x)

        The inclusion of h_x (target features) makes the transport context-dependent,
        allowing the morphism to adapt based on the receiving cell's state.

        Args:
            in_dim: Dimension of input features
            out_dim: Dimension of transported features
            edge_attr_dim: Dimension of edge attributes

        Returns:
            Neural network implementing ρ_{y→x}
        """
        rho_input_dim = 2 * in_dim + edge_attr_dim  # [h_y, h_x, edge_attr]
        return nn.Sequential(
            nn.Linear(rho_input_dim, out_dim),
            nn.Tanh()  # Smooth nonlinearity for stable transport
        )

    def _default_message_factory(self, in_dim: int, out_dim: int):
        """
        Default factory for message function α(h_x, ρ_{y→x}(h_y)).

        MATHEMATICAL SPECIFICATION:
        ==========================
        The message function creates the actual messages sent between cells:

        α: F(x) × F(x) → Messages
        α(h_x, ρ_{y→x}(h_y)) = message from y to x

        DESIGN PRINCIPLES:
        =================
        - Combines current target state h_x with transported source ρ_{y→x}(h_y)
        - Creates messages that are contextually aware of both sender and receiver
        - Provides expressiveness for complex message interactions

        IMPLEMENTATION:
        ==============
        Input: [h_x, ρ_{y→x}(h_y)] - target + transported source
        Architecture: Linear → ReLU (expressive message creation)
        Output: Message vector

        Args:
            in_dim: Dimension of target features h_x
            out_dim: Dimension of output messages

        Returns:
            Neural network implementing α
        """
        return nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),  # [h_x, ρ(h_y)] → message
            nn.ReLU()  # Nonlinearity for expressive message interactions
        )

    def _default_update_factory(self, in_dim: int, out_dim: int):
        """
        Default factory for update function β(h_x, aggregated_messages).

        MATHEMATICAL SPECIFICATION:
        ==========================
        The update function performs the final integration step:

        β: F(x) × F(x) → F'(x)
        h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

        DESIGN RATIONALE:
        ================
        - Integrates old features h_x^l with new aggregated information
        - Allows learned control over update magnitude and direction
        - Two-layer design provides sufficient expressiveness
        - Residual-like structure preserves important information

        IMPLEMENTATION:
        ==============
        Input: [h_x^l, ⊕ messages] - current + aggregated messages
        Architecture: Linear → ReLU → Linear (expressive update dynamics)
        Output: Updated features h_x^{l+1}

        Args:
            in_dim: Dimension of current features h_x^l
            out_dim: Dimension of updated features h_x^{l+1}

        Returns:
            Neural network implementing β
        """
        return nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),  # [h_x, ⊕messages] → hidden
            nn.ReLU(),  # Nonlinearity for complex updates
            nn.Linear(out_dim, out_dim)  # hidden → h_x^{l+1}
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            size: Optional[tuple] = None
    ) -> Tensor:
        """
        Forward pass implementing complete copresheaf message passing (Proposition 1).

        EXECUTION FLOW:
        ==============

        Step 1: MESSAGE COMPUTATION
        For each edge (y→x):
        - Extract source h_y and target h_x features
        - Compute copresheaf transport: ρ_{y→x}(h_y)
        - Create message: α(h_x, ρ_{y→x}(h_y))

        Step 2: MESSAGE AGGREGATION
        For each target node x:
        - Collect all messages directed to x
        - Apply aggregation: ⊕_{(y→x)∈E} messages

        Step 3: FEATURE UPDATE
        For each node x:
        - Combine current features with aggregated messages
        - Apply update: h_x^{l+1} = β(h_x^l, ⊕ messages)

        MATHEMATICAL VERIFICATION:
        =========================
        This implementation directly realizes Proposition 1:
        h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

        Args:
            x: Node feature matrix [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges] (row=source, col=target)
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)
            size: Size of source and target nodes (optional)

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Input validation for copresheaf message passing
        _same_device(x, edge_index, edge_attr if edge_attr is not None else x)

        if edge_index.size(0) != 2:
            raise ValueError("edge_index must be [2, num_edges] for directed edges")

        row, col = edge_index
        _check_index(row, x.size(0), "source nodes")
        _check_index(col, x.size(0), "target nodes")

        # Execute copresheaf message passing via PyTorch Geometric's propagate framework
        # This handles the message/aggregate/update pipeline automatically
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        """
        Compute copresheaf messages α(h_x, ρ_{y→x}(h_y)) for each edge.

        MATHEMATICAL OPERATION:
        ======================
        For each directed edge y → x:
        1. Apply copresheaf morphism: ρ_{y→x}(h_y) (transport to target space)
        2. Create message: α(h_x, ρ_{y→x}(h_y)) (combine target + transported)

        TENSOR INDEXING:
        ===============
        In PyTorch Geometric's message passing convention:
        - x_i: Target node features (h_x)
        - x_j: Source node features (h_y)
        - Edge direction: j → i (source → target)

        Args:
            x_i: Target node features [num_edges, in_dim]
            x_j: Source node features [num_edges, in_dim]
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)

        Returns:
            Messages [num_edges, out_dim]
        """
        # Input validation
        _same_device(x_i, x_j, edge_attr if edge_attr is not None else x_i)

        if x_i.dim() != 2 or x_j.dim() != 2:
            raise ValueError("Node features must be 2D tensors [num_edges, feature_dim]")
        if x_i.size(0) != x_j.size(0):
            raise ValueError("Source and target features must have same number of edges")
        if x_i.size(1) != self.in_dim or x_j.size(1) != self.in_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.in_dim}")

        if edge_attr is not None:
            if edge_attr.dim() != 2 or edge_attr.size(0) != x_i.size(0):
                raise ValueError("Edge attributes must be [num_edges, edge_attr_dim]")
            if edge_attr.size(1) != self.edge_attr_dim:
                raise ValueError(f"Edge attribute dimension mismatch: "
                                 f"got {edge_attr.size(1)}, expected {self.edge_attr_dim}")

        # Step 1: Apply copresheaf morphism ρ_{j→i}: F(j) → F(i)
        if edge_attr is not None:
            rho_input = torch.cat([x_j, x_i, edge_attr], dim=-1)  # [h_j, h_i, edge_attr]
        else:
            rho_input = torch.cat([x_j, x_i], dim=-1)  # [h_j, h_i]

        transported_features = self.copresheaf_map(rho_input)  # ρ_{j→i}(h_j)

        # Step 2: Create messages α(h_i, ρ_{j→i}(h_j))
        message_input = torch.cat([x_i, transported_features], dim=-1)  # [h_i, ρ(h_j)]
        messages = self.message_fn(message_input)  # α(h_i, ρ_{j→i}(h_j))

        return messages

    def aggregate(
            self,
            inputs: Tensor,
            index: Tensor,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None
    ) -> Tensor:
        """
        Aggregate messages using the copresheaf Aggregation framework (⊕ operator).

        MATHEMATICAL OPERATION:
        ======================
        For each target node x, aggregate all incoming messages:

        ⊕_{(y→x)∈E} α(h_x, ρ_{y→x}(h_y))

        The Aggregation class provides various permutation-invariant strategies:
        - Basic: sum, mean, max, min
        - Statistical: std, var
        - Learnable: attention, set2set
        - Composite: multi-aggregation

        COPRESHEAF CONSIDERATIONS:
        =========================
        The aggregation must preserve the geometric structure of the target
        stalk F(x) while combining information from multiple neighbors.
        Different aggregation strategies are appropriate for different
        topological contexts and neighborhood complexities.

        Args:
            inputs: Messages to aggregate [num_edges, out_dim]
            index: Target node indices [num_edges]
            ptr: Pointer tensor for batching (unused by current Aggregation)
            dim_size: Number of target nodes (optional)

        Returns:
            Aggregated messages [num_nodes, out_dim]
        """
        return self.aggregation(inputs, index, dim_size)

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        """
        Update node features using copresheaf update function β.

        MATHEMATICAL OPERATION:
        ======================
        Final step of Proposition 1:

        h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

        Where:
        - x: Current node features h_x^l
        - inputs: Aggregated messages ⊕_{(y→x)∈E} α(...)
        - Output: Updated features h_x^{l+1}

        INTEGRATION STRATEGY:
        ====================
        The update function β learns how to optimally integrate:
        - Current node state (preserving identity)
        - Aggregated neighborhood information (enabling adaptation)

        Args:
            inputs: Aggregated messages [num_nodes, out_dim]
            x: Current node features [num_nodes, in_dim]

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        update_input = torch.cat([x, inputs], dim=-1)  # [h_x^l, ⊕ messages]
        return self.update_fn(update_input)  # β(h_x^l, ⊕ messages)

    def set_aggregation(self, new_aggregation: Union[str, Aggregation], config: Optional[dict] = None):
        """
        Dynamically change the aggregation method.

        This enables adaptive aggregation strategies during training,
        such as starting with simple aggregation and progressing to
        more complex methods as the model learns.

        Args:
            new_aggregation: New aggregation method or instance
            config: Configuration for new aggregation
        """
        self._init_aggregation(new_aggregation, config)

    def get_aggregation_info(self) -> Dict[str, Any]:
        """
        Get information about current aggregation configuration.

        Returns:
            Dictionary with aggregation details for monitoring/debugging
        """
        return {
            'method': self.aggregation.method,
            'dim': self.aggregation.dim,
            'num_heads': getattr(self.aggregation, 'num_heads', None),
            'class': type(self.aggregation).__name__,
            'parameters': sum(p.numel() for p in self.aggregation.parameters())
        }


class HigherOrderMessagePassing(nn.Module):
    """
    Higher-order copresheaf message passing implementing Proposition 3 with aggregation.

    MATHEMATICAL FRAMEWORK (Proposition 3):
    =======================================
    This class implements multi-neighborhood copresheaf message passing:

        h_x^{l+1} = β(h_x^l, ⋀_{k=1}^n ⊕_{y∈N_k(x)} α_k(h_x^l, ρ_k_{y→x}(h_y^l)))

    COMPONENT BREAKDOWN:
    ===================

    1. MULTIPLE NEIGHBORHOODS {N_k(x)}_{k=1}^n
       - Mathematical: Different topological relations (adjacency, incidence, etc.)
       - Implementation: edge_index_list[k] for each neighborhood type
       - Purpose: Capture multi-scale topological structure
       - Examples: N_1 = adjacency, N_2 = incidence, N_3 = co-boundary

    2. PER-NEIGHBORHOOD COPRESHEAF MAPS {ρ_k}_{k=1}^n
       - Mathematical: ρ_k_{y→x}: F(y) → F(x) for neighborhood k
       - Implementation: MessagePassing layers in neighborhood_layers
       - Purpose: Handle different transport rules per neighborhood type
       - Flexibility: Each neighborhood can have different aggregation

    3. INTRA-NEIGHBORHOOD AGGREGATION {⊕_k}_{k=1}^n
       - Mathematical: ⊕_{y∈N_k(x)} α_k(h_x, ρ_k_{y→x}(h_y)) for each k
       - Implementation: Aggregation instances within each MessagePassing layer
       - Purpose: Combine messages within each neighborhood type
       - Strategy: Different aggregation methods per neighborhood type

    4. INTER-NEIGHBORHOOD COMBINATION ⋀
       - Mathematical: ⋀_{k=1}^n (result from step 3)
       - Implementation: self._combine_neighborhoods + self.combine_fn
       - Purpose: Integrate information across different neighborhood types
       - Methods: concatenation, summation, learned weighting

    5. FINAL UPDATE β
       - Mathematical: β(h_x^l, ⋀(...))
       - Implementation: self.final_update
       - Purpose: Create final updated features from multi-scale information

    HIGHER-ORDER TOPOLOGY:
    =====================
    This framework naturally handles:
    - VERTICES (0-cells): Traditional graph nodes
    - EDGES (1-cells): Graph edges, mesh edges
    - FACES (2-cells): Triangular faces, polygonal faces
    - VOLUMES (3-cells): Tetrahedral cells, etc.
    - ARBITRARY k-CELLS: Any combinatorial complex structure

    Each cell type can have different:
    - Feature dimensions (heterogeneous stalks)
    - Transport rules (different ρ_k morphisms)
    - Aggregation strategies (different ⊕_k operators)

    MULTI-SCALE INFORMATION FLOW:
    ============================
    1. LOCAL SCALE: ⊕_k aggregates within each neighborhood N_k(x)
    2. INTERMEDIATE SCALE: ⋀ combines across neighborhood types
    3. GLOBAL SCALE: Multiple layers create hierarchical representations

    This enables simultaneous capture of:
    - Fine-grained local patterns (single neighborhoods)
    - Medium-scale structural patterns (neighborhood interactions)
    - Large-scale global patterns (multi-layer propagation)

    Args:
        in_dim (int): Input feature dimension F(x)
        out_dim (int): Output feature dimension F'(x)
        edge_attr_dim (int): Edge attribute dimension per neighborhood
        num_neighborhoods (int): Number of neighborhood types
        aggr (Union[str, Aggregation, List]): Aggregation config per neighborhood
        combine_method (str): Inter-neighborhood combination method
        aggregation_config (dict): Configuration for aggregation methods
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            edge_attr_dim: int = 0,
            num_neighborhoods: int = 1,
            aggr: Union[str, Aggregation, List[Union[str, Aggregation]]] = 'sum',
            combine_method: str = 'concat',
            aggregation_config: Optional[dict] = None,
            **kwargs
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neighborhoods = num_neighborhoods
        self.combine_method = combine_method

        # Initialize per-neighborhood message passing with custom aggregation
        self._init_neighborhood_layers(aggr, edge_attr_dim, aggregation_config)

        # Initialize inter-neighborhood combination function ⋀
        self._init_combination_function()

        # Final update function β(h_x, ⋀(...))
        self.final_update = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def _init_neighborhood_layers(
            self,
            aggr: Union[str, Aggregation, List],
            edge_attr_dim: int,
            config: Optional[dict]
    ):
        """
        Initialize MessagePassing layers for each neighborhood with tailored aggregation.

        NEIGHBORHOOD-SPECIFIC DESIGN:
        =============================
        Different neighborhood types in combinatorial complexes benefit from
        different aggregation strategies:

        - ADJACENCY (0-0 connections): Simple sum/mean for uniform neighborhoods
        - INCIDENCE (0-1 connections): Statistical aggregation for variable degrees
        - BOUNDARY (k-(k+1) connections): Attention for complex hierarchical relations
        - CO-BOUNDARY ((k+1)-k connections): Set2Set for variable cell counts

        MATHEMATICAL JUSTIFICATION:
        ==========================
        Each neighborhood N_k(x) has different structural properties:
        - Size distribution |N_k(x)|
        - Connectivity patterns
        - Semantic relationships
        - Topological constraints

        The aggregation ⊕_k should respect these properties for optimal performance.

        Args:
            aggr: Aggregation specification (single, list, or instance)
            edge_attr_dim: Edge attribute dimension
            config: Aggregation configuration parameters
        """
        config = config or {}

        # Parse aggregation specification for multiple neighborhoods
        if isinstance(aggr, (str, Aggregation)):
            # Same aggregation for all neighborhoods
            aggregations = [aggr] * self.num_neighborhoods
        elif isinstance(aggr, list):
            # Different aggregation per neighborhood
            if len(aggr) != self.num_neighborhoods:
                raise ValueError(f"Aggregation list length ({len(aggr)}) must match "
                                 f"num_neighborhoods ({self.num_neighborhoods})")
            aggregations = aggr
        else:
            raise TypeError("aggr must be str, Aggregation, or list thereof")

        # Create neighborhood-specific MessagePassing layers
        self.neighborhood_layers = nn.ModuleList()

        for k, neighborhood_aggr in enumerate(aggregations):
            # Create configuration for this specific neighborhood
            neighborhood_config = config.copy()

            # Add neighborhood-specific configuration if provided
            if f'neighborhood_{k}' in config:
                neighborhood_config.update(config[f'neighborhood_{k}'])

            # Optionally set complexity based on neighborhood type
            if 'complexity_per_neighborhood' in config:
                complexities = config['complexity_per_neighborhood']
                if k < len(complexities):
                    neighborhood_config['complexity'] = complexities[k]

            # Create MessagePassing layer with neighborhood-specific aggregation
            layer = MessagePassing(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                edge_attr_dim=edge_attr_dim,
                aggr=neighborhood_aggr,
                aggregation_config=neighborhood_config
            )

            self.neighborhood_layers.append(layer)

    def _init_combination_function(self):
        """
        Initialize inter-neighborhood combination function ⋀ from Proposition 3.

        MATHEMATICAL ROLE:
        =================
        The combination function ⋀ merges information from different neighborhood types:

        ⋀_{k=1}^n (⊕_{y∈N_k(x)} α_k(h_x, ρ_k_{y→x}(h_y)))

        This is distinct from the intra-neighborhood aggregation ⊕_k and enables
        cross-scale information integration across different topological relations.

        COMBINATION STRATEGIES:
        ======================

        1. CONCATENATION ('concat'):
           - Preserves all neighborhood-specific information
           - Output: [⊕_1, ⊕_2, ..., ⊕_n]
           - Use when: Different neighborhoods provide complementary information
           - Dimension: n × out_dim → out_dim (via learned projection)

        2. SUMMATION ('sum'):
           - Equal weighting across neighborhoods
           - Output: ⊕_1 + ⊕_2 + ... + ⊕_n
           - Use when: Neighborhoods provide similar types of information
           - Dimension: out_dim → out_dim (preserved)

        3. AVERAGING ('mean'):
           - Normalized combination across neighborhoods
           - Output: (⊕_1 + ⊕_2 + ... + ⊕_n) / n
           - Use when: Want to avoid neighborhood size bias
           - Dimension: out_dim → out_dim (preserved)

        DESIGN CHOICE IMPLICATIONS:
        ==========================
        - Concat: Maximum expressiveness, higher parameter count
        - Sum/Mean: Parameter efficiency, assumes similar neighborhood semantics
        - Future extensions: Learned attention weighting across neighborhoods
        """
        if self.combine_method == 'concat':
            # Concatenation requires projection from expanded dimension
            combine_input_dim = self.num_neighborhoods * self.out_dim
        elif self.combine_method in ['sum', 'mean']:
            # Sum/mean preserve dimension
            combine_input_dim = self.out_dim
        else:
            raise ValueError(f"Unknown combine_method '{self.combine_method}'. "
                             f"Supported: 'concat', 'sum', 'mean'")

        # Learnable combination function
        self.combine_fn = nn.Sequential(
            nn.Linear(combine_input_dim, self.out_dim),
            nn.ReLU()  # Nonlinearity for expressive combinations
        )

    def forward(
            self,
            x: Tensor,
            edge_index_list: List[Tensor],
            edge_attr_list: Optional[List[Optional[Tensor]]] = None
    ) -> Tensor:
        """
        Forward pass implementing higher-order copresheaf message passing (Proposition 3).

        COMPLETE EXECUTION FLOW:
        =======================

        Step 1: PARALLEL NEIGHBORHOOD PROCESSING
        For each neighborhood type k ∈ {1, 2, ..., n}:
        - Extract connectivity: edge_index_list[k]
        - Extract attributes: edge_attr_list[k]
        - Apply MessagePassing layer k with aggregation ⊕_k
        - Result: ⊕_{y∈N_k(x)} α_k(h_x, ρ_k_{y→x}(h_y))

        Step 2: INTER-NEIGHBORHOOD COMBINATION
        - Collect outputs from all neighborhoods
        - Apply combination strategy (concat/sum/mean)
        - Apply learnable combination function ⋀
        - Result: ⋀_{k=1}^n (⊕_{y∈N_k(x)} ...)

        Step 3: FINAL FEATURE UPDATE
        - Combine original features with multi-neighborhood result
        - Apply final update function β
        - Result: h_x^{l+1} = β(h_x^l, ⋀(...))

        TOPOLOGICAL INTERPRETATION:
        ==========================
        Each step respects the mathematical structure:

        - LOCALITY: Each ⊕_k operates within a single neighborhood type
        - SCALE SEPARATION: Different k capture different topological scales
        - INTEGRATION: ⋀ enables cross-scale information transfer
        - ADAPTATION: β allows learned integration with previous features

        COMPUTATIONAL BENEFITS:
        ======================
        - PARALLELISM: Neighborhoods processed independently
        - MODULARITY: Each neighborhood can use optimal aggregation
        - FLEXIBILITY: Easy to add/remove neighborhood types
        - EFFICIENCY: Only relevant neighborhoods need computation

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index_list: List of edge indices [2, num_edges_k] per neighborhood
            edge_attr_list: List of edge attributes [num_edges_k, edge_attr_dim] per neighborhood

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Input validation for higher-order message passing
        if edge_attr_list is None:
            edge_attr_list = [None] * len(edge_index_list)

        if len(edge_index_list) != self.num_neighborhoods:
            raise ValueError(f"Expected {self.num_neighborhoods} edge_index tensors, "
                             f"got {len(edge_index_list)}")

        if len(edge_attr_list) != self.num_neighborhoods:
            raise ValueError(f"Expected {self.num_neighborhoods} edge_attr tensors, "
                             f"got {len(edge_attr_list)}")

        # Step 1: Process each neighborhood independently with specific aggregation
        neighborhood_outputs = []
        aggregation_info = []  # For monitoring and debugging

        for k, (edge_index, edge_attr) in enumerate(zip(edge_index_list, edge_attr_list)):
            # Validate neighborhood k
            if edge_index.dtype != torch.long:
                raise TypeError(f"edge_index[{k}] must have dtype torch.long")
            if edge_index.size(0) != 2:
                raise ValueError(f"edge_index[{k}] must be [2, num_edges]")

            # Apply neighborhood-specific copresheaf message passing
            # Each layer uses its own aggregation strategy ⊕_k
            layer_output = self.neighborhood_layers[k](x, edge_index, edge_attr)
            neighborhood_outputs.append(layer_output)

            # Collect aggregation information for analysis
            aggr_info = self.neighborhood_layers[k].get_aggregation_info()
            aggr_info['neighborhood_index'] = k
            aggr_info['num_edges'] = edge_index.size(1)
            aggregation_info.append(aggr_info)

        # Step 2: Combine information across neighborhoods using ⋀
        combined_representation = self._combine_neighborhoods(neighborhood_outputs)

        # Step 3: Final update β(h_x^l, ⋀(...))
        final_input = torch.cat([x, combined_representation], dim=-1)
        updated_features = self.final_update(final_input)

        return updated_features

    def _combine_neighborhoods(self, neighborhood_outputs: List[Tensor]) -> Tensor:
        """
        Apply inter-neighborhood combination ⋀ from Proposition 3.

        MATHEMATICAL OPERATION:
        ======================
        Given outputs from each neighborhood:
        {⊕_{y∈N_1(x)} α_1(...), ⊕_{y∈N_2(x)} α_2(...), ..., ⊕_{y∈N_n(x)} α_n(...)}

        Compute: ⋀_{k=1}^n (⊕_{y∈N_k(x)} α_k(...))

        IMPLEMENTATION STRATEGIES:
        =========================

        1. Concatenation Strategy:
           - Preserves all neighborhood-specific information
           - Allows learning of complex inter-neighborhood relationships
           - Higher memory usage but maximum expressiveness

        2. Summation Strategy:
           - Assumes neighborhoods contribute additively
           - Memory efficient but loses neighborhood identity
           - Good when neighborhoods provide similar information types

        3. Averaging Strategy:
           - Normalized version of summation
           - Reduces bias from neighborhoods with different scales
           - Good for stable training dynamics

        Args:
            neighborhood_outputs: List of tensors [num_nodes, out_dim] from each neighborhood

        Returns:
            Combined representation [num_nodes, out_dim]
        """
        if len(neighborhood_outputs) != self.num_neighborhoods:
            raise ValueError(f"Expected {self.num_neighborhoods} neighborhood outputs")

        # Apply specified combination strategy
        if self.combine_method == 'concat':
            # Concatenate: [⊕_1, ⊕_2, ..., ⊕_n]
            combined = torch.cat(neighborhood_outputs, dim=-1)

        elif self.combine_method == 'sum':
            # Element-wise sum: ⊕_1 + ⊕_2 + ... + ⊕_n
            combined = torch.stack(neighborhood_outputs, dim=0).sum(dim=0)

        elif self.combine_method == 'mean':
            # Element-wise mean: (⊕_1 + ⊕_2 + ... + ⊕_n) / n
            combined = torch.stack(neighborhood_outputs, dim=0).mean(dim=0)

        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

        # Apply learnable combination function
        return self.combine_fn(combined)

    def set_neighborhood_aggregation(
            self,
            neighborhood_idx: int,
            new_aggregation: Union[str, Aggregation],
            config: Optional[dict] = None
    ):
        """
        Dynamically update aggregation for a specific neighborhood.

        ADAPTIVE AGGREGATION:
        ====================
        This enables adaptive learning strategies where different neighborhoods
        may benefit from different aggregation methods at different training phases:

        - Early training: Simple aggregation (sum/mean) for stable gradients
        - Mid training: Statistical aggregation (std/var) to capture diversity
        - Late training: Attention aggregation for fine-tuned importance weighting

        MATHEMATICAL MOTIVATION:
        =======================
        Different neighborhood types N_k(x) may have different optimal ⊕_k operators
        based on their structural properties. This function allows experimentation
        and adaptation of these operators during training.

        Args:
            neighborhood_idx: Index of neighborhood to update (0 ≤ idx < num_neighborhoods)
            new_aggregation: New aggregation method or Aggregation instance
            config: Additional configuration for the new aggregation
        """
        if not (0 <= neighborhood_idx < self.num_neighborhoods):
            raise IndexError(f"Neighborhood index {neighborhood_idx} out of range "
                             f"[0, {self.num_neighborhoods})")

        # Update the specific neighborhood's aggregation
        self.neighborhood_layers[neighborhood_idx].set_aggregation(new_aggregation, config)

    def get_aggregation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of aggregation configuration across all neighborhoods.

        MONITORING HIGHER-ORDER AGGREGATION:
        ===================================
        This method provides insight into the complex aggregation hierarchy:

        1. INTRA-NEIGHBORHOOD: What ⊕_k is used for each neighborhood k
        2. INTER-NEIGHBORHOOD: How ⋀ combines across neighborhoods
        3. COMPUTATIONAL: Parameter counts and complexity metrics
        4. STRUCTURAL: Neighborhood sizes and connectivity patterns

        Returns:
            Comprehensive dictionary with aggregation configuration details
        """
        summary = {
            'total_neighborhoods': self.num_neighborhoods,
            'inter_neighborhood_combination': self.combine_method,
            'neighborhoods': [],
            'total_aggregation_parameters': 0,
            'combination_parameters': sum(p.numel() for p in self.combine_fn.parameters()),
            'update_parameters': sum(p.numel() for p in self.final_update.parameters())
        }

        # Collect information from each neighborhood
        for k, layer in enumerate(self.neighborhood_layers):
            neighborhood_info = layer.get_aggregation_info()
            neighborhood_info.update({
                'neighborhood_index': k,
                'neighborhood_type': f'N_{k}',  # Mathematical notation
            })
            summary['neighborhoods'].append(neighborhood_info)
            summary['total_aggregation_parameters'] += neighborhood_info['parameters']

        # Add derived statistics
        summary['avg_parameters_per_neighborhood'] = (
                summary['total_aggregation_parameters'] / self.num_neighborhoods
        )
        summary['total_parameters'] = (
                summary['total_aggregation_parameters'] +
                summary['combination_parameters'] +
                summary['update_parameters']
        )

        return summary

    def get_neighborhood_types_info(self) -> Dict[str, str]:
        """
        Get information about typical neighborhood interpretations.

        TOPOLOGICAL NEIGHBORHOOD TYPES:
        ===============================
        Common interpretations for different neighborhood indices:

        Returns:
            Dictionary mapping neighborhood indices to typical interpretations
        """
        common_types = {
            0: "Adjacency (0-0): Vertex-vertex connections, traditional graph edges",
            1: "Incidence (0-1): Vertex-edge connections, node-to-hyperedge relations",
            2: "Co-boundary (1-2): Edge-face connections, boundary relationships",
            3: "Higher-order (2-3): Face-volume connections, 3D structure",
        }

        return {f"neighborhood_{k}": common_types.get(k, f"Custom neighborhood type {k}")
                for k in range(self.num_neighborhoods)}

import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import Optional
from torch import Tensor
import torch.nn.functional as F

from toposheafx.utils.guards import _same_device, _check_index


class Aggregation(nn.Module):
    """
    Unified aggregation class for copresheaf message passing implementing the ⊕ operator.

    MATHEMATICAL FOUNDATION:
    =======================
    This class implements the permutation-invariant aggregator ⊕ from Proposition 1:

        ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l))

    The aggregator combines multiple messages directed to the same target node x
    into a single aggregated message that preserves important structural information
    while remaining invariant to the order of neighbors.

    AGGREGATION METHODS:
    ===================

    1. BASIC AGGREGATORS (permutation-invariant):
       - 'sum': ⊕ messages = Σ messages (most common, preserves total information)
       - 'mean': ⊕ messages = (1/|N(x)|) Σ messages (normalizes by neighborhood size)
       - 'max': ⊕ messages = max(messages) (captures strongest signal)
       - 'min': ⊕ messages = min(messages) (captures weakest signal)

    2. STATISTICAL AGGREGATORS:
       - 'std': Standard deviation (captures message diversity)
       - 'var': Variance (captures message spread)

    3. LEARNABLE AGGREGATORS:
       - 'attention': Weighted sum with learned attention weights
       - 'lstm': Sequential processing with LSTM (order-dependent)
       - 'set2set': Set-to-set aggregation with attention

    4. MULTI-AGGREGATOR:
       - 'multi': Concatenation of multiple aggregation methods

    COPRESHEAF CONSIDERATIONS:
    =========================
    In the copresheaf framework, aggregation operates within each feature space F(x).
    Unlike traditional GNNs where all nodes share the same space, copresheaves allow
    heterogeneous feature spaces, making the choice of aggregator particularly important
    for preserving the geometric structure of each stalk.

    Args:
        method (str): Aggregation method name
        dim (int, optional): Feature dimension (required for learnable aggregators)
        num_heads (int): Number of attention heads (for attention-based aggregators)
        dropout (float): Dropout probability for learnable components
        **kwargs: Additional method-specific parameters
    """

    # Registry of available aggregation methods
    AGGREGATION_METHODS = {
        'sum', 'mean', 'max', 'min', 'std', 'var',
        'attention', 'lstm', 'set2set', 'multi'
    }

    def __init__(
            self,
            method: str = 'sum',
            dim: Optional[int] = None,
            num_heads: int = 1,
            dropout: float = 0.0,
            **kwargs
    ):
        super().__init__()

        if method not in self.AGGREGATION_METHODS:
            raise ValueError(f"Unknown aggregation method '{method}'. "
                             f"Available: {self.AGGREGATION_METHODS}")

        self.method = method
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.kwargs = kwargs

        # Initialize learnable components for advanced aggregators
        self._init_learnable_components()

    def _init_learnable_components(self):
        """Initialize learnable parameters for advanced aggregation methods."""

        if self.method == 'attention':
            if self.dim is None:
                raise ValueError("dim must be specified for attention aggregation")

            # Multi-head attention for message aggregation
            self.attention = nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )

        elif self.method == 'lstm':
            if self.dim is None:
                raise ValueError("dim must be specified for LSTM aggregation")

            self.lstm = nn.LSTM(
                input_size=self.dim,
                hidden_size=self.dim,
                batch_first=True,
                dropout=self.dropout if self.dim > 1 else 0
            )

        elif self.method == 'set2set':
            if self.dim is None:
                raise ValueError("dim must be specified for set2set aggregation")

            # Set2Set processing steps (default to 3)
            self.processing_steps = self.kwargs.get('processing_steps', 3)
            self.lstm = nn.LSTM(self.dim, self.dim, batch_first=True)
            self.attention_linear = nn.Linear(2 * self.dim, 1)

        elif self.method == 'multi':
            # Multi-aggregator: combine multiple methods
            self.sub_methods = self.kwargs.get('methods', ['sum', 'mean', 'max'])
            self.sub_aggregators = nn.ModuleDict({
                method: Aggregation(method, self.dim, self.num_heads, self.dropout)
                for method in self.sub_methods
                if method != 'multi'  # Prevent recursion
            })

    def forward(
            self,
            messages: Tensor,
            index: Tensor,
            dim_size: Optional[int] = None,
            **kwargs
    ) -> Tensor:
        """
        Apply aggregation to messages directed to the same target nodes.

        MATHEMATICAL OPERATION:
        ======================
        For each target node x, aggregate all incoming messages:

            aggregated_x = ⊕_{y∈N(x)} message_{y→x}

        Where message_{y→x} = α(h_x, ρ_{y→x}(h_y)) from the copresheaf framework.

        INPUT STRUCTURE:
        ===============
        - messages: [num_edges, dim] - one message per directed edge
        - index: [num_edges] - target node index for each message
        - dim_size: number of target nodes (inferred if not provided)

        OUTPUT STRUCTURE:
        ================
        - aggregated: [dim_size, dim] - one aggregated message per target node

        Args:
            messages (Tensor): Messages to aggregate [num_edges, dim]
            index (Tensor): Target node indices [num_edges]
            dim_size (int, optional): Number of target nodes
            **kwargs: Method-specific parameters

        Returns:
            Tensor: Aggregated messages [num_nodes, dim]
        """
        # Validate inputs
        self._validate_inputs(messages, index, dim_size)

        # Determine output size
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        # Route to appropriate aggregation method
        if self.method in {'sum', 'mean', 'max', 'min'}:
            return self._basic_aggregation(messages, index, dim_size)
        elif self.method in {'std', 'var'}:
            return self._statistical_aggregation(messages, index, dim_size)
        elif self.method == 'attention':
            return self._attention_aggregation(messages, index, dim_size, **kwargs)
        elif self.method == 'lstm':
            return self._lstm_aggregation(messages, index, dim_size, **kwargs)
        elif self.method == 'set2set':
            return self._set2set_aggregation(messages, index, dim_size, **kwargs)
        elif self.method == 'multi':
            return self._multi_aggregation(messages, index, dim_size, **kwargs)
        else:
            raise NotImplementedError(f"Aggregation method '{self.method}' not implemented")

    def _validate_inputs(self, messages: Tensor, index: Tensor, dim_size: Optional[int]):
        """Validate input tensors for aggregation."""
        _same_device(messages, index)

        if messages.dim() != 2:
            raise ValueError("messages must be 2D tensor [num_edges, dim]")
        if index.dim() != 1:
            raise ValueError("index must be 1D tensor [num_edges]")
        if messages.size(0) != index.size(0):
            raise ValueError("messages and index must have same first dimension")

        if dim_size is not None:
            _check_index(index, dim_size, "target index")

    def _basic_aggregation(self, messages: Tensor, index: Tensor, dim_size: int) -> Tensor:
        """
        Basic aggregation methods using torch_scatter.

        MATHEMATICAL OPERATIONS:
        =======================
        - sum: ⊕ = Σ (preserves total magnitude)
        - mean: ⊕ = (1/|N(x)|) Σ (normalizes by degree)  
        - max: ⊕ = max (captures strongest signal)
        - min: ⊕ = min (captures weakest signal)
        """
        return scatter(
            messages,
            index,
            dim=0,
            dim_size=dim_size,
            reduce=self.method
        )

    def _statistical_aggregation(self, messages: Tensor, index: Tensor, dim_size: int) -> Tensor:
        """
        Statistical aggregation methods.

        MATHEMATICAL OPERATIONS:
        =======================
        - std: Standard deviation across neighbors (captures diversity)
        - var: Variance across neighbors (captures spread)

        These provide information about the distribution of messages,
        which can be valuable for understanding neighborhood heterogeneity.
        """
        if self.method == 'std':
            # Compute mean first, then standard deviation
            mean_msgs = scatter(messages, index, dim=0, dim_size=dim_size, reduce='mean')
            expanded_mean = mean_msgs[index]
            sq_diff = (messages - expanded_mean) ** 2
            var_msgs = scatter(sq_diff, index, dim=0, dim_size=dim_size, reduce='mean')
            return torch.sqrt(var_msgs + 1e-8)  # Add epsilon for numerical stability

        elif self.method == 'var':
            # Compute variance directly
            mean_msgs = scatter(messages, index, dim=0, dim_size=dim_size, reduce='mean')
            expanded_mean = mean_msgs[index]
            sq_diff = (messages - expanded_mean) ** 2
            return scatter(sq_diff, index, dim=0, dim_size=dim_size, reduce='mean')

    def _attention_aggregation(self, messages: Tensor, index: Tensor, dim_size: int, **kwargs) -> Tensor:
        """
        Attention-based aggregation using multi-head attention.

        MATHEMATICAL OPERATION:
        ======================
        For each target node x with neighbors N(x):

            ⊕ messages = Σ_{y∈N(x)} α_{x,y} * message_{y→x}

        Where α_{x,y} are learned attention weights that focus on important neighbors.

        This is particularly useful in copresheaf settings where different neighbors
        may contribute different amounts of information to the target stalk F(x).
        """
        # Group messages by target node
        output = torch.zeros(dim_size, self.dim, device=messages.device, dtype=messages.dtype)

        for node_idx in range(dim_size):
            mask = (index == node_idx)
            if mask.sum() == 0:
                continue  # No messages for this node

            node_messages = messages[mask]  # [num_neighbors, dim]
            node_messages = node_messages.unsqueeze(0)  # [1, num_neighbors, dim]

            # Self-attention over neighbors
            attended, _ = self.attention(
                node_messages, node_messages, node_messages
            )

            # Aggregate attended messages
            output[node_idx] = attended.squeeze(0).mean(dim=0)  # [dim]

        return output

    def _lstm_aggregation(self, messages: Tensor, index: Tensor, dim_size: int, **kwargs) -> Tensor:
        """
        LSTM-based aggregation processing messages sequentially.

        NOTE: This method is NOT permutation-invariant and should be used carefully
        in copresheaf settings. It's included for cases where order matters (e.g., 
        temporal sequences of topological changes).

        MATHEMATICAL OPERATION:
        ======================
        For each target node x, process messages sequentially:

            h_0, c_0 = 0, 0
            h_t, c_t = LSTM(message_t, h_{t-1}, c_{t-1})
            ⊕ messages = h_final
        """
        output = torch.zeros(dim_size, self.dim, device=messages.device, dtype=messages.dtype)

        for node_idx in range(dim_size):
            mask = (index == node_idx)
            if mask.sum() == 0:
                continue

            node_messages = messages[mask].unsqueeze(0)  # [1, num_neighbors, dim]

            # Process with LSTM
            lstm_out, (h_final, _) = self.lstm(node_messages)
            output[node_idx] = h_final.squeeze(0).squeeze(0)  # [dim]

        return output

    def _set2set_aggregation(self, messages: Tensor, index: Tensor, dim_size: int, **kwargs) -> Tensor:
        """
        Set2Set aggregation with attention mechanism.

        MATHEMATICAL OPERATION:
        ======================
        This implements the Set2Set algorithm which uses an LSTM with attention
        to create a permutation-invariant representation of variable-size sets.

        The algorithm proceeds in multiple processing steps, each refining the
        set representation through attention over the input messages.
        """
        output = torch.zeros(dim_size, 2 * self.dim, device=messages.device, dtype=messages.dtype)

        for node_idx in range(dim_size):
            mask = (index == node_idx)
            if mask.sum() == 0:
                continue

            node_messages = messages[mask]  # [num_neighbors, dim]
            set_repr = self._set2set_process(node_messages)
            output[node_idx] = set_repr

        # Project back to original dimension
        projection = nn.Linear(2 * self.dim, self.dim).to(output.device)
        return projection(output)

    def _set2set_process(self, messages: Tensor) -> Tensor:
        """Process a single set of messages with Set2Set algorithm."""
        batch_size = 1
        num_messages = messages.size(0)

        # Initialize LSTM state
        h = torch.zeros(1, batch_size, self.dim, device=messages.device)
        c = torch.zeros(1, batch_size, self.dim, device=messages.device)

        for _ in range(self.processing_steps):
            # LSTM step
            lstm_input = torch.zeros(batch_size, 1, self.dim, device=messages.device)
            _, (h, c) = self.lstm(lstm_input, (h, c))

            # Attention over messages
            h_expanded = h.squeeze(0).expand(num_messages, -1)  # [num_messages, dim]
            attention_input = torch.cat([messages, h_expanded], dim=1)  # [num_messages, 2*dim]
            attention_scores = self.attention_linear(attention_input)  # [num_messages, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [num_messages, 1]

            # Weighted sum
            weighted_messages = messages * attention_weights  # [num_messages, dim]
            aggregated = weighted_messages.sum(dim=0, keepdim=True)  # [1, dim]

            # Update LSTM input for next step
            lstm_input = aggregated.unsqueeze(1)  # [1, 1, dim]

        return torch.cat([h.squeeze(), aggregated.squeeze()], dim=0)  # [2*dim]

    def _multi_aggregation(self, messages: Tensor, index: Tensor, dim_size: int, **kwargs) -> Tensor:
        """
        Multi-aggregation combining multiple aggregation methods.

        MATHEMATICAL OPERATION:
        ======================
        Applies multiple aggregation methods and concatenates results:

            ⊕ messages = [⊕_1 messages, ⊕_2 messages, ..., ⊕_k messages]

        This provides a rich representation capturing different statistical
        properties of the neighborhood structure.
        """
        aggregated_parts = []

        for method_name, aggregator in self.sub_aggregators.items():
            part = aggregator(messages, index, dim_size, **kwargs)
            aggregated_parts.append(part)

        return torch.cat(aggregated_parts, dim=-1)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'method={self.method}, dim={self.dim}, num_heads={self.num_heads}'


class AggregationFactory:
    """
    Factory class for creating aggregation instances with common configurations.

    This provides convenient constructors for typical copresheaf aggregation patterns
    used in different contexts (e.g., basic GNNs, higher-order networks, attention models).
    """

    @staticmethod
    def create_basic(method: str = 'sum') -> Aggregation:
        """Create basic aggregation (sum, mean, max, min)."""
        return Aggregation(method=method)

    @staticmethod
    def create_statistical(method: str = 'std', dim: int = None) -> Aggregation:
        """Create statistical aggregation (std, var)."""
        return Aggregation(method=method, dim=dim)

    @staticmethod
    def create_attention(dim: int, num_heads: int = 1, dropout: float = 0.0) -> Aggregation:
        """Create attention-based aggregation."""
        return Aggregation(
            method='attention',
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

    @staticmethod
    def create_set2set(dim: int, processing_steps: int = 3) -> Aggregation:
        """Create Set2Set aggregation."""
        return Aggregation(
            method='set2set',
            dim=dim,
            processing_steps=processing_steps
        )

    @staticmethod
    def create_multi(methods: list, dim: int = None) -> Aggregation:
        """Create multi-aggregation combining several methods."""
        return Aggregation(
            method='multi',
            dim=dim,
            methods=methods
        )

    @staticmethod
    def create_for_copresheaf(
            stalk_dim: int,
            neighborhood_complexity: str = 'simple',
            dropout: float = 0.0
    ) -> Aggregation:
        """
        Create aggregation appropriate for copresheaf message passing.

        Args:
            stalk_dim: Dimension of the copresheaf stalk F(x)
            neighborhood_complexity: 'simple', 'complex', or 'heterogeneous'
            dropout: Dropout rate for learnable components

        Returns:
            Configured aggregation instance
        """
        if neighborhood_complexity == 'simple':
            # Basic neighborhoods, use simple aggregation
            return AggregationFactory.create_basic('sum')
        elif neighborhood_complexity == 'complex':
            # Complex neighborhoods, use statistical aggregation
            return AggregationFactory.create_multi(['sum', 'mean', 'std'], stalk_dim)
        elif neighborhood_complexity == 'heterogeneous':
            # Heterogeneous neighborhoods, use attention
            return AggregationFactory.create_attention(stalk_dim, num_heads=4, dropout=dropout)
        else:
            raise ValueError(f"Unknown complexity level: {neighborhood_complexity}")
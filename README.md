# TopoSheafX

**A PyTorch library for copresheaf-based topological deep learning on graphs, simplicial complexes, and combinatorial complexes.**

TopoSheafX implements the mathematical framework introduced in *"Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework"* (Hajij et al.) and extends it with the continuous dynamics of *"Continuous Simplicial Neural Networks"* (COSIMO, Einizade et al.). The library provides a clean, modular API for building neural networks that respect heterogeneous feature geometries, directional information flow, and higher-order topological structure.

---

## Table of Contents

1. [What TopoSheafX Does](#what-toposheafx-does)
2. [Mathematical Background](#mathematical-background)
3. [Installation](#installation)
4. [Package Layout](#package-layout)
5. [Core Components](#core-components)
6. [Models](#models)
7. [Aggregation Strategies](#aggregation-strategies)
8. [Continuous Dynamics (COSIMO)](#continuous-dynamics-cosimo)
9. [Diagnostics](#diagnostics)
10. [Tutorials](#tutorials)
11. [Running the Examples](#running-the-examples)
12. [Testing](#testing)
13. [References](#references)

---

## What TopoSheafX Does

Standard Graph Neural Networks assume every node lives in the same feature space and that information flows symmetrically between neighbors. TopoSheafX lifts both assumptions.

The library lets you build networks where:

- **Every cell has its own feature space (stalk).** Vertices, edges, and triangles can carry features of different dimensions, each with their own geometry.
- **Information transport is directional and learnable.** A copresheaf morphism `ρ_{y→x}: F(y) → F(x)` translates features from a source stalk into the target stalk, so the same neighbor can contribute different information to different receivers.
- **Higher-order relations are first-class.** Message passing is defined over arbitrary neighborhood functions on combinatorial complexes, so edge-to-triangle and triangle-to-vertex flows are as natural as vertex-to-vertex flows.
- **Dynamics can be continuous.** Instead of stacking discrete polynomial filters, layers can apply the matrix-exponential filter `exp(-t·L_k)` for the Hodge Laplacian `L_k`, with learnable time parameters `t`.

Practically, TopoSheafX provides:

- Drop-in models (`CMPNN`, `CHOMPNN`, `NodeCMPNN`) for graph and node classification, compatible with PyTorch Geometric data loaders.
- Multi-dimensional models (`SimplicialComplexCopresheafNetwork`, `CrossDimensionalSimplicialCopresheafNetwork`) for simplicial-complex workloads built on TopoNetX.
- Continuous-dynamics models (`ContinuousCopresheafSimplicialModel`, `AdaptiveContinuousSimplicialModel`) implementing COSIMO Equation 10 with copresheaf morphisms.
- A unified `Aggregation` class covering ten permutation-invariant and learnable aggregators.
- Diagnostic tools for over-smoothing analysis (Dirichlet energy, COSIMO Proposition 5.5 condition checks).
- Four worked tutorials covering graph classification, simplicial learning, cross-dimensional message passing, and continuous dynamics.

---

## Mathematical Background

### Copresheaf Message Passing (Proposition 1)

The fundamental update rule implemented in TopoSheafX:

```
h_x^{l+1} = β( h_x^l , ⊕_{(y→x) ∈ E} α( h_x^l , ρ_{y→x}(h_y^l) ) )
```

Each symbol maps to a concrete neural component:

| Symbol | Role | Implementation |
|---|---|---|
| `ρ_{y→x}` | Copresheaf morphism — transport map between stalks | `copresheaf_map` (MLP) |
| `α` | Message function — combines target state with transported source | `message_fn` (MLP) |
| `⊕` | Permutation-invariant aggregator | `Aggregation` module |
| `β` | Update function — merges aggregated message with current state | `update_fn` (MLP) |

### Higher-Order Copresheaf Message Passing (Proposition 3)

For multiple neighborhood functions `{N_k}`:

```
h_x^{l+1} = β( h_x^l , ⋀_{k=1}^{n} ⊕_{y ∈ N_k(x)} α_k( h_x^l , ρ^k_{y→x}(h_y^l) ) )
```

Each neighborhood `N_k` has its own morphism `ρ^k`, its own message function `α_k`, and its own aggregator. Their results are then combined by an inter-neighborhood operator `⋀` (implemented as concatenation or summation).

### Continuous Dynamics (COSIMO Equation 10)

Continuous layers replace discrete filters with the heat-kernel-style operator `exp(-t·L)`:

```
X^l_k = σ( exp(-t_d L_{k,d}) X^{l-1}_{k,d} Θ^l_{k,d}
         + exp(-t_u L_{k,u}) X^{l-1}_{k,u} Θ^l_{k,u}
         + exp(-t_d L_{k,d}) X^{l-1}_k   Ψ^l_{k,d}
         + exp(-t_u L_{k,u}) X^{l-1}_k   Ψ^l_{k,u} )
```

where `L_{k,d} = B_k^T B_k` is the lower Hodge Laplacian, `L_{k,u} = B_{k+1} B_{k+1}^T` is the upper Hodge Laplacian, and `t_d`, `t_u` are learnable time parameters. `continuous_simplicial_conv` in `conv.py` evaluates this either via a full matrix exponential or via a top-K eigenvalue approximation.

---

## Installation

TopoSheafX targets Python 3.10+ and depends on PyTorch and PyTorch Geometric.

```bash
# Clone the repository
git clone <your-repo-url>
cd toposheafx

# Option A: editable install via pip
pip install -e .

# Option B: reproduce the tested environment with conda
conda env create -f config/environment.yml
conda activate toposheafx
```

Core dependencies: `torch`, `torch-geometric`, `torch-scatter`, `toponetx`, `numpy`, `scipy`. Tutorials also use `networkx`, `matplotlib`, and `jupyter`.

---

## Package Layout

```
toposheafx/
├── base/
│   ├── aggregation.py       # Unified Aggregation class + AggregationFactory
│   ├── message_passing.py   # MessagePassing, HigherOrderMessagePassing, CrossDimensional
│   └── conv.py              # Conv, HigherOrderConv, ContinuousConv, continuous_simplicial_conv
├── nn/
│   ├── layer.py             # CMPNNLayer, CHOMPNNLayer, ContinuousCopresheafLayer, …
│   └── model.py             # CMPNN, CHOMPNN, NodeCMPNN, SimplicialComplexCopresheafNetwork, …
├── data/
│   └── simplicial.py        # SimplicialFeatures container + neighborhood extractors
└── utils/
    ├── diagnostics.py       # Over-smoothing checks, Dirichlet energy, weight norms
    ├── guards.py            # Input validation (indices, device matching)
    ├── mlp.py               # Reusable MLP factory
    └── debug.py             # Developer-facing debug prints

tutorials/
├── graph_classification/    # MUTAG CMPNN/CHOMPNN experiments
└── simplicial/              # Simplicial, cross-dimensional, continuous tutorials

tests/                       # Pytest suite covering every module
train/                       # Generic training loop utilities
data/                        # MUTAG dataset + graph_datasets.py loader
```

---

## Core Components

### `MessagePassing` (`toposheafx.base.message_passing`)

The foundation class for copresheaf message passing over a single neighborhood. It subclasses PyTorch Geometric's `MessagePassing` but disables PyG's built-in aggregation and routes through TopoSheafX's own `Aggregation` module. The constructor takes factory callables for `ρ`, `α`, and `β`, so you can swap in custom transport maps, message functions, and update functions without subclassing.

```python
from toposheafx.base.message_passing import MessagePassing

layer = MessagePassing(
    in_dim=32, out_dim=64,
    aggr="attention",                      # aggregator name or Aggregation instance
    aggregation_config={"num_heads": 4},
)
h_out = layer(x, edge_index, edge_attr=edge_attr)
```

### `HigherOrderMessagePassing`

Stacks several `MessagePassing` instances, one per neighborhood function, and combines their outputs by concatenation or summation. This is the class backing Proposition 3.

### `CrossDimensionalMessagePassing`

Specialized for edges whose endpoints live in *different* dimensions (e.g. a vertex-to-edge edge). It accepts separate source and target feature dimensions, which the standard class does not.

### `Conv` and `HigherOrderConv` (`toposheafx.base.conv`)

Convenience wrappers that preconfigure the standard copresheaf `ρ`, `α`, `β` factories (Linear→Tanh for `ρ`, Linear→ReLU for `α`, two-layer MLP for `β`). Use these when you want the canonical architecture from the paper.

### `ContinuousConv`

A continuous counterpart to `Conv` that applies `exp(-t·L)` via `continuous_simplicial_conv`. Used inside `AdaptiveContinuousSimplicialModel`.

### `SimplicialFeatures` (`toposheafx.data.simplicial`)

A light container mapping simplex dimension → feature tensor, e.g. `{0: vertex_features, 1: edge_features, 2: triangle_features}`. Provides `from_dict`, indexing by dimension, and iteration helpers so layers can remain dimension-agnostic.

---

## Models

All models live in `toposheafx.nn.model`. A quick map from problem type to model:

| Problem | Recommended model |
|---|---|
| Graph classification / regression | `CMPNN`, `CHOMPNN`, or `CopresheafGNN` (config-driven) |
| Node classification on graphs | `NodeCMPNN` |
| Node classification on simplicial complexes | `SimplicialComplexCopresheafNetwork` |
| Cross-dimensional simplicial tasks | `CrossDimensionalSimplicialCopresheafNetwork` |
| Continuous dynamics (COSIMO) | `ContinuousCopresheafSimplicialModel` or `AdaptiveContinuousSimplicialModel` |

### CMPNN

Copresheaf Message Passing Neural Network — multi-layer stack of `CMPNNLayer` implementing Proposition 1. Appropriate when a single neighborhood function is enough (e.g. standard graphs).

```python
from toposheafx.nn.model import CMPNN

model = CMPNN(in_dim=7, hidden_dim=64, out_dim=2,
              num_layers=3, edge_attr_dim=4, num_neighborhoods=1)

logits = model(x, edge_index_list=[edge_index], batch=batch,
               edge_attr_list=[edge_attr])
```

Architecture: `Linear encoder → L × CMPNNLayer (with residuals) → mean pool → MLP readout`.

### CHOMPNN

Copresheaf Higher-Order Message Passing Neural Network — stack of `CHOMPNNLayer` implementing Proposition 3. Use when your data carries multiple neighborhood relations (adjacency + incidence + coboundary, for example).

```python
from toposheafx.nn.model import CHOMPNN

model = CHOMPNN(in_dim=7, hidden_dim=64, out_dim=2,
                num_layers=3, num_neighborhoods=3)

# edge_index_list[k] carries the k-th neighborhood's edges
logits = model(x, edge_index_list=[edges_adj, edges_inc, edges_cob],
               batch=batch)
```

Architecture: `encoder with LayerNorm → L × CHOMPNNLayer with residual+LayerNorm → mean pool → deeper MLP readout`.

### NodeCMPNN

Node-level variant of CMPNN (no graph pooling). Returns `[num_nodes, out_dim]`.

### CopresheafGNN

Config-driven wrapper. Pass a dict like `{"model_type": "cmpnn", "in_dim": 7, "hidden_dim": 64, "out_dim": 2, ...}` and it constructs the appropriate underlying model. Useful for sweeps.

### SimplicialComplexCopresheafNetwork

Multi-neighborhood network for simplicial complexes with an *adaptive* aggregation schedule: mean aggregation at the input layer (normalizes by degree), sum in the middle (accumulates information), and attention at the top (learns neighbor importance). Pass `aggregation_strategy='adaptive'` or override with any single aggregator name.

```python
from toposheafx.nn.model import SimplicialComplexCopresheafNetwork

model = SimplicialComplexCopresheafNetwork(
    in_dim=16, hidden_dim=64, out_dim=5,
    num_layers=3, aggregation_strategy='adaptive'
)
out = model(x, edge_index_list, batch=batch)
```

### CrossDimensionalSimplicialCopresheafNetwork

Network operating on *separate* feature spaces per dimension with true cross-dimensional message passing. Neighborhoods include `0→0`, `0→1`, `1→0`, `1→1`, `1→2`, `2→1`, `2→0`.

```python
from toposheafx.nn.model import CrossDimensionalSimplicialCopresheafNetwork
from toposheafx.data.simplicial import SimplicialFeatures

model = CrossDimensionalSimplicialCopresheafNetwork(
    initial_dims={0: 16, 1: 8, 2: 4},
    hidden_dims={0: 64, 1: 32, 2: 16},
    out_dim=5, num_layers=3
)

features = SimplicialFeatures.from_dict({0: x0, 1: x1, 2: x2})
logits = model(features, neighborhoods)   # readout from vertices
```

### ContinuousCopresheafSimplicialModel

Full multi-dimensional continuous model combining copresheaf morphisms with COSIMO dynamics. Processes vertices, edges, and triangles simultaneously, uses boundary operators for cross-dimensional projection, and has learnable time parameters per dimension.

```python
from toposheafx.nn.model import ContinuousCopresheafSimplicialModel

model = ContinuousCopresheafSimplicialModel(
    feature_dims={0: 70, 1: 30, 2: 41},
    hidden_dims={0: 64, 1: 32, 2: 16},
    out_dim=10,
    target_dim=0,           # produce predictions at vertex level
    num_layers=3,
    t_d_init=0.1, t_u_init=0.1,
    learnable_t=True,
    K=None                  # use full matrix exponential; set an int for top-K approx.
)
preds = model(features_dict, boundary_operators)
```

### AdaptiveContinuousSimplicialModel

Lighter continuous model with one learnable time parameter *per neighborhood*. Each layer applies `ContinuousConv` separately across neighborhoods and concatenates/projects the results, enabling multi-resolution diffusion.

---

## Aggregation Strategies

The `Aggregation` class in `toposheafx.base.aggregation` implements the `⊕` operator from Proposition 1 and supports ten methods. Every method can be selected by name via the `aggr=` argument on any layer or model, or instantiated explicitly and passed in.

| Name | Category | Description |
|---|---|---|
| `sum` | Basic | `Σ messages` — preserves total information. |
| `mean` | Basic | Normalizes by neighborhood size. |
| `max` | Basic | Captures the strongest signal per feature dim. |
| `min` | Basic | Captures the weakest signal per feature dim. |
| `std` | Statistical | Standard deviation — measures message diversity. |
| `var` | Statistical | Variance — measures message spread. |
| `attention` | Learnable | Multi-head self-attention over neighbors (requires `dim`). |
| `lstm` | Learnable | Sequential LSTM aggregation (*not* permutation-invariant — use with caution). |
| `set2set` | Learnable | Vinyals-style Set2Set with attention, permutation-invariant. |
| `multi` | Composite | Concatenates several sub-aggregators (default `['sum','mean','max']`). |

### Direct instantiation

```python
from toposheafx.base.aggregation import Aggregation, AggregationFactory

agg_sum   = Aggregation("sum")
agg_attn  = Aggregation("attention", dim=64, num_heads=4, dropout=0.1)
agg_multi = Aggregation("multi", dim=64, methods=["sum", "mean", "std"])

# Factory shortcuts
agg_s2s   = AggregationFactory.create_set2set(dim=64, processing_steps=3)
agg_auto  = AggregationFactory.create_for_copresheaf(
    stalk_dim=64, neighborhood_complexity="heterogeneous", dropout=0.1
)
```

### Plugging into a layer

```python
from toposheafx.nn.layer import CMPNNLayer

layer = CMPNNLayer(in_dim=32, out_dim=64, num_neighborhoods=1,
                   aggregation="attention")
# or
layer = CMPNNLayer(in_dim=32, out_dim=64, num_neighborhoods=1,
                   aggregation=Aggregation("multi", dim=64,
                                           methods=["sum","mean","max"]))
```

### Adaptive aggregation

`aggr="copresheaf_adaptive"` selects a sensible default based on `complexity ∈ {simple, complex, heterogeneous}` passed via `aggregation_config`. `simple` → `sum`, `complex` → multi-aggregator `[sum, mean, std]`, `heterogeneous` → multi-head attention.

---

## Continuous Dynamics (COSIMO)

Continuous layers replace discrete polynomial filters with the matrix-exponential filter of the Hodge Laplacian. The key primitive is:

```python
from toposheafx.base.conv import continuous_simplicial_conv

y = continuous_simplicial_conv(
    x_k=features,      # [num_simplices, feature_dim]
    laplacian_k=L,     # [num_simplices, num_simplices]
    t=t_param,         # scalar or nn.Parameter
    K=None,            # None → full matrix_exp; int → top-K eigen approx.
)
```

Two reasons you might want `K` set to an integer: (1) large complexes where `torch.matrix_exp` is expensive, and (2) spectral regularization — limiting to the dominant eigenvectors smooths the filter.

`ContinuousCopresheafLayer` and `ContinuousCHOMPNNLayer` embed this primitive into a full layer with copresheaf morphisms (`ρ_down`, `ρ_up`), COSIMO projections (`Θ_down`, `Θ_up`, `Ψ_down`, `Ψ_up`), and message/update functions. You typically don't call them directly — use `ContinuousCopresheafSimplicialModel` or `AdaptiveContinuousSimplicialModel`.

Hodge Laplacians are computed from boundary operators via `toposheafx.data.simplicial` helpers (`compute_hodge_laplacians`, `compute_lower_laplacians`, `compute_upper_laplacians`). These support symmetric normalization for numerical stability.

---

## Diagnostics

The `toposheafx.utils.diagnostics` module provides tools for over-smoothing analysis.

### COSIMO Proposition 5.5 check

Continuous models are stable only when time parameters satisfy `t < ln(s) / (λ_min(L_k) + κ_f(L_k))`, where `s` is a weight-norm bound and `κ_f` is the condition number of the Laplacian. Use:

```python
from toposheafx.utils.diagnostics import (
    check_oversmoothing_condition,
    extract_dimension_specific_weight_norms,
)

weight_norms = extract_dimension_specific_weight_norms(model)
report = check_oversmoothing_condition(
    laplacians={0: L0, 1: L1, 2: L2},
    time_params={"t_d": 0.1, "t_u": 0.1},
    weight_norms=weight_norms,
)
# report["satisfies_condition"] is False if any dimension is at risk.
```

### Dirichlet energy

`compute_dirichlet_energy(features, laplacian)` returns `tr(X^T L X)`. Tracking it across layers gives a direct measurement of how quickly your network is smoothing the signal. The `continuous_copresheaf_simplicial_neural_network_tutorial.ipynb` walks through this analysis end-to-end.

### Related helpers

- `analyze_smoothing_rate` — fit an exponential to energy-vs-depth and report a decay rate.
- `validate_time_parameters` — sanity-check learned `t_d`, `t_u` (NaN/Inf guards, valid range).

---

## Tutorials

Four Jupyter notebooks in `tutorials/` walk through increasingly sophisticated use cases.

### 1. `tutorials/graph_classification/copresheaf_gnn_experiments.ipynb`

Compares `CMPNN` and `CHOMPNN` against standard GNN baselines (GCN, GAT, GIN) on the MUTAG molecular graph classification dataset. Covers:

1. Introduction to CTNN theory (stalks, morphisms, directional message passing).
2. The mathematical framework (Propositions 1 and 3).
3. Loading MUTAG via the bundled `graph_datasets.py` loader.
4. Defining standard baselines for comparison.
5. Building CMPNN and CHOMPNN with several configurations.
6. Training loops with fixed-seed repeats.
7. Accuracy plots and confusion matrices.
8. Statistical significance testing across seeds.
9. Architecture and parameter-count summaries.

Start here if you want to train a classifier on real graph data.

### 2. `tutorials/simplicial/copresheaf_simplicial_tutorial.ipynb`

Introduces copresheaf message passing on simplicial complexes built with TopoNetX. Covers:

1. Constructing simplicial complexes from NetworkX graphs.
2. Extracting adjacency, incidence, co-incidence, and higher-order neighborhoods.
3. Building a specialized `SimplicialComplexCopresheafLayer`.
4. A complete network for node classification on a simplicial complex.
5. Training on synthetic labels with rich topological context.
6. A side-by-side comparison of every aggregation strategy.
7. Adaptive copresheaf morphisms that change with the local topology.
8. Summary of the framework's advantages.

Start here if your data is a simplicial complex.

### 3. `tutorials/simplicial/cross_dimensional_simplicial_tutorial.ipynb`

Focuses specifically on **cross-dimensional** message passing — information flowing *between* simplex types rather than within the same type. Covers:

1. The distinction between higher-order and cross-dimensional message passing.
2. Building a simplicial complex with vertices, edges, and triangles.
3. Extracting cross-dimensional neighborhoods (0→1, 1→0, 1→2, 2→1, 2→0, …).
4. Initializing multi-dimensional features via `SimplicialFeatures`.
5. Running a single cross-dimensional message-passing step to understand the mechanics.
6. Building a full `MultiDimensionalSimplicialCopresheafLayer`.
7. A complete network with node-classification readout from vertices.
8. Training on a synthetic task where labels depend on topological context.
9. Inspecting learned representations at each dimension.
10. Visualizing the flow of information across dimensions.
11. A direct comparison with a vertex-only GNN.

Start here if cross-dimensional information flow is central to your problem.

### 4. `tutorials/simplicial/continuous_copresheaf_simplicial_neural_network_tutorial.ipynb`

Demonstrates the COSIMO framework with copresheaf morphisms. Covers:

1. Architecture overview — when to pick the adaptive vs. the full multi-dimensional continuous model.
2. A single-dimensional example using vertex-only input and edge-level dynamics.
3. A full multi-dimensional continuous network.
4. Over-smoothing diagnostics: the COSIMO Proposition 5.5 condition check.
5. Dirichlet energy analysis layer-by-layer.
6. Depth-vs-energy ablation for different `t_d`, `t_u`.
7. The trajectory-prediction experiment from Section 6.1 of the COSIMO paper, using a 2D complex built by Delaunay triangulation.

Start here if you want continuous Hodge-Laplacian dynamics or you're debugging over-smoothing.

---

## Running the Examples

Each tutorial notebook inserts the project root into `sys.path` automatically in its first cell, so you can launch them from anywhere:

```bash
# From the project root
jupyter notebook tutorials/graph_classification/copresheaf_gnn_experiments.ipynb
# or
jupyter lab tutorials/simplicial/cross_dimensional_simplicial_tutorial.ipynb
```

A minimal end-to-end CMPNN training script in plain Python:

```python
import torch
from torch_geometric.loader import DataLoader
from data.graph_datasets import load_mutag       # bundled loader
from toposheafx.nn.model import CMPNN

dataset = load_mutag(root="data/MUTAG")
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

model = CMPNN(in_dim=dataset.num_node_features,
              hidden_dim=64,
              out_dim=dataset.num_classes,
              num_layers=3,
              edge_attr_dim=dataset.num_edge_features,
              num_neighborhoods=1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch.x,
                       edge_index_list=[batch.edge_index],
                       batch=batch.batch,
                       edge_attr_list=[batch.edge_attr])
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
```

For simplicial tasks, the training loop is nearly identical but you pass a list of neighborhood `edge_index` tensors (one per neighborhood type) and, for continuous models, a dict of boundary operators.

The `train/` directory contains a generic `Trainer` class (`train/trainer.py`) and a runnable script (`train/train_model.py`) that wraps these patterns with logging and checkpointing — useful if you want a ready-made training loop rather than rolling your own.

---

## Testing

The `tests/` directory contains a pytest suite covering aggregation, message passing, convolutions, layers, full models, the MUTAG loader, and training loops.

```bash
# Run the whole suite
pytest

# Run with coverage
pytest --cov=toposheafx --cov-report=term-missing

# Or use the bundled driver
python run_tests.py
```

Test organization:

- `test_aggregation.py` — every aggregation method, edge cases (empty neighborhoods, single neighbors), shape correctness.
- `test_message_passing.py` — base `MessagePassing` and `HigherOrderMessagePassing`.
- `test_conv.py` — `Conv`, `HigherOrderConv`, `ContinuousConv`, and `continuous_simplicial_conv`.
- `test_copresheaf.py`, `test_copresheaf_ext.py` — integration tests of the full Proposition 1 and 3 updates.
- `test_cmpnn.py`, `test_model.py`, `test_models_coverage.py` — end-to-end model forward/backward tests.
- `test_layers_parametrized.py` — parametrized sweeps over layer configurations.
- `test_mutag_loader.py` — MUTAG data pipeline.
- `test_trainer_smoke.py`, `test_trainer_loops.py` — training-loop smoke and regression tests.

CI runs this suite via GitHub Actions, with Ruff + Black enforcing style.

---

## References

Primary sources implemented in this codebase:

- **Hajij et al.**, *"Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework"* — defines CTNNs, Propositions 1 and 3, the copresheaf morphism `ρ_{y→x}`, combinatorial complexes, and neighborhood functions. Everything in `base/message_passing.py`, `nn/layer.py` (CMPNN/CHOMPNN layers), and `nn/model.py` (CMPNN/CHOMPNN/simplicial networks) is a direct implementation of this paper. A copy is included in the repo as `copresheafNeuralNetworks.pdf`.
- **Einizade et al.**, *"Continuous Simplicial Neural Networks"* (COSIMO) — defines Equation 10 (matrix-exponential filters on Hodge Laplacians), the over-smoothing condition in Proposition 5.5, and multi-branch extensions. Everything in `base/conv.py:continuous_simplicial_conv`, `nn/layer.py:ContinuousCopresheafLayer`, `nn/model.py:ContinuousCopresheafSimplicialModel` / `AdaptiveContinuousSimplicialModel`, and `utils/diagnostics.py` is derived from this paper. A copy is included as `ContinuousSimplicialNeuralNetworks.pdf`.

Related frameworks this library integrates with:

- **PyTorch Geometric** — provides the base `MessagePassing` class, `scatter`, and data containers.
- **TopoNetX** — provides `SimplicialComplex`, `CellComplex`, and `CombinatorialComplex` data structures used throughout the simplicial tutorials.

The bundled MUTAG dataset is the standard molecular graph classification benchmark originally published by Debnath et al.; see `data/MUTAG/README.txt` for full attribution.

---

*If you use TopoSheafX in your work, please cite the two primary papers above along with this library.*

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset


# python scripts/train_model.py --dataset synthetic_complex
class SyntheticComplexDataset(InMemoryDataset):
    """
    A synthetic dataset of combinatorial complexes with multiple neighborhood
    structures and edge attributes, for higher-order message passing models.
    """
    def __init__(self, num_samples=200, transform=None):
        super().__init__('.', transform)
        self.data_list = self._generate_synthetic_complexes(num_samples)
        self.data, self.slices = self.collate(self.data_list)

    def _generate_random_edges(self, num_nodes, prob=0.3):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and torch.rand(1).item() < prob:
                    edge_index.append([i, j])
        if len(edge_index) == 0:
            edge_index.append([0, 1])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def _generate_triangle_edges(self, num_nodes, dense=True):
        edge_index = set()
        if dense:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    for k in range(j + 1, num_nodes):
                        edge_index.update([(i, j), (j, k), (k, i)])
        else:
            for i in range(0, num_nodes - 2, 3):
                edge_index.update([(i, i + 1), (i + 1, i + 2), (i + 2, i)])
        if len(edge_index) == 0:
            edge_index.add((0, 1))
        edge_index = list(edge_index)
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def _generate_random_edge_attrs(self, edge_index):
        num_edges = edge_index.size(1)
        return torch.rand(num_edges, 2)  # 2D edge features

    def _generate_triangle_edge_attrs(self, edge_index):
        num_edges = edge_index.size(1)
        return torch.ones(num_edges, 2) * 0.5  # constant edge features

    def _generate_synthetic_complexes(self, num_samples):
        data_list = []
        for i in range(num_samples):
            num_nodes = 10
            x = torch.randn(num_nodes, 4)  # node features

            label = torch.tensor([0 if i < num_samples // 2 else 1], dtype=torch.long)

            # Neighborhood 0: random graph edges
            edge_index_0 = self._generate_random_edges(num_nodes, prob=0.3)
            edge_attr_0 = self._generate_random_edge_attrs(edge_index_0)

            # Neighborhood 1: triangle edges (dense or sparse)
            edge_index_1 = self._generate_triangle_edges(num_nodes, dense=(label.item() == 1))
            edge_attr_1 = self._generate_triangle_edge_attrs(edge_index_1)

            edge_index_list = [edge_index_0, edge_index_1]
            edge_attr_list = [edge_attr_0, edge_attr_1]
            batch = torch.zeros(num_nodes, dtype=torch.long)

            data = Data(
                x=x,
                edge_index_list=edge_index_list,
                edge_attr_list=edge_attr_list,
                batch=batch,
                y=label
            )
            data_list.append(data)
        return data_list




# python scripts/train_model.py --dataset mutag
class MUTAGCopresheaf(InMemoryDataset):
    """
    Prepares the MUTAG dataset for use with the Copresheaf-based MPNN framework.

    Converts each graph in the MUTAG dataset into a format compatible with CMPNN:
    - `edge_index_list`: a list of edge_index tensors, supporting multiple neighborhoods.
    - `edge_attr_list`: a list of edge attribute tensors (or None) corresponding to each edge set.
    - `batch`: a tensor assigning each node to a single graph (needed for pooling).
    - `y`: the graph-level class label.

    The implementation currently uses a single neighborhood structure (1-skeleton)
    based on the molecular bond graph.

    Args:
        root (str): Root directory to store the processed dataset.
        transform (callable, optional): Data transformation.
        pre_transform (callable, optional): Pre-processing transformation.
    """
    def __init__(self, root='./data', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.dataset = TUDataset(root=root, name='MUTAG')
        self.data, self.slices = self.collate([self._convert_graph(g) for g in self.dataset])

    def _convert_graph(self, data):
        """
        Converts a PyG graph into a Data object with CMPNN-compatible fields.

        Args:
            data (Data): A PyTorch Geometric Data object from the MUTAG dataset.

        Returns:
            Data: Augmented Data object with:
                - x (Tensor): Node features.
                - edge_index_list (List[Tensor]): List of edge indices.
                - edge_attr_list (List[Tensor or None]): List of edge attribute tensors.
                - batch (Tensor): Batch vector of zeros (since graphs are individual samples).
                - y (Tensor): Graph label.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        y = data.y

        edge_index_list = [edge_index]  # One neighborhood: molecular bonds
        edge_attr_list = [edge_attr if edge_attr is not None else None]
        batch = torch.zeros(x.size(0), dtype=torch.long)  # All nodes belong to one graph

        return Data(
            x=x,
            edge_index_list=edge_index_list,
            edge_attr_list=edge_attr_list,
            batch=batch,
            y=y
        )

def load_complex_dataset(name):
    """
    Return a dataset for a given name.

    Args:
        name (str): Dataset identifier string.

    Returns:
        Dataset object.
    """
    if name == "synthetic_complex":
        return SyntheticComplexDataset()
    elif name == "mutag":
        return MUTAGCopresheaf()
    else:
        raise ValueError(f"Unknown dataset: {name}")

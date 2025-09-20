# test_mutag_loader.py

from data.graph_datasets import load_complex_dataset


def test_mutag_dataset():
    dataset = load_complex_dataset("mutag")
    print(f"Loaded MUTAG dataset with {len(dataset)} graphs.")
    sample = dataset[0]

    print("Sample Graph:")
    print(f" - Node features shape: {sample.x.shape}")
    print(f" - Edge index list count: {len(sample.edge_index_list)}")
    print(f" - Edge index[0] shape: {sample.edge_index_list[0].shape}")
    print(f" - Edge attr[0] shape: {sample.edge_attr_list[0].shape if sample.edge_attr_list[0] is not None else 'None'}")
    print(f" - Label: {sample.y.item()}")
    print(f" - Batch: {sample.batch}")

if __name__ == "__main__":
    test_mutag_dataset()

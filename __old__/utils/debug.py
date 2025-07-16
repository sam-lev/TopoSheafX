def print_copresheaf_map(ρ, label="ρ"):
    print(f"{label} =")
    print(ρ)

def dump_neighborhood_matrix(matrix, row_labels=None, col_labels=None):
    print("Neighborhood Matrix:")
    if row_labels and col_labels:
        print("    ", "  ".join(col_labels))
        for i, row in enumerate(matrix):
            print(f"{row_labels[i]}: ", "  ".join(map(str, row)))
    else:
        for row in matrix:
            print("  ".join(map(str, row)))

def inspect_cell_features(features_dict):
    for cell, feat in features_dict.items():
        print(f"Cell {cell}: {feat}")

def debug_print(*args):
    """
    Print each argument on a new line in the order given.
    Accepts strings, floats, ints, lists, and tuples.
    """
    for arg in args:
        print(arg)
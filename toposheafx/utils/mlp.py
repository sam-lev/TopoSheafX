import torch.nn as nn
def mlp_factory(dim_in: int, dim_out: int, mlp_type='lin_lrelu_lin'):
    if mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )

    elif mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )
    else:
        raise ValueError("Unknown gin_mlp_type!")
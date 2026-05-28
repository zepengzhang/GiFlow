import torch
import tsl.datasets as datasets
from tsl.data import ImputationDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.ops.imputation import add_missing_values
import pandas as pd
import numpy as np
from tsl.datasets.prototypes import TabularDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from torch_geometric.utils import dense_to_sparse

def dataset_loading(dataset_name: str, 
                    missing_rate: float, 
                    missing_type: str, 
                    window: int,
                    stride: int,
                    adj_threshold: float,
                    val_len: float,
                    test_len: float,
                    seed: int, 
                    batch_size):
    
    if dataset_name == 'airquality_small':
        DatasetClass = getattr(datasets, 'AirQuality', None)
        dataset = DatasetClass(root=f'./data/{dataset_name}',small=True)
    else:
        DatasetClass = getattr(datasets, dataset_name, None)
        dataset = DatasetClass(root=f'./data/{dataset_name}')
    
    # dataset.eval_mask = dataset.training_mask
    if missing_type == 'point':
        dataset = add_missing_values(dataset, 
                                    p_fault=0, 
                                    p_noise=missing_rate,
                                    seed=seed)
        
    # time embedding, this consider the periodicity of day/week
    time_emb = dataset.datetime_encoded(['day', 'week']).values
    time_dim = time_emb.shape[-1]
    covariates = {'temporal_encoding': time_emb}

    input_map = {'u': 'temporal_encoding', 'x': 'target'}
    # dataset.similarity_options = ['distance', 'correlation']
    adj = dataset.get_connectivity(method=None,
                                    threshold=adj_threshold,
                                   include_self=False,
                                   force_symmetric=True)
    # instantiate dataset
    torch_dataset = ImputationDataset(target=dataset.numpy(return_idx=False),
                                      mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=adj,
                                      input_map=input_map,
                                      window=window,
                                      stride=stride,
                                      covariates=covariates)

    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)
    scalers = {'target': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size)

    dm.setup()

    edge_index = torch_dataset[0].input.edge_index
    edge_weight = torch_dataset[0].input.edge_weight

    return dm, edge_index, edge_weight, dataset.n_nodes, time_dim

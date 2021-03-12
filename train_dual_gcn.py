import os
import collections
import argparse
from parse_config import ConfigParser
import torch
import numpy as np
import networkx as nx

from multiscale_interactome.msi.dli import DLI
from method.dataset import DiffusionDataLoader, DiffusionDataSet
from helpers.helper import preprocess_graph, convert_sparse_matrix_to_sparse_tensor


def gen_graph_data(dli, dim):
    adj = nx.adjacency_matrix(dli.graph)  # (N, N) sparse matrix
    N = adj.shape[0]
    features = np.eyes((N, dim), dtype=np.int)
    return features, adj


def main(args):
    # load data
    dli = DLI()
    dli.load()
    # form networkx graph

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    all_features, all_adj = gen_graph_data()
    all_adj_tensor = convert_sparse_matrix_to_sparse_tensor(all_adj_normed)

    # the model
    model = ResidualGraphConvolutionalNetwork(train_batch_size=args.batch_size if args.batch_size > 0 else x_adj_normed.shape[0],
                                              val_batch_size=all_adj_normed.shape[0],
                                              num_layers=args.num_layers,
                                              hidden_units=args.hidden_units,
                                              init_weights=args.init_weights,
                                              layer_decay=args.layer_decay)

    if args.gpu_id is not None:
        model.cuda()
        all_adj_tensor = all_adj_tensor.cuda()
        print('using gpu')

    training_dataset = DiffusionDataSet(features=x_features,
                                        adj=x_adj_normed_sparse_tensor)
    training_loader = DiffusionDataLoader(training_dataset,
                                          batch_size=args.batch_size if args.batch_size > 0 else len(
                                              training_dataset),
                                          num_workers=6,
                                          shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=0)

    trainer.train()
    # save the embedding
    trainer.save_emb()
    # test

    return dli


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    args = parser.parse_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    return main(args)


if __name__ == '__main__':
    run()

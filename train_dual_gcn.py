import os
import collections
import argparse
from parse_config import ConfigParser
import torch
import numpy as np
import networkx as nx

from multiscale_interactome.msi.dli import DLI


def main(config):
    # load data
    dli = DLI()
    dli.load()
    # form networkx graph
    adj = nx.adjacency_matrix(dli.g)
    # data generator
    # train and valid on the fly
    # save the embedding
    # test
    return


def run():
    args = argparse.ArgumentParser(description='Drug Repurposing Dual GCN')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-s', '--save-dir', default=None, type=str,
                      help='path to save and load (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float,
                   target=('optimizer', 'args', 'lr'))
    ]
    config = ConfigParser(args, options)
    return main(config)


if __name__ == '__main__':
    run()

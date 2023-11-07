"""Utility functions for Input/Output."""
import argparse
import os
import torch
from torch.optim import Adam


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
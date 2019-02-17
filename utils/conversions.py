import torch

from .characters import Characters
from .stages import Stages


def char_to_names(indices):
    chars = [Characters.toNamePartial[idx.item()] for idx in indices]
    return chars

def stage_to_names(indices):
    stages = [Stages.toNamePartial[idx.item()] for idx in indices]
    return stages

def char_to_names_tensor(output_tensor):
    max_indices = output_tensor.max(1)[1]
    chars = [Characters.toNamePartial[idx.item()] for idx in max_indices]
    return chars

def stage_to_names_tensor(output_tensor):
    max_indices = output_tensor.max(1)[1]
    stages = [Stages.toNamePartial[idx.item()] for idx in max_indices]
    return stages

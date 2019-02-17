import torch

from .characters import Characters
from .stages import Stages

def char_to_names(output_tensor):
    max_indices = output_tensor.max(1)[1]
    chars = [Characters.toNamePartial[idx.item()] for idx in max_indices]
    return chars

def stage_to_names(output_tensor):
    max_indices = output_tensor.max(1)[1]
    stages = [Stages.toNamePartial[idx.item()] for idx in max_indices]
    return stages

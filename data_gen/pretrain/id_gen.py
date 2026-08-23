# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math_gen.problem_gen import Problem
from data_gen.prototype.id_gen import IdGen_PT

class IdGen(IdGen_PT):
    def __init__(self, max_op=10, max_edge=15, op=None, perm_level: str = None, detail_level: str = None, be_shortest: bool=True, reduce_mod: bool=True) -> None:
        super().__init__('light', 'light', max_op, max_edge, op, perm_level, detail_level, be_shortest, reduce_mod)
    
    def gen_prob(self, ava_hash, p_format: str, problem: Problem=None):
        super().gen_prob(ava_hash, p_format, problem=problem)


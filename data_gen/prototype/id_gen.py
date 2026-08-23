# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import random
from const import params
from const.params import dot
from math_gen.problem_gen import Problem
from typing import Optional
from tools.tools import choose_from_softmax, tokenizer


@contextlib.contextmanager
def mod_reduction(enabled: bool):
    '''Set params.reduce_mod for the duration of one problem generation, then restore it.

    Scoped to a single gen_prob() call rather than set once at import, because one
    process generates several kinds of problem in sequence and the flag must not leak
    past the caller that asked for it.
    '''
    previous = params.reduce_mod
    params.reduce_mod = enabled
    try:
        yield
    finally:
        params.reduce_mod = previous


class IdGen_PT(object):
    def __init__(self, style: str, op_style: str, max_op=10, max_edge=15, op=None, perm_level: str=None, detail_level: str=None, be_shortest: bool=True, reduce_mod: bool=True) -> None:
        self.reduce_mod = reduce_mod
        self.style = style
        self.op_style = op_style
        self.max_op = max_op
        self.max_edge = max_edge
        self.op = op
        self.perm_level = int(perm_level) if isinstance(perm_level, str) else perm_level
        if isinstance(detail_level, str):
            case0 = int(detail_level[0])
            case1 = int(detail_level[1])
            case2 = int(detail_level[2])
            detail_level = case0 + 3 * case1 + 6 * case2
        self.detail_level = detail_level

        self.be_shortest = be_shortest

        self.op_ = self.gen_sol_op(op_style)
        self.perm_level_ = random.randint(0, 6) if self.perm_level == None else self.perm_level
        self.detail_level_ = random.randint(0, 11) if self.detail_level == None else self.detail_level

    def gen_param(self):
        gen_param = getattr(self, f"gen_param_{self.style}")
        gen_param()

    def gen_param_heavy(self):
        self.p = random.random()
        '''n = random.choice(range(1, max_op))
        m = random.choice(range(n, max_op))
        s = random.choice(range(m, max_op))'''
        if self.op == None:
            self.n = random.choice(range(1, self.max_op+1))
            self.m = random.choice(range(self.n, self.max_op+1))
            self.s = random.choice(range(self.m, self.max_op+1))
        else:
            self.s = self.op
            self.n = random.choice(range(1, self.s+1))
            self.m = random.choice(range(self.n, self.s+1))
        
        relative_dist = (self.s - 1) / (self.max_edge - 1) # from 0 to 1
        temperature = 1.
        pos = [0.2, 0.5, 0.8]
        weight = [-(relative_dist - i)**2 / temperature for i in pos]
        
        self.d = choose_from_softmax([2, 3, 4], weight=weight)
        t0 = choose_from_softmax([2, 3, 4], weight=weight)
        t1 = choose_from_softmax([2, 3, 4], weight=weight)
        self.w0 = min(t0, t1)
        self.w1 = max(t0, t1)
        
        # n, m, s = 12, 16, 16
        if random.random() < 0.5:
            self.e = random.choice(range(self.max_edge+1))
        else:
            self.e = random.choice(range(self.s, self.max_edge+1))

    def gen_param_uniform(self):
        self.p = random.random()
        '''n = random.choice(range(1, max_op))
        m = random.choice(range(n, max_op))
        s = random.choice(range(m, max_op))'''
        if self.op == None:
            self.s = random.choice(range(1, self.max_op+1))
            self.m = random.choice(range(1, self.s+1))
            self.n = random.choice(range(1, self.m+1))
        else:
            self.s = self.op
            self.m = random.choice(range(1, self.s+1))
            self.n = random.choice(range(1, self.m+1))
        
        relative_dist = (self.s - 1) / (self.max_edge - 1) # from 0 to 1
        temperature = 1.
        pos = [0.2, 0.5, 0.8]
        weight = [-(relative_dist - i)**2 / temperature for i in pos]
        
        self.d = choose_from_softmax([2, 3, 4], weight=weight)
        t0 = choose_from_softmax([2, 3, 4], weight=weight)
        t1 = choose_from_softmax([2, 3, 4], weight=weight)
        self.w0 = min(t0, t1)
        self.w1 = max(t0, t1)
        # n, m, s = 12, 16, 16
        if random.random() < 0.5:
            self.e = random.choice(range(self.max_edge+1))
        else:
            self.e = random.choice(range(self.s, self.max_edge+1))

    def gen_param_middle(self):
        self.p = random.random()
        '''n = random.choice(range(1, max_op))
        m = random.choice(range(n, max_op))
        s = random.choice(range(m, max_op))'''
        if self.op == None:
            t0 = random.choice(range(1, self.max_op+1))
            t1 = random.choice(range(1, self.max_op+1))
            t2 = random.choice(range(1, self.max_op+1))
            self.s = max(t0, t1, t2)
            self.n = min(t0, t1, t2)
            self.m = t0 + t1 + t2 - self.s - self.n
        else:
            self.s = self.op
            t0 = random.choice(range(1, self.s+1))
            t1 = random.choice(range(1, self.s+1))
            self.m = max(t0, t1)
            self.n = min(t0, t1)
        
        relative_dist = (self.s - 1) / (self.max_edge - 1) # from 0 to 1
        temperature = 1.
        pos = [0.2, 0.5, 0.8]
        weight = [-(relative_dist - i)**2 / temperature for i in pos]
        
        self.d = choose_from_softmax([2, 3, 4], weight=weight)
        t0 = choose_from_softmax([2, 3, 4], weight=weight)
        t1 = choose_from_softmax([2, 3, 4], weight=weight)
        self.w0 = min(t0, t1)
        self.w1 = max(t0, t1)

        # n, m, s = 12, 16, 16
        if random.random() < 0.5:
            self.e = random.choice(range(self.max_edge+1))
        else:
            self.e = random.choice(range(self.s, self.max_edge+1))

    def gen_param_light(self):
        if self.op == None:
            t0 = random.choice(range(1, self.max_op+1))
            t1 = random.choice(range(1, self.max_op+1))
            self.s = min(t0, t1)
            t0 = random.randint(1, self.s)
            t1 = random.randint(1, self.s)
            self.n = max(t0, t1)
            self.m = random.randint(self.n, self.s)
        else:
            self.s = self.op
            t0 = random.randint(1, self.s)
            t1 = random.randint(1, self.s)
            self.n = max(t0, t1)
            self.m = random.randint(self.n, self.s)
        
        relative_dist = (self.s - 1) / (self.max_edge - 1) # from 0 to 1
        temperature = 1.
        pos = [0.2, 0.5, 0.8]
        weight = [-(relative_dist - i)**2 / temperature for i in pos]
        
        self.d = choose_from_softmax([2, 3, 4], weight=weight)
        t0 = choose_from_softmax([2, 3, 4], weight=weight)
        t1 = choose_from_softmax([2, 3, 4], weight=weight)
        self.w0 = min(t0, t1)
        self.w1 = max(t0, t1)
        self.p = random.random()
        '''n = random.choice(range(1, max_op))
        m = random.choice(range(n, max_op))
        s = random.choice(range(m, max_op))'''
        # n, m, s = 12, 16, 16
        min_e = (self.d-1) * self.w0
        t0 = random.randint(min_e, self.max_edge)
        t1 = random.randint(min_e, self.max_edge)
        self.e = min(t0, t1)

    def gen_sol_op(self, style):
        if self.op != None:
            return self.op
        if style == "uniform":
            return random.choice(range(1, self.max_op+1))
        if style == "heavy":
            t0 = random.choice(range(1, self.max_op+1))
            t1 = random.choice(range(t0, self.max_op+1))
            t2 = random.choice(range(t1, self.max_op+1))
            return t2
        if style == "middle":
            t0 = random.choice(range(1, self.max_op+1))
            t1 = random.choice(range(1, self.max_op+1))
            t2 = random.choice(range(1, self.max_op+1))
            return max(t0, t1, t2)
        if style == "light":
            t0 = random.choice(range(1, self.max_op+1))
            t1 = random.choice(range(1, self.max_op+1))
            return min(t0, t1)

    def gen_prob(self, ava_hash, p_format: str, problem: Optional[Problem]=None):
        # Every Num built below reads params.reduce_mod, so the whole generation -- the
        # rejection loop included -- has to run inside one consistent setting.
        with mod_reduction(self.reduce_mod):
            return self._gen_prob(ava_hash, p_format, problem=problem)

    def _gen_prob(self, ava_hash, p_format: str, problem: Optional[Problem]=None):
        if not problem:
            while True:
                self.gen_param()

                # define permutation level
                if self.perm_level_ <= 4:
                    rand_perm = f"mild_{self.perm_level_}"
                    perm = True
                elif self.perm_level_ == 5:
                    rand_perm = "hard"
                    perm = True
                else:
                    rand_perm = "none"
                    perm = False

                # define detail level
                case0 = self.detail_level_ % 3
                rand_ = self.detail_level_ // 3
                case1 = rand_ % 2
                rand_ = rand_ // 2
                case2 = rand_
                if case0 == 0:
                    define_var = True
                    define_detail = True
                elif case0 == 1:
                    define_var = True
                    define_detail = False
                else:
                    define_var = False
                    define_detail = False
                name_omit = True if case1 == 1 else False
                cal_omit = True if case2 == 1 else False

                args = {
                    "rand_perm": rand_perm, # none or mild or hard, num >= 0
                    "define_var": define_var, # "Define a = Meta# Alice = 1" or "Meta# Alice = 1"
                    "define_detail": define_detail, # "Define a = Meta# Alice = 1" or "a = 1"
                    "inter_var": True, # "d = a + b + c" or "d = a + b, e = d + c"
                    "name_omit": name_omit, # hint part. "Define a = Meta# Alice = 1" or "a = 1". If define_var is False, name_omit must be False.
                    "cal_omit": cal_omit, # "Meta# Alice = a + b = 1 + 1 = 2" or "Meta# Alice = a + b = 2"
                    "dot": dot, # "Meta# Alice"
                    "symbol_method": "rand", # method to get a symbol. Can be 'rand' or 'seq'
                    "sol_sort": False, # not important for now.
                    "perm": perm, # make the solution's order different from problem's.
                }
                self.problem = Problem(self.d, self.w0, self.w1, self.e, self.p, args=args, be_shortest=self.be_shortest)
                feasible = self.problem.gen(self.n, self.m, self.s)
                if not feasible:
                    continue
                self.problem.to_problem()
                if self.problem.n_op != self.op_ or self.problem.to_hash() not in ava_hash:
                    # if self.problem.n_op != self.op_:
                    #     print(f"reject reason: {self.problem.n_op} != {self.op_}")
                    # else:
                    #     print("reject reason: not in ava_hash")
                    continue
                break
        else:
            self.problem = problem

        self.problem.perm_level_ = self.perm_level_
        self.problem.detail_level_ = self.detail_level_
        prob = " " + ". ".join(self.problem.problem[:-1]) + "."
        self.vanilla_prob = prob
        ques = " " + self.problem.problem[-1]
        self.ques = ques

        self.ques_token = tokenizer.encode(ques, return_tensors='pt')[0].tolist()

        self.prob = ""
        for char in p_format:
            if char == "p":
                self.prob += prob
            elif char == "q":
                self.prob += ques
        # self.prob += f" Answer in detail level_{self.detail_level_}."
        self.sol = " " + ". ".join(self.problem.solution) + "."

        prob_token = tokenizer.encode(self.prob, return_tensors='pt')[0].tolist()
        self.prob_token = prob_token
        sol_token = tokenizer.encode(self.sol, return_tensors='pt')[0].tolist()
        self.sol_token = sol_token
        ans_ = f" {self.problem.ans}"
        ans_token = tokenizer.encode(ans_, return_tensors='pt')[0].tolist()
        self.ans_token = ans_token
        self.token_id = [222] + prob_token + [223] + sol_token + [224] + ans_token + [50256]
        self.prob_id = [50256] + [222] + prob_token + [223]


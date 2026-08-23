# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from data_gen.categ import Data
from math_gen.graph_gen import Graph
from tools.tools import random_topological_sort, to_sketch, to_hash, wrap_label
import random, copy, math, hashlib, string
import networkx as nx
import numpy as np
from heapq import heappush, heappop
from itertools import count, product
from typing import List, Dict, Union, Callable, Any
from const import params
from const.params import mod, try_num, feasible_symbols

class Num(object):
    def __init__(self, a: Union[int, str]=None, mod=mod, mul=False) -> None:
        self.mod = mod
        if a == None:
            # A randomly drawn leaf is always in [0, mod), whether or not reduction is on:
            # params.reduce_mod widens the values a solution *computes*, not the quantities
            # the problem statement mentions.
            if mul:
                self.a = random.randint(1, self.mod - 1)
            else:
                self.a = random.randint(0, self.mod - 1)
        elif isinstance(a, str):
            self.a = int(a) % self.mod if params.reduce_mod else int(a)
        else: self.a = a % self.mod if params.reduce_mod else a
    
    def __add__(self, other):
        if isinstance(other, Num):
            return Num(self.a + other.a, self.mod)
        else:
            return Num(self.a + other, self.mod)
    
    def __sub__(self, other):
        if isinstance(other, Num):
            return Num(self.a - other.a, self.mod)
        else:
            return Num(self.a - other, self.mod)
    
    def __mul__(self, other):
        if isinstance(other, Num):
            return Num(self.a * other.a, self.mod)
        else:
            return Num(self.a * other, self.mod)
    
    def __eq__(self, other: Union['Num', int]) -> bool:
        if isinstance(other, int):
            return self.a == other
        else:
            return self.a == other.a
    
    def __str__(self) -> str:
        return str(self.a)

class Expression(object):
    def __init__(self, value: Union[List['Expression'], Num, int]=None, op: str=None, param: tuple=None, set_value: Union[Num, int]=None) -> None:
        '''
        Three cases:
        1. it is a parameter which can be defined by other things: value=list(...), op=..., param=param's code
        2. it is an intermidiate variable that is defined by other things: value=list(...), op=..., param=None
        3. it is just a number: value=value, op=None, param=None
        '''

        if isinstance(value, list):
            self.param_list = value
            if set_value != None:
                self.value = set_value if isinstance(set_value, Num) else Num(set_value)
        else:
            if isinstance(value, int):
                self.value = Num(value)
            elif isinstance(value, Num):
                self.value = value
            self.param_list = []

        self.op = op
        self.param = param

    @property
    def get_value(self):
        if hasattr(self, "value"):
            return self.value
        iterable = [param.get_value for param in self.param_list]
        if self.op == "diff":
            return iterable[0] - iterable[1]
        if self.op == "mul":
            return iterable[0] * iterable[1]

        return sum(iterable, start=Num(0))

    def simplify(self):
        if len(self.param_list) == 1:
            if self.param == None:
                if hasattr(self.param_list[0], "value"):
                    self.value = self.param_list[0].value
                self.op = self.param_list[0].op
                self.param = self.param_list[0].param
                self.param_list = self.param_list[0].param_list
                self.simplify()
                return
            
            elif self.param_list[0].param == None:
                if hasattr(self.param_list[0], "value"):
                    self.value = self.param_list[0].value
                self.op = self.param_list[0].op
                self.param_list = self.param_list[0].param_list
                self.simplify()
                return

        for param in self.param_list:
            param.simplify()

    def binarify(self):
        for param in self.param_list:
            param.binarify()
        '''n = len(self.param_list)
        pool = self.param_list
        now = self
        if n > 2:
            while n > 2:
                now.param_list = [pool.pop(0), Expression(value=[], op="add")]
                now = now.param_list[1]
                n -= 1

            now.param_list = pool'''
        if len(self.param_list) > 2:
            self = self.make_bi(self.param_list)
    
    def make_bi(self, lst: list) -> 'Expression':
        if len(lst) == 2:
            return Expression(value=lst, op="add")
        else:
            # exp0, exp1 = random.sample(lst, 2)
            exp0, exp1 = lst[0], lst[1]
            lst.remove(exp0)
            lst.remove(exp1)
            exp = Expression(value=[exp0, exp1], op="add")
            # lst.append(exp)
            lst.insert(0, exp)
            return self.make_bi(lst=lst)
            

    def display(self, level=0, scale=4):
        space = " " * (4*level)
        op = " -" + self.op if self.op != None else ""
        if hasattr(self, "value"):
            if self.param == None:
                n = "rand"
            else:
                n = self.param
            print(f"{space}{n} ({self.value.a}){op}")
        else:
            if self.param:
                txt = f"{self.param}"
            else:
                txt = "temp"
            print(f"{space}{txt} ({self.get_value.a}){op}")
        
        for param in self.param_list:
            param.display(level=level+1, scale=scale)

class Problem(Graph):
    rand_perm: str
    define_var: bool
    define_detail: bool
    inter_var: bool # decompose into small steps
    name_omit: bool # hint part
    cal_omit: bool # cal part
    dot: str
    symbol_method: str
    sol_sort: bool
    def __init__(self, d, w0, w1, e, p, args: dict, dist: Dict[str, Callable[[], Any]]=None, be_shortest: bool=True) -> None:
        '''
        Only define_detail=True is verified. When using False case, please print out to see if the output is correct.

        Generate the problem class.
        Handle all the graph structures, parameter names in both the problems and the solutions.

        d is the generator for depth
        w0, w1 is the min and max width
        e is the generator for the number of edges
        p is the probability of adding vertex when it's possible
        if e does not satisfy: (d-1) * w1 ** 2 >= e >= (d-1) * w0,
        choose the feasible e closest to the original e.
        '''
        super().__init__(d, w0, w1, e, p, args['perm'], dist=dist)
        self.args = args
        for key, val in args.items():
            setattr(self, key, val)
        self.be_shortest = be_shortest
        self.lookup:Dict[tuple, Num] = {} # map from parameter to its value
        self.name_dict:Dict[tuple, str] = {} # map from parameter to param_name (symbol)
        self.prob_dict:Dict[tuple, str] = {} # map from parameter to its problem
        self.sketch:Dict[tuple, Expression] = {} # map from parameter to (op0, op1, val) pair. to expression instance
        self.problem:List[str] = []
        self.question = []
        self.solution:List[str] = []
        self.answer = []
        self.symbols = copy.deepcopy(feasible_symbols)
        self.all_symbols = copy.deepcopy(feasible_symbols)
        self.perm_level_ = -1
        self.detail_level_ = -1

    def gen(self, n, m, s, first=-1, max_param=4, fix_categ: Union[None, int]=None):
        '''
        n is the operation needed for internal parameters
        m is the total number of minimal required operations
        s is the total number of operations
        first determine the type of the final question
        max_param determine the maximal number of parametered can appear in a single sentence
        '''
        count = 0
        for i in range(try_num):
            self.init()
            data = Data()
            self.ln = data(None, self.d, fix_categ=fix_categ) # layer name
            self.N = [] # Nodes' name
            for i in range(self.d):
                self.N.append(data(self.ln[i], self.l[i]))
            self.unique_name = data.unique
            self.assign_unique()
            self.choose_param(n, m)
            self.setup_template()
            # print("finished")
            valid = self.reasonable_sort(first=first)
            count += 1
            if valid:
                # print(f"count: {count}")
                self.design(s, max_param=max_param)
                self.fill_all()
                self.ques_pos = len(self.topological_order)
                self.ques_idx = self.topological_order[-1]
                self.design_unused(max_param=max_param)

                # print([n for n in self.template.nodes])

                self.ori_order = copy.deepcopy(self.problem_order)
                if self.perm:
                    self.problem_order = random_topological_sort(self.template)
                
                if not self.be_shortest:
                    random_solution_order = random_topological_sort(self.template)
                    random_solution_order.remove(self.rand)
                    query_idx = random_solution_order.index(self.ques_idx)
                    self.random_solution_order = random_solution_order[:query_idx+1]
                    self.ques_pos = len(self.random_solution_order)
                    
                return True
        return False

    def assign_unique(self):
        '''
        assign uniqueness to parameters according to data.
        '''
        for i in range(self.d):
            for j, name in enumerate(self.N[i]):
                if name in self.unique_name:
                    self.unique.append((i, j))
                    self.graph.nodes[(i, j)]['unique'] = True

    def to_problem(self):
        '''
        Suppose we've already used self.gen to generate a problem.
        Now we can use this method to translate the abstract template to discrete problems using self.ln and self.N.
        '''
        for param in self.problem_order:
            self.parse(param)
            '''if param[0] != -1:
                exp = self.sketch[param]
                exp.display()'''
        
        l, i, j, k = self.ques_idx
        if l == 0:
            ques = f"How many {self.N[i+1][k]} does {self.N[i][j]} have?"
        elif l == 1:
            ques = f"How many {self.ln[k]} does {self.N[i][j]} have?"

        self.shuffle()

        # generate problem and solution
        for param in self.problem_order:
            if param[0] == 0:
                self.problem.append(self.prob_dict[param])
        self.problem.append(ques)

        # self.draw()
        
        '''if self.sol_sort:
            self.sol_template = nx.DiGraph()'''
        my_queue = self.topological_order if self.be_shortest else self.random_solution_order
        for param in my_queue:
            self.decode(param)
        
        self.ans = self.lookup[self.ques_idx].a

        self.set_whole_template()
        # self.solution.append(f"The answer is {self.lookup[self.ques_idx].a}.")

    def to_partial_problem(self, partial=None):
        # choose params in original problem
        self.partial_problem = []
        self.valid_prob_param = [param for param in self.problem_order if param[0] == 0]
        if partial == None:
            partial = random.randint(1, len(self.valid_prob_param))
        elif isinstance(partial, float):
            partial = math.ceil(partial * len(self.valid_prob_param))
        elif isinstance(partial, int):
            partial = min(partial, len(self.valid_prob_param))
        else:
            raise ValueError(f"The type of partial should be None or float or int, but it is {type(partial)} here.")
        for i in range(partial):
            param = self.valid_prob_param[i]
            self.partial_problem.append(self.prob_dict[param])
        
        # new partial template
        self.partial_template = nx.DiGraph()
        self.partial_inter = []
        for param in self.valid_prob_param[:partial]:
            if param not in self.partial_template.nodes():
                self.partial_template.add_node(param)
            for dep_param in self.template.predecessors(param):
                if dep_param[0] == 1:
                    self.partial_inter.append(dep_param)
                if dep_param not in self.partial_template.nodes():
                    self.partial_template.add_node(dep_param)
                self.partial_template.add_edge(dep_param, param)
        
        self.partial_inst_param = [param for param in self.partial_template.nodes() if param[0] == 0]
        self.partial_param = [param for param in self.partial_template.nodes() if param[0] != -1]
        
        # assign inter params
        while self.partial_inter:
            param = self.partial_inter.pop()
            self.add_partial_param(param)

    def to_hash(self, mod_num=mod, method='sol'):
        '''
        return a hash value in [0, 1, ..., mod-1]
        use after self.to_problem()
        '''
        problem = " " + ". ".join(self.problem)
        solution = " " + ". ".join(self.solution) + "."
        sketch = to_sketch(self, prob=problem, sol=solution)
        hash_val = to_hash(sketch[method], mod_num=mod_num)
        return hash_val

    def add_partial_param(self, param):
        _, i, j, k = param
        if i+1 == k:
            for l_, i_, j_, k_ in self.partial_inst_param:
                if l_==0 and i_==i and j_==j:
                    self.partial_template.add_edge((l_, i_, j_, k_), param)
        else:
            for l_, i_, j_, k_ in self.partial_inst_param:
                if l_==0 and i_==i and j_==j:
                    self.partial_template.add_edge((l_, i_, j_, k_), param)
                    if (1, i_+1, k_, k) not in self.partial_param:
                        self.partial_param.append((1, i_+1, k_, k))
                        self.partial_inter.append((1, i_+1, k_, k))
                        self.partial_template.add_node((1, i_+1, k_, k))
                    self.partial_template.add_edge((1, i_+1, k_, k), param)

    def parse(self, param, inter_only=False):
        '''
        generate problem
        '''
        l, i, j, k = param
        if l == 0:
            if inter_only:
                return
            pre = list(self.template.predecessors(param))
            if not pre:
                self.prob_dict[param] = f"{self.get_name((i, j))} has {self.get_name((i+1, k))}"
                self.lookup[param] = Num(1)
                self.sketch[param] = Expression(1, param=param)
                return

            txt = [f"The number of {self.get_param(param)} equals"]
            rand = None
            exp0 = Expression(param=param)
            if self.rand in pre:
                pre.remove(self.rand)
                if len(pre) == 0:
                    rand = Num()
                    num0 = rand
                    exp0.param_list.append(Expression(rand))
                    txt.append(f"{rand.a}")
                    self.prob_dict[param] = " ".join(txt)
                    self.sketch[param] = exp0
                    self.lookup[param] = rand
                    return
                if random.random() < 0.5:
                    op0 = "add"
                    rand = Num()
                    num0 = rand
                    exp0.param_list.append(Expression(rand))
                    txt.append(f"{rand.a} more than")
                    exp0.op = "add"
                    exp1 = Expression()
                    exp0.param_list.append(exp1)
                else:
                    op0 = "mul"
                    rand = Num(mul=True)
                    num0 = rand
                    exp0.param_list.append(Expression(rand))
                    txt.append(f"{rand.a} times as much as")
                    exp0.op = "mul"
                    exp1 = Expression()
                    exp0.param_list.append(exp1)
            else:
                op0 = None
                exp1 = exp0

            n_param = len(pre)
            num1 = Num(0)
            if n_param == 1:
                num1 += self.lookup[pre[0]]
                txt.append(self.get_param(param=pre[0]))
                exp1.param_list.append(Expression(self.lookup[pre[0]], param=pre[0]))
            elif n_param == 2:
                if random.random() < 0.5:
                    txt.append(f"the sum of {self.get_param(param=pre[0])} and {self.get_param(param=pre[1])}")
                    num1 += self.lookup[pre[0]] + self.lookup[pre[1]]
                    exp1.op = "sum"
                    exp1.param_list.append(Expression(self.lookup[pre[0]], param=pre[0]))
                    exp1.param_list.append(Expression(self.lookup[pre[1]], param=pre[1]))
                else:
                    txt.append(f"the difference of {self.get_param(param=pre[0])} and {self.get_param(param=pre[1])}")
                    num1 += self.lookup[pre[0]] - self.lookup[pre[1]]
                    exp1.op = "diff"
                    exp1.param_list.append(Expression(self.lookup[pre[0]], param=pre[0]))
                    exp1.param_list.append(Expression(self.lookup[pre[1]], param=pre[1]))
            else:
                exp1.op = "sum"
                # len(pre) >= 3
                txt.append("the sum of")
                for i, param_ in enumerate(pre):
                    if i == len(pre) - 1:
                        txt.append(f"and {self.get_param(param=param_)}")
                    elif i == len(pre) - 2:
                        txt.append(f"{self.get_param(param=param_)}")
                    else:
                        txt.append(f"{self.get_param(param=param_)},")

                    num1 += self.lookup[param_]
                    exp1.param_list.append(Expression(value=self.lookup[param_], param=param_))
            
            self.sketch[param] = exp0
            self.prob_dict[param] = " ".join(txt)
            if op0 == None:
                self.lookup[param] = num1
            elif op0 == "add":
                self.lookup[param] = num0 + num1
            elif op0 == "mul":
                self.lookup[param] = num0 * num1
            return
        elif l == 1:
            '''only need to complete the lookup table'''
            num = Num(0)
            exp0 = Expression(op="sum", param=param)
            if i+1 == k:
                for param_ in self.template.predecessors(param):
                    num += self.lookup[param_]
                    exp0.param_list.append(Expression(value=self.lookup[param_], param=param_))
            else:
                for k_ in [idx for idx in range(self.l[i+1]) if self.G[i][j, idx]]:
                    num += self.lookup[(0, i, j, k_)] * self.lookup[(1, i+1, k_, k)]
                    param0 = Expression(value=self.lookup[(0, i, j, k_)], param=(0, i, j, k_))
                    param1 = Expression(value=self.lookup[(1, i+1, k_, k)], param=(1, i+1, k_, k))
                    exp0.param_list.append(Expression(value=[param0, param1], op="mul"))
            self.lookup[param] = num
            self.sketch[param] = exp0

    def decode(self, param):
        '''
        generate solution
        '''
        # print(param)
        exp = self.sketch[param]

        # self.solution.append(f"param {param} ----------------------------------------------")

        exp.simplify()
        exp.binarify()
        # exp.display()
        _, _, _ = self.to_sol(exp)

    def shuffle(self):
        '''
        shuffle the problem description.
        mild 0, 1, 2, 3, 4 + hard
        '''
        args = self.rand_perm.split("_")
        if args[0] == "none":
            # print('Shuffle Level: None')
            return
        if args[0] == "mild":
            num = int(args[1])
            # print('Shuffle Level: mild', num)
            idxs = []
            for i in range(self.n_param):
                idxs.append((i+random.randint(0, num), i))
            sorted_idxs = []
            for i in range(self.n_param + num):
                pool = [k for j, k in idxs if j == i]
                random.shuffle(pool)
                for idx in pool:
                    sorted_idxs.append(idx)
            
            self.problem_order.remove(self.rand)
            problem_order = [self.problem_order[idx] for idx in sorted_idxs]
            self.problem_order = problem_order + [self.rand]
            
        if args[0] == "hard":
            # print('Shuffle Level: totally random')
            random.shuffle(self.problem_order)
            return

    def get_param(self, param):
        '''
        (each) something's something
        '''
        l, i, j, k = param
        name = ""
        if l == 0:
            name0 = self.get_name((i, j), True)
            name1 = self.get_name((i+1, k))
            return f"{name0}'s {name1}"
        if l == 1:
            name0 = self.get_name((i, j), True)
            name1 = self.ln[k]
            return f"{name0}'s {name1}"

    def get_name(self, param, arg=False):
        '''
        <<something>>
        '''
        i, j = param
        if arg and param not in self.unique:
            return f"each {self.N[i][j]}"
        return f"{self.N[i][j]}"

    def get_ntn(self, param):
        '''
        something# something
        '''
        if param == None:
            return "None"
        l, i, j, k = param
        if l == 0:
            return f"{self.N[i][j]}{self.dot}{self.N[i+1][k]}"
        if l == 1:
            return f"{self.N[i][j]}{self.dot}{self.ln[k]}"

    def get_symbol(self):
        '''
        a
        '''
        if not self.symbols:
            return '...'
        if self.symbol_method == 'rand':
            a = random.choice(self.symbols)
            self.symbols.remove(a)
            return a
        elif self.symbol_method == 'seq':
            a = self.symbols.pop(0)
            return a

    def to_sol(self, exp: Expression, append=True):
        '''
        def_part, hint_part, cal_part, res_part
        '''
        if exp.param != None and exp.param in self.name_dict:
            return [], self.lookup[exp.param], self.name_dict[exp.param]
        if exp.param == None and not exp.param_list:
            return [], exp.value, str(exp.value.a)
        def_part, hint_part, cal_part, res_part = None, None, None, None
        num_list = []
        name_list = []
        output_whole_lst = []
        if not exp.param_list:
            if hasattr(exp, "value"):
                num_list = [exp.value]
                name_list = []
        else:
            for param in exp.param_list:
                output_lst, num_, name_ = self.to_sol(param, append=False)
                output_whole_lst += output_lst
                num_list.append(num_)
                name_list.append(name_)
        
        if exp.op == "diff":
            sign = " - "
        elif exp.op == "mul":
            sign = " * "
        else: # add, sum or None
            sign = " + "
        
        if not self.name_omit and name_list:
            hint_part = sign.join(name_list)
        
        if not self.cal_omit and len(num_list) > 1:
            cal_lst = [str(num_.a) for num_ in num_list]
            cal_part = sign.join(cal_lst)

        if exp.op == "diff":
            ans = num_list[0] - num_list[1]
        elif exp.op == "mul":
            ans = num_list[0] * num_list[1]
        else:
            ans = sum(num_list, start=Num(0))
        res_part = str(ans.a)

        if exp.param is not None:
            if not self.define_var:
                def_part = self.get_ntn(exp.param)
                name = def_part
            elif self.define_detail:
                name = self.get_symbol()
                def_part = f"Define {self.get_ntn(exp.param)} as {name}"
            else:
                name = self.get_symbol()
                def_part = f"Define {name}"
        else:
            def_part = self.get_symbol()
            name = def_part

        if exp.param is not None:
            self.name_dict[exp.param] = name
            if self.lookup[exp.param].a != ans.a:
                raise ValueError(f"{self.lookup[exp.param].a} != {ans.a}")
        self.lookup[def_part] = ans
        
        output_lst = []
        for part in [def_part, hint_part, cal_part, res_part]:
            if part is not None:
                output_lst.append(part)
        output = " = ".join(output_lst)

        if append:
            current_sentence = []
            current_sentence.append(def_part)
            for previous_output in output_whole_lst:
                current_sentence.append(previous_output)
            so_part = f"so {name}"
            post_output_lst = []
            for part in [so_part, hint_part, cal_part, res_part]:
                if part is not None:
                    post_output_lst.append(part)
            post_output = " = ".join(post_output_lst)
            current_sentence.append(post_output)
            self.solution.append("; ".join(current_sentence))
            # print(f"Debug: output = {output}")
            return [], ans, name
        
        else:
            output_whole_lst.append(output)
            return output_whole_lst, ans, name

    def draw_structure(self, ax=None):
        import matplotlib.patches as mpatches
        # define color map:
        color = {
            False: "#1f78b4",
            True: "#228B22" #"#FF7F50",
        }
        if hasattr(self, "partial_param"):
            selected = set()
            for l, i, j, k in self.partial_param:
                if l == 0:
                    selected.add((i, j))
                    selected.add((i+1, k))
            map_to_color = {True: 1, False: 0.5}
            node_alphas = [map_to_color[node in selected] for node in self.graph.nodes()]
        else:
            node_alphas = 1
        pos = {}
        labels = {}
        max_width = max(self.l)
        for i in range(self.d):
            ax.text(-0.4 * max_width, self.d-1-i, self.ln[i])
            for j in range(self.l[i]):
                pos[(i, j)] = (j, self.d-1-i)
                labels[(i, j)] = self.N[i][j] # .replace("/", "\n")
        node_colors = [color[node[1]['unique']] for node in self.graph.nodes(data=True)]
        # newly added on 12-21
        max_chars_per_line = 12  # Set the max number of chars per line in the label
        for node, label in labels.items():
            labels[node] = wrap_label(label, max_chars_per_line)

        #(self.graph, pos, ax=ax, node_color=node_colors, label=labels, alpha=node_alphas, bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        #nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color=node_colors, label=labels, alpha=node_alphas, bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        #nx.draw_networkx_edges(self.graph, pos, ax=ax)
        #nx.draw_networkx_labels(self.graph, pos, labels=labels, ax=ax, font_family="monospace")
        nx.draw(self.graph, pos, node_color=node_colors,font_size=10, node_size=1, with_labels=True, ax=ax, labels=labels, bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'), margins=(0.25,0.1)) #, min_target_margin=25,   alpha=node_alphas, 

        # create legend:
        unique_patch = mpatches.Patch(color='#228B22', label='unique item')
        duplicate_patch = mpatches.Patch(color='#1f78b4', label='duplicatable item')

        # ax.legend(handles=[unique_patch, duplicate_patch], loc='upper center', bbox_to_anchor=(0.5, -0.), ncol=2)

    def set_whole_template(self):
        whole_template = copy.deepcopy(self.template)
        self.all_param = [(0, i, j, k) for i in range(self.d-1) for j in range(self.l[i]) for k in range(self.l[i+1])]
        self.all_param += [(1, i, j, k) for i in range(self.d-1) for j in range(self.l[i]) for k in range(i+1, self.d)]
        for param in self.all_param:
            if param not in self.problem_order:
                whole_template.add_node(param)
        
        for l, i, j, k in self.all_param:
            if (l, i, j, k) not in self.problem_order and l == 1:
                for x in range(self.l[i+1]):
                    if self.G[i][j, x]:
                        whole_template.add_edge((0, i, j, x), (1, i, j, k))
                        # print(f"add {(0, i, j, x)} -> {(1, i, j, k)}")
                if k - i > 1:
                    for x in range(self.l[i+1]):
                        if self.G[i][j, x]:
                            whole_template.add_edge((1, i+1, x, k), (1, i, j, k))
                            # print(f"add {(1, i+1, x, k)} -> {(1, i, j, k)}")
        
        # gen labels
        self.whole_template = whole_template

    def lora_label(self, keys: list):
        '''
        return a numpy tensor with shape: (1 + len(self.topological_order), len(self.all_param), len(keys)).
        In each position, for each parameter and for each key, predict its label (int).
        '''
        if len(keys) == 0:
            return None
        labels = np.zeros((1 + len(self.topological_order), len(self.all_param), len(keys)), dtype=int)

        in_degree = dict(self.whole_template.in_degree())
        zero_in_degree = [v for v, d in in_degree.items() if d == 0]
        for i_, param in enumerate([(-1, 0, 0, 0)] + self.topological_order):
            zero_in_degree.remove(param)
            for v in self.whole_template.successors(param):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    zero_in_degree.append(v)
            for j, q_param in enumerate(self.all_param):
                for k, key in enumerate(keys):
                    if key == "known":
                        if i_ > 0 and q_param in self.topological_order[:i_]:
                            labels[i_, j, k] = 1
                    if key == "can_next":
                        if q_param in zero_in_degree:
                            labels[i_, j, k] = 1
                    if key == "nece_next":
                        if q_param in zero_in_degree and q_param in self.topological_order:
                            labels[i_, j, k] = 1
                    if key == "nece":
                        if q_param in self.topological_order:
                            labels[i_, j, k] = 1
                    if key == "val":
                        if i_ > 0 and q_param in self.topological_order[:i_]:
                            labels[i_, j, k] = self.lookup[q_param].a
                        else:
                            labels[i_, j, k] = mod
        
        return labels

    def lora_label2(self, key: str):
        '''
        return a numpy tensor with shape: (len(self.all_param), len(self.all_param)).
        the i-j entry represents whether param_i depends on param_j.
        '''
        index_dict = {}
        nece_index = []
        unnece_index = []
        for i, param in enumerate(self.all_param):
            index_dict[param] = i
            if param in self.topological_order:
                nece_index.append(i)
            else:
                unnece_index.append(i)
        
        # gen labels
        n = len(self.all_param)
        labels = np.zeros((n, n), dtype=int)
        
        if key in ['dep', 'dep_nece', 'dep_unnece']:
            tc = nx.transitive_closure_dag(self.whole_template)
        elif key in ['neighbor']:
            tc = self.whole_template
        else:
            raise ValueError(f"key ({key}) must be in list ['dep', 'dep_nece', 'dep_unnece', 'neighbor']")

        for (node_i, node_j) in tc.edges():
            if node_i != (-1, 0, 0, 0):
                i = index_dict[node_i]
                j = index_dict[node_j]
                labels[j, i] = 1 # if j depends on i, set the label to 1.

        return labels, nece_index, unnece_index

    def lora_label3(self):
        '''
        used for partial
        '''
        # all parameters
        partial_template = copy.deepcopy(self.partial_template)
        structure_nodes = []
        for l, i, j, k in self.partial_param:
            if l == 0:
                if (i, j) not in structure_nodes:
                    structure_nodes.append((i, j))
                if (i+1, k) not in structure_nodes:
                    structure_nodes.append((i+1, k))
        
        part_d = max(i for (i, j) in structure_nodes) + 1
        self.all_partial_param = [(0, i, j, j_) for (i, j), (i_, j_) in product(structure_nodes, structure_nodes) if i+1==i_]
        self.all_partial_param += [(1, i, j, k) for (i, j) in structure_nodes if i<part_d-1 for k in range(i+1, part_d)]
        self.all_partial_param = list(set(self.all_partial_param + self.partial_param))
        # print(self.all_partial_param)

        index_dict = {}
        for i, param in enumerate(self.all_partial_param):
            index_dict[param] = i
            if param not in self.partial_param:
                partial_template.add_node(param)
        
        for param in self.all_partial_param:
            l, i, j, k = param
            if (l, i, j, k) not in self.partial_param and l == 1:
                if i+1 == k:
                    for l_, i_, j_, k_ in self.partial_inst_param:
                        if l_==0 and i_==i and j_==j:
                            partial_template.add_edge((l_, i_, j_, k_), param)
                else:
                    for l_, i_, j_, k_ in self.partial_inst_param:
                        if l_==0 and i_==i and j_==j:
                            partial_template.add_edge((l_, i_, j_, k_), param)
                            partial_template.add_edge((1, i_+1, k_, k), param)
        
        # gen labels
        n = len(self.all_partial_param)
        labels = np.zeros((n, n), dtype=int)
        
        tc = nx.transitive_closure_dag(partial_template)

        for (node_i, node_j) in tc.edges():
            if node_i != (-1, 0, 0, 0):
                i = index_dict[node_i]
                j = index_dict[node_j]
                labels[j, i] = 1 # if j depends on i, set the label to 1.

        return labels

    def lora_label_dip(self):
        index_dict0 = {}
        for i, param in enumerate(self.all_param):
            index_dict0[param] = i
        index_dict1 = {}
        for i, param in enumerate(self.topological_order):
            index_dict1[param] = i
        
        len0 = len(self.topological_order)
        len1 = len(self.all_param)
        
        labels = np.zeros((len0, len1), dtype=int)

        for i in range(len0):
            node0 = self.topological_order[i]
            for j in range(len1):
                node1 = self.all_param[j]
                if (node1, node0) in self.whole_template.edges():
                    labels[i, j] = 1
        
        return labels

    def name2param(self, param_name:str):
        if not hasattr(self, "name2param_dict"):
            self.name2param_dict = {}
            for param in self.all_param:
                self.name2param_dict[self.get_ntn(param)] = param
        
        if param_name.startswith(" "):
            param_name = param_name[1:]
        if param_name.startswith("each"):
            param_name = param_name[5:]
        
        if param_name in self.name2param_dict.keys():
            return self.name2param_dict[param_name]
        else:
            return None

    def replace_names(self):
        problem2 = copy.deepcopy(self)

        data = Data()
        problem2.ln = data(None, self.d) # layer name
        problem2.N = [] # Nodes' name
        problem2.unique = []
        for i in range(problem2.d):
            problem2.N.append(data(problem2.ln[i], problem2.l[i]))
        problem2.unique_name = data.unique
        problem2.assign_unique()

        # lookup and sketch are already defined.

        # problem part
        for param in problem2.problem_order:
            if param[0] != 0:
                continue
            txt = self.prob_dict[param]
            counter = 0

            txt = txt.replace(self.get_param(param), f"        {counter}        ", 1)
            counter += 1

            for pre_param in self.template.predecessors(param):
                if pre_param[0] != -1:
                    txt = txt.replace(self.get_param(pre_param), f"        {counter}        ", 1)
                    counter += 1
            
            re_counter = 0
            txt = txt.replace(f"        {re_counter}        ", problem2.get_param(param), 1)
            re_counter += 1

            for pre_param in self.template.predecessors(param):
                if pre_param[0] != -1:
                    txt = txt.replace(f"        {re_counter}        ", problem2.get_param(pre_param), 1)
                    re_counter += 1
            
            problem2.prob_dict[param] = txt
        
        l, i, j, k = problem2.ques_idx
        if l == 0:
            ques = f"How many {problem2.N[i+1][k]} does {problem2.N[i][j]} have?"
        elif l == 1:
            ques = f"How many {problem2.ln[k]} does {problem2.N[i][j]} have?"

        problem2.problem = []
        for param in problem2.problem_order:
            if param[0] == 0:
                problem2.problem.append(problem2.prob_dict[param])
        problem2.problem.append(ques)

        # solution part
        for i, param in enumerate(problem2.topological_order):
            if not problem2.define_var:
                name = problem2.get_ntn(param)
            else:
                name = self.name_dict[param]
            
            problem2.name_dict[param] = name

            problem2.solution[i] = self.solution[i].replace(self.get_ntn(param), problem2.get_ntn(param), 1)
        
        problem2.name2param_dict = {}
        for param in problem2.all_param:
            problem2.name2param_dict[problem2.get_ntn(param)] = param

        return problem2

def auto_easy():
    '''
    auto gen easy problem template
    '''
    d = random.choice((2, 3))
    t0 = random.choice((2, 3, 4))
    t1 = random.choice((2, 3, 4))
    w0 = min(t0, t1)
    w1 = max(t0, t1)
    p = random.random()
    max_op = 11 # fixed hyper param
    n = random.choice(range(1, max_op))
    m = random.choice(range(n, max_op))
    s = random.choice(range(m, max_op))
    # n, m, s = 12, 16, 16
    if random.random() < 0.5:
        e = random.choice(range(max_op))
    else:
        e = random.choice(range(s, max_op))
    print(d, w0, w1, e, p)
    print(n, m, s)

    # 12 cases appear uniformly.
    args = {
        "rand_perm": "mild_0", # none or mild or hard, num >= 0
        "define_var": True, # "Define a = Meta# Alice = 1" or "Meta# Alice = 1"
        "define_detail": True, # "Define a = Meta# Alice = 1" or "a = 1"
        "inter_var": True, # "d = a + b + c" or "d = a + b, e = d + c"
        "name_omit": False, # hint part. "Define a = Meta# Alice = 1" or "a = 1". If define_var is False, name_omit must be False.
        "cal_omit": True, # "Meta# Alice = a + b = 1 + 1 = 2" or "Meta# Alice = a + b = 2"
        "dot": "# ", # "Meta# Alice"
        "symbol_method": "seq", # method to get a symbol. Can be 'rand' or 'seq'
        "sol_sort": False, # not important for now.
        "perm": True, # make the solution's order different from problem's.
    }

    problem = Problem(d, w0, w1, e, p, args=args)
    problem.gen(n, m, s)
    problem.to_problem()
    print("problem-----------------------------------------------------------")
    for sentence in problem.problem:
        print(sentence)
    print("solution----------------------------------------------------------")
    for sol_step in problem.solution:
        print(sol_step)
    problem.display(detail=False)
    problem.draw()

def speed_easy(num, phase):
    for i in range(num):
        d = random.choice((2, 3))
        t0 = random.choice((2, 3))
        t1 = random.choice((2, 3))
        w0 = min(t0, t1)
        w1 = max(t0, t1)
        p = random.random()
        n = random.choice(range(1, 11))
        m = random.choice(range(n, 11))
        s = random.choice(range(m, 11))
        if random.random() < 0.5:
            e = random.choice(range(11))
        else:
            e = random.choice(range(s, 11))
        problem = Problem(d, w0, w1, e, p)
        problem.gen(n, m, s)
        if i % phase == phase - 1:
            print(i, i/num)

def exp_debug():
    '''x0 = Expression(5)
    x1 = Expression(7)
    x3 = Expression(2)
    e0 = Expression([x0, x1, x3], op="diff")
    e3 = Expression([e0])
    e1 = Expression([e3])
    x2 = Expression(3)
    e2 = Expression([e1, x2], op="sum", param=(1, 2))
    e2 = Expression(0)
    print()
    e2.display()
    e2.simplify()
    e2.binarify()
    print()
    e2.display()'''
    x0 = Expression(0)
    x1 = Expression(param=(1, 0))
    x2 = Expression(param=(0, 0))
    print(x2)
    pass

if __name__ == "__main__":
    # speed_easy(10000, 100)
    auto_easy()
    # exp_debug()
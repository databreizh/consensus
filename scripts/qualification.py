"""Exact fixed-input qualification model. Independent reimplementation, 2026-11-26.
No observed qualification labels are used to select a probe before it is paid.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math

@dataclass(frozen=True)
class Pair:
    u: int
    v: int
    probability: float
    cost: float
    name: str

@dataclass(frozen=True)
class Instance:
    vertices: int
    violations: tuple
    pairs: tuple
    threshold: int = 2

    def __post_init__(self):
        if self.vertices < 1 or not 0 <= self.threshold <= len(self.violations):
            raise ValueError('Invalid vertices or threshold')
        if len(set(self.violations)) != len(self.violations) or any(v < 0 or v >= self.vertices for v in self.violations):
            raise ValueError('Invalid violation vertices')
        seen = set()
        for p in self.pairs:
            if not 0 <= p.u < p.v < self.vertices or (p.u,p.v) in seen:
                raise ValueError('Pairs must be unique ordered vertex pairs')
            if not 0 < p.probability < 1 or not math.isfinite(p.cost) or p.cost <= 0:
                raise ValueError('Strictly positive finite costs, probabilities in (0,1) required')
            seen.add((p.u,p.v))

@lru_cache(maxsize=None)
def repair_outcome(vertices, violations, threshold, edges):
    """1 = certainly false, 2 = certainly true, 3 = contingent, over all repairs.
    Edges outside the candidate universe are explicitly compatible in this model.
    """
    neighbors = [0]*vertices
    for u,v in edges:
        neighbors[u] |= 1<<v
        neighbors[v] |= 1<<u
    all_vertices = (1<<vertices)-1
    violation_mask = sum(1<<v for v in violations)
    values = 0
    for retained in range(1<<vertices):
        covered = retained
        bits = retained
        independent = True
        while bits:
            bit = bits & -bits
            v = bit.bit_length()-1
            if neighbors[v] & retained:
                independent = False
                break
            covered |= neighbors[v]
            bits -= bit
        if independent and covered == all_vertices:
            truth = (retained & violation_mask).bit_count() >= threshold
            values |= 2 if truth else 1
            if values == 3:
                return values
    return values

class Planner:
    """Ternary partial states: 0 unknown, 1 compatible, 2 incompatible.
    Determinacy requires the SAME repair value set in every completion.
    """
    def __init__(self, instance):
        self.instance = instance
        self.pairs = instance.pairs
        self.m = len(self.pairs)
        self.powers = tuple(3**i for i in range(self.m))
        self.ratio_order = sorted(range(self.m), key=lambda i:(self.pairs[i].cost/self.pairs[i].probability,i))
        self._outcomes = lru_cache(maxsize=None)(self._possible_outcomes)
        self._costs = {}

    def unknown(self,state):
        if not 0 <= state < 3**self.m:
            raise ValueError('Partial state outside universe')
        return tuple(i for i,power in enumerate(self.powers) if state//power%3 == 0)

    def _possible_outcomes(self,state):
        unknown = self.unknown(state)
        if not unknown:
            edges = tuple(sorted((p.u,p.v) for i,p in enumerate(self.pairs) if state//self.powers[i]%3 == 2))
            value = repair_outcome(self.instance.vertices,self.instance.violations,self.instance.threshold,edges)
            return 1<<value
        i = unknown[0]
        return self._outcomes(state+self.powers[i]) | self._outcomes(state+2*self.powers[i])

    def determined(self,state):
        values = self._outcomes(state)
        return values & (values-1) == 0

    def next_pair(self,state,method):
        unknown = self.unknown(state)
        if not unknown or self.determined(state):
            return None
        if method == 'ratio':
            return next(i for i in self.ratio_order if i in unknown)
        if method == 'greedy':
            def key(i):
                p = self.pairs[i]
                stop = ((1-p.probability)*self.determined(state+self.powers[i])
                        +p.probability*self.determined(state+2*self.powers[i]))
                return (-stop/p.cost,p.cost/p.probability,i)
            return min(unknown,key=key)
        if method == 'optimal':
            return min(unknown,key=lambda i:(self._probe_cost(state,i,method),i))
        raise ValueError('Random is a distribution over probes, not a deterministic next pair')

    def _probe_cost(self,state,i,method):
        p = self.pairs[i]
        value = self.cost_function(method)
        return p.cost+(1-p.probability)*value(state+self.powers[i])+p.probability*value(state+2*self.powers[i])

    def cost_function(self,method):
        if method not in {'random','ratio','greedy','optimal'}:
            raise ValueError('Unknown policy: '+method)
        if method not in self._costs:
            @lru_cache(maxsize=None)
            def value(state):
                if self.determined(state):
                    return 0.0
                unknown = self.unknown(state)
                if method == 'random':
                    return sum(self._probe_cost(state,i,method) for i in unknown)/len(unknown)
                if method == 'optimal':
                    return min(self._probe_cost(state,i,method) for i in unknown)
                return self._probe_cost(state,self.next_pair(state,method),method)
            self._costs[method] = value
        return self._costs[method]

    def expected_costs(self):
        costs = {'assess_all':sum(p.cost for p in self.pairs)}
        costs.update({m:self.cost_function(m)(0) for m in ['random','ratio','greedy','optimal']})
        return costs

    def clear(self):
        self._outcomes.cache_clear()
        for f in self._costs.values(): f.cache_clear()
        self._costs.clear()

CROSS = tuple(combinations(range(4),2))
def four_posts(topology,cross_cost):
    pairs = [Pair(i,4+i,.5,1.,'e'+str(i+1)) for i in range(4)]
    pairs += [Pair(u,v,.3,float(cross_cost),'x%d%d'%(u+1,v+1)) for i,(u,v) in enumerate(CROSS) if topology>>i&1]
    return Instance(8,(0,1,2,3),tuple(pairs))

def worked_example():
    return Instance(4,(0,1),(Pair(0,2,.5,1,'e1'),Pair(1,3,.5,1,'e2'),Pair(0,1,.3,1,'x')))

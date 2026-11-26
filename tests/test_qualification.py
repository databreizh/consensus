import itertools,math,sys,unittest
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qualification import Pair,Instance,Planner,repair_outcome,worked_example,four_posts

# Independent reference: Bron-Kerbosch maximal cliques of compatibility graph.
def reference_outcome(instance,assignment):
    n=instance.vertices;adj=[set(range(n))-{i} for i in range(n)]
    for bit,pair in zip(assignment,instance.pairs):
        if bit:adj[pair.u].remove(pair.v);adj[pair.v].remove(pair.u)
    repairs=[]
    def bk(r,p,x):
        if not p and not x:repairs.append(r);return
        for v in list(sorted(p)):
            bk(r|{v},p&adj[v],x&adj[v]);p.remove(v);x.add(v)
    bk(set(),set(range(n)),set())
    vals={len(set(instance.violations)&r)>=instance.threshold for r in repairs}
    return (1 if False in vals else 0)|(2 if True in vals else 0)

class SemanticsTests(unittest.TestCase):
    def test_all_1024_full_graphs_against_independent_clique_oracle(self):
        inst=four_posts(63,1)
        for a in itertools.product([0,1],repeat=10):
            edges=tuple(sorted((p.u,p.v) for bit,p in zip(a,inst.pairs) if bit))
            self.assertEqual(repair_outcome(8,inst.violations,2,edges),reference_outcome(inst,a))

    def test_worked_example_expectations_by_explicit_full_assignments(self):
        inst=worked_example();plan=Planner(inst)
        def run(order,a):
            state=0;cost=0
            for i in order:
                # Independent stopping oracle enumerates all compatible completions.
                outcomes=set()
                for b in itertools.product([0,1],repeat=3):
                    if all(state//(3**j)%3 in (0,1+b[j]) for j in range(3)):
                        outcomes.add(reference_outcome(inst,b))
                if len(outcomes)==1:break
                cost+=inst.pairs[i].cost;state+=(1+a[i])*3**i
            return cost
        def expected(order):
            return sum(math.prod(p.probability if a[i] else 1-p.probability for i,p in enumerate(inst.pairs))*run(order,a) for a in itertools.product([0,1],repeat=3))
        self.assertAlmostEqual(expected((0,1,2)),3)
        self.assertAlmostEqual(expected((2,0,1)),2.05)
        self.assertAlmostEqual(plan.cost_function('random')(0),sum(expected(o) for o in itertools.permutations(range(3)))/6)
        self.assertAlmostEqual(plan.cost_function('greedy')(0),2.05)
        self.assertAlmostEqual(plan.cost_function('optimal')(0),2.05)

    def test_every_partial_state_against_explicit_completion_sets(self):
        inst=worked_example();p=Planner(inst)
        for state in range(27):
            outcomes={reference_outcome(inst,a) for a in itertools.product([0,1],repeat=3) if all(state//3**j%3 in (0,1+a[j]) for j in range(3))}
            self.assertEqual(p.determined(state),len(outcomes)==1)

    def test_optimal_worked_case_by_independent_adaptive_tree_search(self):
        inst=worked_example();assignments=list(itertools.product([0,1],repeat=3))
        weights={x:math.prod(p.probability if x[i] else 1-p.probability for i,p in enumerate(inst.pairs)) for x in assignments}
        def solve(observed):
            possible=[x for x in assignments if all(v is None or x[i]==v for i,v in enumerate(observed))]
            if len({reference_outcome(inst,x) for x in possible})==1:return 0.
            mass=sum(weights[x] for x in possible);costs=[]
            for i,v in enumerate(observed):
                if v is not None:continue
                q=inst.pairs[i].cost
                for value in [0,1]:
                    child=list(observed);child[i]=value
                    branchmass=sum(weights[x] for x in possible if x[i]==value)
                    q+=branchmass/mass*solve(tuple(child))
                costs.append(q)
            return min(costs)
        self.assertAlmostEqual(solve((None,None,None)),Planner(inst).cost_function('optimal')(0))

    def test_empty_candidate_universe_costs_zero(self):
        p=Planner(Instance(2,(0,1),(),2))
        self.assertTrue(p.determined(0))
        self.assertTrue(all(v==0 for v in p.expected_costs().values()))

    def test_one_cross_edge_can_make_count_certainly_false(self):
        i=Instance(2,(0,1),(Pair(0,1,.3,1,'x'),),2);p=Planner(i)
        self.assertEqual(p._outcomes(1),1<<2)
        self.assertEqual(p._outcomes(2),1<<1)
        self.assertFalse(p.determined(0))  # Both completions invariant, different truth values.

    def test_atomic_ratio_rule_against_all_orders(self):
        inst=Instance(4,(0,),tuple(Pair(0,i,p,c,str(i)) for i,p,c in [(1,.2,1),(2,.8,2),(3,.6,1)]),1)
        plan=Planner(inst)
        def cost(order):
            survive=1.;value=0.
            for i in order:value+=survive*inst.pairs[i].cost;survive*=1-inst.pairs[i].probability
            return value
        self.assertAlmostEqual(plan.cost_function('optimal')(0),min(cost(o) for o in itertools.permutations(range(3))))
        self.assertAlmostEqual(plan.cost_function('ratio')(0),plan.cost_function('optimal')(0))

    def test_all_topologies_optimal_dominates_other_stopping_policies(self):
        for t in range(64):
            for c in (1,3):
                p=Planner(four_posts(t,c));values=p.expected_costs()
                for k in ['random','ratio','greedy']:
                    self.assertLessEqual(values['optimal'],values[k]+1e-10)
                    self.assertLessEqual(values[k],values['assess_all']+1e-10)
                p.clear()

if __name__=='__main__':unittest.main()

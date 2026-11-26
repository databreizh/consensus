"""Generate NEW, explicitly seeded results; never fit the published table."""
import argparse,csv,json,platform,random,statistics,time
from pathlib import Path
from qualification import Planner,four_posts,worked_example,CROSS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--instances',type=int,default=300);ap.add_argument('--seed',type=int,default=20261126);ap.add_argument('--out',type=Path,default=Path('results/qualification'));args=ap.parse_args()
    if args.instances < 2:ap.error('--instances must be >= 2')
    args.out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed)
    topologies=[sum((rng.random()<.5)<<j for j in range(6)) for _ in range(args.instances)]
    (args.out/'instances.json').write_text(json.dumps({'seed':args.seed,'instances':[{'id':i,'cross_topology':t,'cross_pairs':[list(e) for j,e in enumerate(CROSS) if t>>j&1]} for i,t in enumerate(topologies)]},indent=2)+'\n')
    unique=sorted(set(topologies));lookup={};start=time.perf_counter()
    for cost in [1,3]:
        for t in unique:
            plan=Planner(four_posts(t,cost));lookup[cost,t]=plan.expected_costs();plan.clear()
        print('Computed %d distinct topologies at cross cost %d'%(len(unique),cost),flush=True)
    rows=[{'instance':i,'topology':t,'cross_cost':cost,**lookup[cost,t]} for cost in [1,3] for i,t in enumerate(topologies)]
    with (args.out/'per_instance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=[];deltas=[]
    for cost in [1,3]:
        cohort=[x for x in rows if x['cross_cost']==cost]
        summary.append({'cross_cost':cost,**{m:statistics.mean(x[m] for x in cohort) for m in ['assess_all','random','ratio','greedy','optimal']}})
        for a,b in [('greedy','optimal'),('ratio','greedy')]:
            values=[x[a]-x[b] for x in cohort];se=statistics.stdev(values)/len(values)**.5
            deltas.append({'cross_cost':cost,'difference':a+'-'+b,'mean':statistics.mean(values),'ci95_normal_halfwidth':1.96*se,'n':len(values)})
    plan=Planner(worked_example());example=plan.expected_costs();plan.clear()
    report={'status':'independent_reimplementation_not_original_table3_code','python':platform.python_version(),'seed':args.seed,'n':args.instances,'same_topologies_across_costs':True,'summary':summary,'paired_differences':deltas,'worked_example':example,'elapsed_seconds':time.perf_counter()-start}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    tab='\\begin{tabular}{lrrrrr}\nCross cost & All & Random & Ratio & Greedy & Optimal \\\\\n'
    for x in summary:tab+=str(x['cross_cost'])+' & '+' & '.join('%.2f'%x[m] for m in ['assess_all','random','ratio','greedy','optimal'])+' \\\\\n'
    tab+='\\end{tabular}\n'
    (args.out/'table3_recomputed.tex').write_text('% NEW seeded cohort; replace paper table only after review.\n'+tab)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

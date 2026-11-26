"""Portable offline reproduction entry point. Does not alter the manuscript."""
import argparse,csv,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def compare(a,b,path=''):
    errors=[]
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():errors.append(path+': different keys')
        for k in a.keys()&b.keys():errors+=compare(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):errors.append(path+': different lengths')
        for i,(x,y) in enumerate(zip(a,b)):errors+=compare(x,y,path+'/'+str(i))
    elif isinstance(a,(float,int)) and isinstance(b,(float,int)):
        if abs(a-b)>1e-10:errors.append(path+': %r != %r'%(a,b))
    elif a!=b:errors.append(path+': different values')
    return errors

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-dices',action='store_true');ap.add_argument('--out',type=Path,default=ROOT/'results/local_run');args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.time();commands=[]
    def run(cmd):
        commands.append(cmd)
        subprocess.run(cmd,cwd=ROOT,check=True)
    run([sys.executable,'-m','unittest','discover','-s','tests','-v'])
    run([sys.executable,'scripts/run_qualification.py','--out',str(out/'qualification')])
    comparison={'status':'skipped'}
    if not args.skip_dices:
        import numpy,pandas
        derived=ROOT/'data/derived'
        manifest=json.loads((ROOT/'data/manifest.json').read_text())
        for f in manifest['derived']:
            p=ROOT/f['path'];digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest!=f['sha256']:raise ValueError('Data hash mismatch: '+str(p))
        dest=out/'dices_results.json'
        run([sys.executable,'-I','originals/dices_measure.py',str(derived),str(dest)])
        actual=json.loads(dest.read_text());reference=json.loads((ROOT/'results/reference/dices_results.json').read_text());errors=compare(reference,actual)
        comparison={'status':'pass' if not errors else 'fail','exact_json_values_equal':reference==actual,'absolute_tolerance':1e-10,'differences':errors,'numpy':numpy.__version__,'pandas':pandas.__version__}
        if errors:
            (out/'dices_comparison.json').write_text(json.dumps(comparison,indent=2));raise AssertionError('DICES outputs differ; inspect dices_comparison.json')
    (out/'dices_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (out/'run_manifest.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'commands':commands,'elapsed_seconds':time.time()-start},indent=2)+'\n')
    print('PASS. Outputs:',out)
if __name__=='__main__':main()

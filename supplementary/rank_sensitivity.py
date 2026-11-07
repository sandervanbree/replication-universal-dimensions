import numpy as np, pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Rank-tolerance sensitivity from the saved PCA states.")
parser.add_argument('--pca-root', type=Path, default=REPO / 'data/pca')
parser.add_argument('--output', type=Path, default=REPO / 'outputs/report/rank_tolerance.csv')
parser.add_argument('--workers', type=int, default=2)
args = parser.parse_args()
root = args.pca_root
eps = np.finfo(np.float32).eps
def work(row):
    model,node,rank,nfeat = row
    with np.load(f'{root}/layers/{model}/node={node}/pca.npz') as z:
        ev = z['eigenvalues'].astype(np.float64); V = z['eigenvectors'].astype(np.float64); mu = z['mean'].astype(np.float64).ravel(); n=int(z['n_fit_presentations'])
    s2 = ev*(n-1)
    # power iteration for lambda_max of n mu mu^T + V diag(s2) V^T
    v = mu/np.linalg.norm(mu) if np.linalg.norm(mu)>0 else np.ones_like(mu)/np.sqrt(len(mu))
    lam=0
    for _ in range(200):
        w = n*mu*(mu@v) + V@(s2*(V.T@v))
        lam_new = np.linalg.norm(w); v = w/lam_new
        if abs(lam_new-lam) < 1e-10*lam_new: break
        lam = lam_new
    smax = np.sqrt(lam_new)
    s = np.sqrt(np.maximum(s2,0))
    big = max(n, nfeat)
    k_rep = int((s > big*eps*smax).sum())
    k_orig = int((s > max(72128, nfeat)*eps*smax).sum())
    return model,node,rank,k_rep,k_orig
if __name__=='__main__':
    m = pd.read_csv(root / 'layer_manifest.csv')
    rows = list(m[['model','node','rank','input_features']].itertuples(index=False, name=None))
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(work, rows, chunksize=4))
    out = pd.DataFrame(res, columns=['model','node','rank','k_rep_approx','k_orig_tol'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out[['rank','k_rep_approx','k_orig_tol']].sum())
    print('approx==rank fraction', (out['rank']==out.k_rep_approx).mean(), 'mean abs diff', (out['rank']-out.k_rep_approx).abs().mean())

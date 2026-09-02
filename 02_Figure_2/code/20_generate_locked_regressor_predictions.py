"""Generate untouched-test predictions for the locked no-hypothetical regressor."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from v4_common import SEEDS, build_regressor, load_primary_data

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--matrix-dir',type=Path,required=True); p.add_argument('--results',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--n-jobs',type=int,default=4); a=p.parse_args()
    X, labels, _ = load_primary_data(a.matrix_dir)
    manifest=pd.read_csv(a.results/'regressor_split_manifest.csv',keep_default_na=False)
    locked=pd.read_csv(a.results/'regressor_locked_parameters_by_run.csv',keep_default_na=False)
    pos=np.flatnonzero(pd.to_numeric(labels['has_tRNA']).to_numpy()==1)
    y=pd.to_numeric(labels['tRNA_count']).to_numpy(dtype=float)[pos]
    rows=[]
    for run,seed in enumerate(SEEDS,1):
        m=manifest.loc[(manifest.run.astype(int)==run)&manifest.split.eq('untouched_test')]
        dev=manifest.loc[(manifest.run.astype(int)==run)&manifest.split.isin(['development_inner_train','development_validation'])]
        tr=dev.positive_subset_row_index.to_numpy(dtype=int); te=m.positive_subset_row_index.to_numpy(dtype=int)
        model=build_regressor(json.loads(locked.loc[locked.run.astype(int).eq(run),'params'].iloc[0]),seed,a.n_jobs)
        model.fit(X[pos[tr]],y[tr]); pred=model.predict(X[pos[te]])
        for j,pr in zip(te,pred):
            row=labels.iloc[pos[j]]
            rows.append({'run':run,'seed':seed,'Sample_ID':row['Sample_ID'],'Assembly_ID':row['Assembly_ID'],'Replicon_ID':row['Replicon_ID'],'y_true_tRNA_count':float(y[j]),'y_pred_tRNA_count':float(pr)})
    pd.DataFrame(rows).to_csv(a.out,index=False); return 0
if __name__=='__main__': raise SystemExit(main())

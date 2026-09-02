from pathlib import Path
import argparse,json,numpy as np,pandas as pd
from scipy import sparse
from v4_common import build_classifier,continuous_score,classification_metrics
p=argparse.ArgumentParser(); p.add_argument('--run',type=int,required=True); p.add_argument('--matrix-dir',type=Path,required=True); p.add_argument('--taxonomy',type=Path,required=True); p.add_argument('--classifier-results',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--n-jobs',type=int,default=1); a=p.parse_args(); run=a.run; seed=41+run
X=sparse.load_npz(a.matrix_dir/'X_plasmids_by_codes_no_tRNA.npz'); s=pd.read_csv(a.matrix_dir/'sample_ids.csv'); l=pd.read_csv(a.matrix_dir/'derived_labels_rebuilt.csv'); t=pd.read_csv(a.taxonomy,keep_default_na=False)[['GCF_ID','phylum']].rename(columns={'GCF_ID':'Assembly_ID'}); m=s.merge(l[['Sample_ID','has_tRNA']],on='Sample_ID').merge(t,on='Assembly_ID',how='left'); m.phylum=m.phylum.replace('','Unclassified').fillna('Unclassified'); y=m.has_tRNA.astype(int).to_numpy(); man=pd.read_csv(a.classifier_results/'classifier_split_manifest.csv'); z=man[man.run.astype(int)==run]; tr=z[z.split.isin(['development_inner_train','development_validation'])].row_index.astype(int).to_numpy(); te=z[z.split.eq('untouched_test')].row_index.astype(int).to_numpy(); lock=pd.read_csv(a.classifier_results/'classifier_locked_parameters_by_run.csv'); par=json.loads(lock.loc[(lock.run==run)&(lock.model=='LightGBM'),'params'].iloc[0]); model=build_classifier('LightGBM',par,seed,y[tr],a.n_jobs); model.fit(X[tr],y[tr]); sc=continuous_score(model,X[te]); th=float(lock.loc[(lock.run==run)&(lock.model=='LightGBM'),'F1_threshold_from_validation'].iloc[0]); rows=[]
for ph in sorted(m.iloc[te].phylum.unique()):
 ix=np.flatnonzero(m.iloc[te].phylum.to_numpy()==ph); yy=y[te][ix]
 if yy.sum()==0 or len(np.unique(yy))<2: continue
 rows.append({'task':'classification','run':run,'seed':seed,'phylum':ph,'test_n':len(ix),'positive_n':int(yy.sum()),**classification_metrics(yy,sc[ix],th)})
pd.DataFrame(rows).to_csv(a.out,index=False)

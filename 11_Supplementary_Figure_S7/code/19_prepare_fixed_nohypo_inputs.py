"""Prepare the no-hypothetical matrix and fixed-parameter lock tables."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, precision_recall_curve
from v4_common import build_classifier, continuous_score

FIXED_CLASS = {"n_estimators":600,"learning_rate":0.036575225398754393,"num_leaves":127,"max_depth":-1,"min_child_samples":10,"subsample":0.8912973651692675,"subsample_freq":1,"colsample_bytree":0.6750465967387895,"reg_alpha":0.007897750060475092,"reg_lambda":0.0033217354983947392}
FIXED_REG = {"n_estimators":300,"learning_rate":0.0923761728181156,"num_leaves":127,"max_depth":16,"min_child_samples":5,"subsample":0.9471644127982595,"subsample_freq":1,"colsample_bytree":0.616145612015363,"reg_alpha":0.0005798594476883552,"reg_lambda":0.0001823901176321363}
def threshold(y, score):
    p,r,t=precision_recall_curve(y,score); f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12); return float(t[int(np.nanargmax(f))])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--classifier-results',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    X=sparse.load_npz(a.source/'X_plasmids_by_codes_no_tRNA.npz').tocsr().astype(np.float32); names=np.load(a.source/'feature_names_no_tRNA.npy',allow_pickle=True).astype(str); mapping=pd.read_csv(a.source/'feature_code_mapping.csv',keep_default_na=False); labels=pd.read_csv(a.source/'derived_labels_rebuilt.csv',keep_default_na=False); samples=pd.read_csv(a.source/'sample_ids.csv',keep_default_na=False)
    adfu=np.flatnonzero(names=='ADFU');
    if len(adfu)!=1: raise ValueError('Expected exactly one ADFU feature')
    keep=np.ones(X.shape[1],dtype=bool); keep[adfu[0]]=False
    sparse.save_npz(a.out/'X_plasmids_by_codes_no_tRNA.npz',X[:,keep]); np.save(a.out/'feature_names_no_tRNA.npy',names[keep]); mapping.loc[mapping.Code.ne('ADFU')].to_csv(a.out/'feature_code_mapping.csv',index=False); labels.to_csv(a.out/'derived_labels_rebuilt.csv',index=False); samples.to_csv(a.out/'sample_ids.csv',index=False)
    splits=pd.read_csv(a.classifier_results/'classifier_split_manifest.csv',keep_default_na=False); y=labels.has_tRNA.astype(int).to_numpy(); rows=[]
    for run,seed in zip((1,2,3),(42,43,44)):
        inner=splits[(splits.run.astype(int)==run)&splits.split.eq('development_inner_train')].row_index.to_numpy(int); val=splits[(splits.run.astype(int)==run)&splits.split.eq('development_validation')].row_index.to_numpy(int)
        model=build_classifier('LightGBM',FIXED_CLASS,seed,y[inner],4); model.fit(X[inner][:,keep],y[inner]); score=continuous_score(model,X[val][:,keep]); rows.append({'run':run,'seed':seed,'model':'LightGBM','selected_candidate_id':'fixed_latest','selected_validation_AUPRC':float(average_precision_score(y[val],score)),'F1_threshold_from_validation':threshold(y[val],score),'params':json.dumps(FIXED_CLASS,sort_keys=True)})
    pd.DataFrame(rows).to_csv(a.out/'classifier_locked_parameters_by_run.csv',index=False)
    reg=pd.DataFrame([{'run':r,'seed':s,'selected_candidate_id':'fixed_latest','selected_validation_RMSE':'','params':json.dumps(FIXED_REG,sort_keys=True)} for r,s in zip((1,2,3),(42,43,44))]); reg.to_csv(a.out/'regressor_locked_parameters_by_run.csv',index=False)
    for src in ('classifier_split_manifest.csv','classifier_run_metadata.json'):
        (a.out/src).write_text((a.classifier_results/src).read_text(encoding='utf-8'),encoding='utf-8')
    (a.out/'fixed_parameter_metadata.json').write_text(json.dumps({'classification':FIXED_CLASS,'regression':FIXED_REG,'hypothetical_feature_removed':'ADFU','split_policy':'existing three 80:20 split manifests; validation-derived F1 thresholds'},indent=2),encoding='utf-8')
if __name__=='__main__': main()

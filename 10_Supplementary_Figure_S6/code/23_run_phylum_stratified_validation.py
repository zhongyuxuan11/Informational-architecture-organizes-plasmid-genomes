"""Validate locked models within phyla by filtering the untouched test rows only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from v4_common import SEEDS, build_classifier, build_regressor, classification_metrics, continuous_score, regression_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--matrix-dir',type=Path,required=True); p.add_argument('--taxonomy',type=Path,required=True); p.add_argument('--classifier-results',type=Path,required=True); p.add_argument('--regressor-results',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True); p.add_argument('--n-jobs',type=int,default=4); a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=False)
    from scipy import sparse
    X=sparse.load_npz(a.matrix_dir/'X_plasmids_by_codes_no_tRNA.npz'); samples=pd.read_csv(a.matrix_dir/'sample_ids.csv'); labels=pd.read_csv(a.matrix_dir/'derived_labels_rebuilt.csv'); tax=pd.read_csv(a.taxonomy,keep_default_na=False)[['GCF_ID','phylum']].rename(columns={'GCF_ID':'Assembly_ID'}); meta=samples.merge(labels[['Sample_ID','has_tRNA','tRNA_count']],on='Sample_ID',validate='one_to_one').merge(tax,on='Assembly_ID',how='left',validate='many_to_one'); meta['phylum']=meta.phylum.replace('','Unclassified').fillna('Unclassified'); y=meta.has_tRNA.astype(int).to_numpy(); count=meta.tRNA_count.astype(float).to_numpy()
    cm=pd.read_csv(a.classifier_results/'classifier_split_manifest.csv'); cp=pd.read_csv(a.classifier_results/'classifier_locked_parameters_by_run.csv'); rm=pd.read_csv(a.regressor_results/'regressor_split_manifest.csv'); rp=pd.read_csv(a.regressor_results/'regressor_locked_parameters_by_run.csv'); rows=[]; excluded=[]
    for run,seed in enumerate(SEEDS,1):
        cmr=cm[cm.run.astype(int)==run]; tr=cmr[cmr.split.isin(['development_inner_train','development_validation'])].row_index.astype(int).to_numpy(); te=cmr[cmr.split.eq('untouched_test')].row_index.astype(int).to_numpy(); model=build_classifier('LightGBM',json.loads(cp.loc[(cp.run==run)&(cp.model=='LightGBM'),'params'].iloc[0]),seed,y[tr],a.n_jobs); model.fit(X[tr],y[tr]); score=continuous_score(model,X[te]); threshold=float(cp.loc[(cp.run==run)&(cp.model=='LightGBM'),'F1_threshold_from_validation'].iloc[0]);
        for ph in sorted(meta.iloc[te].phylum.unique()):
            ix=np.flatnonzero(meta.iloc[te].phylum.to_numpy()==ph); yy=y[te][ix]
            if yy.sum()==0 or len(np.unique(yy))<2: excluded.append({'task':'classification','run':run,'phylum':ph,'reason':'no positive or no negative test rows'}); continue
            rows.append({'task':'classification','run':run,'seed':seed,'phylum':ph,'test_n':len(ix),'positive_n':int(yy.sum()),**classification_metrics(yy,score[ix],threshold)})
        rmr=rm[rm.run.astype(int)==run]; rtr=rmr[rmr.split.isin(['development_inner_train','development_validation'])].positive_subset_row_index.astype(int).to_numpy(); rte=rmr[rmr.split.eq('untouched_test')].positive_subset_row_index.astype(int).to_numpy(); pos=np.flatnonzero(y==1); reg=build_regressor(json.loads(rp.loc[rp.run==run,'params'].iloc[0]),seed,a.n_jobs); reg.fit(X[pos[rtr]],count[pos[rtr]]); pred=reg.predict(X[pos[rte]]); test_meta=meta.iloc[pos[rte]].reset_index(drop=True); yy=count[pos[rte]]
        for ph in sorted(test_meta.phylum.unique()):
            ix=np.flatnonzero(test_meta.phylum.to_numpy()==ph); yt=yy[ix]
            if len(ix)<1: excluded.append({'task':'regression','run':run,'phylum':ph,'reason':'no positive test rows'}); continue
            if len(ix)>1 and np.std(yt)>0 and np.std(pred[ix])>0:
                rho=float(pd.Series(yt).corr(pd.Series(pred[ix]),method='spearman'))
            else: rho=float('nan')
            rows.append({'task':'regression_tRNA_positive_only','run':run,'seed':seed,'phylum':ph,'test_n':len(ix),'positive_n':len(ix),'R2':float(r2_score(yt,pred[ix])),'RMSE':float(mean_squared_error(yt,pred[ix])**0.5),'Spearman_rho':rho,'AUPRC':float('nan'),'AUROC':float('nan'),'F1':float('nan')})
    detail=pd.DataFrame(rows); summary=detail.groupby(['task','phylum'],as_index=False).agg(run_n=('run','nunique'),test_n_mean=('test_n','mean'),positive_n_mean=('positive_n','mean'),AUPRC_mean=('AUPRC','mean'),AUPRC_SD=('AUPRC','std'),R2_mean=('R2','mean'),R2_SD=('R2','std'),RMSE_mean=('RMSE','mean'),RMSE_SD=('RMSE','std'),Spearman_rho_mean=('Spearman_rho','mean'),Spearman_rho_SD=('Spearman_rho','std')); detail.to_csv(a.out_dir/'phylum_stratified_metrics_each_run.csv',index=False); summary.to_csv(a.out_dir/'phylum_stratified_metrics_summary.csv',index=False); pd.DataFrame(excluded).to_csv(a.out_dir/'phylum_stratified_exclusions.csv',index=False); (a.out_dir/'run_metadata.json').write_text(json.dumps({'evaluation':'filter untouched test rows within each phylum; no phylum-specific retraining','positive_case_rule':'classification phyla retain both positive and negative test rows; regression uses positive plasmids only','seeds':list(SEEDS)},indent=2),encoding='utf-8')
if __name__=='__main__': main()

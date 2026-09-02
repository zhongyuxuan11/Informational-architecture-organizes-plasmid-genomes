from pathlib import Path
import numpy as np, pandas as pd, json
from sklearn.metrics import r2_score, mean_squared_error
p=Path('ml_v4_unified_20260820/results/phylum_stratified_validation')
tax=pd.read_csv('organized_update_20260803/supplementary_tables/00_preprocessing/gcf_with_full_taxonomy.csv',keep_default_na=False)[['GCF_ID','phylum']].rename(columns={'GCF_ID':'Assembly_ID'})
pred=pd.read_csv('ml_v4_unified_20260820/results/regressor_locked/regressor_test_predictions.csv').merge(tax,on='Assembly_ID',how='left')
pred['phylum']=pred['phylum'].replace('','Unclassified').fillna('Unclassified'); rows=[]
for (run,ph),g in pred.groupby(['run','phylum']):
    y=g.y_true_tRNA_count.to_numpy(float); yp=g.y_pred_tRNA_count.to_numpy(float)
    rows.append({'task':'regression_tRNA_positive_only','run':int(run),'seed':int(g.seed.iloc[0]),'phylum':ph,'test_n':len(g),'positive_n':len(g),'AUPRC':np.nan,'AUROC':np.nan,'F1':np.nan,'R2':float(r2_score(y,yp)),'RMSE':float(mean_squared_error(y,yp)**0.5),'Spearman_rho':np.nan})
for f in sorted(p.glob('class_run*.csv')): rows.extend(pd.read_csv(f).to_dict('records'))
d=pd.DataFrame(rows)
s=d.groupby(['task','phylum'],as_index=False).agg(run_n=('run','nunique'),test_n_mean=('test_n','mean'),positive_n_mean=('positive_n','mean'),AUPRC_mean=('AUPRC','mean'),AUPRC_SD=('AUPRC','std'),R2_mean=('R2','mean'),R2_SD=('R2','std'),RMSE_mean=('RMSE','mean'),RMSE_SD=('RMSE','std'))
d.to_csv(p/'phylum_stratified_metrics_each_run.csv',index=False); s.to_csv(p/'phylum_stratified_metrics_summary.csv',index=False)
(p/'run_metadata.json').write_text(json.dumps({'evaluation':'filter untouched test rows within each phylum; no phylum-specific retraining','classification_rule':'retain phyla with positive and negative test rows','regression_rule':'tRNA-positive test rows; R2 retained for constant targets'},indent=2),encoding='utf-8')

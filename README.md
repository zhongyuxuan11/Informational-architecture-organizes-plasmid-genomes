# Plasmid tRNA analysis

This directory contains the code and compact tables used for Figures 1-4 and Supplementary Figures S1-S16 in manuscript. Each numbered directory is self-contained for the corresponding figure. 

## Analysis order

1. Download complete bacterial and archaeal assemblies from NCBI RefSeq (original cohort: releases before June 2025; temporal cohort: June 2025-August 2026). Retain PGAP-annotated complete, non-atypical assemblies and parse GBFF replicons, products and tRNAs.
2. Join the RefSeq assembly summary and NCBI Taxonomy lineages. Annotate ARGs with CARD/RGI, validate tRNA-bearing replicons with PlasFlow, and annotate proteins with eggNOG-mapper for COG/NOG assignments.
3. Build the plasmid-by-product sparse count matrix, labels and fixed feature vocabulary. Remove every tRNA-derived predictor.
4. Run the three independent 80:20 splits (seeds 42, 43 and 44). Tune 50 candidates only within each training partition, refit the selected model on the complete development partition, and evaluate once on its held-out test partition.
5. Run robustness, type-specific, CCI and central-dogma analyses. Temporal validation trains one classifier and one regressor once on the complete historical cohort and evaluates each once on the non-overlapping later cohort.
6. Run the figure-specific plotting entry points. Plotting requires Arial; all text is 7 pt, graphical marks are 0.7 pt, and x/y axes are 1 pt.

## External resources

- NCBI RefSeq/PGAP and Assembly Summary: <https://www.ncbi.nlm.nih.gov/refseq/> and <https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/>
- NCBI Taxonomy: <https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/>
- CARD/RGI: <https://card.mcmaster.ca/>
- PlasFlow: <https://github.com/smaegol/PlasFlow>
- eggNOG-mapper/COG-NOG: <https://eggnog-mapper.embl.de/>
- KEGG BRITE keyword workbook: manually downloaded from <https://www.genome.jp/kegg/brite.html>

Exact database releases and cohort dates are those stated in manuscript V6. Large inputs are not duplicated in GitHub; see each `EXTERNAL_LARGE_FILES.tsv`.

## Software

Python 3.10 packages: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`, `Pillow`, `biopython`, `ete3` and `joblib`. R plotting, where used, requires `tidyverse`, `ape`, `ggtree`, `ggtreeExtra`, `ggnewscale` and `showtext`. External command-line software comprises NCBI Datasets/FTP tools, CARD RGI, PlasFlow and eggNOG-mapper.

## Reproducibility rules

Held-out test data never enter feature selection, hyperparameter selection or threshold calibration. AUPRC is the classification target metric; regression reports R2 on tRNA-positive plasmids. Raw replicate points are retained for three-split analyses; the temporal figure contains single estimates and no mean or SD. CCI statistics use raw CDS-level values, while display transformations are applied only where explicitly stated in the manuscript.

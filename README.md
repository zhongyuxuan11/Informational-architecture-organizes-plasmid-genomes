# Plasmid tRNA analysis and figure package

This directory contains the code and compact tables used for Figures 1-4 and Supplementary Figures S1-S16 in manuscript V6. Each numbered directory is self-contained for the corresponding figure. Final images are stored in the parallel `GitHub_upload_V5_figures_20260831_figures_only` directory.

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

## Large CCI inputs

- `Fig3_CCI_latest_20260831/inputs_large/3_CDS_64_Codon_Counts.csv` (4,050,431,902 bytes).
- `Fig3_CCI_latest_20260831/tables/01_gene_level_pooled_CCI_all_proteins_triad_genomes.csv` (1,965,536,229 bytes). This is the required external CDS-level table for Figure 3 and S13.

## Software

Python 3.10 packages: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`, `Pillow`, `biopython`, `ete3` and `joblib`. R plotting, where used, requires `tidyverse`, `ape`, `ggtree`, `ggtreeExtra`, `ggnewscale` and `showtext`. External command-line software comprises NCBI Datasets/FTP tools, CARD RGI, PlasFlow and eggNOG-mapper.

## Figure map

Directories `01_Figure_1`-`04_Figure_4` map to main Figures 1-4; directories `05_Supplementary_Figure_S1`-`20_Supplementary_Figure_S16` map directly to S1-S16. Analysis scripts generate the included tables; scripts containing `plot`, `redraw` or `svg` are rendering entry points. S4 and S8 have separate renderers. S15 uses `02_plot_s15_fourteen_ring_landscape.py`, with rings fixed in the V6 caption order (4 replication, 4 transcription and 6 translation subcategories).

In the parallel figure directory, each `*_composite.png` is the exact raster embedded in manuscript V6. Files named `*_generated.png`, `*_generated.svg` or `*_source.svg` are reproducible code outputs. Generated figures must match the composite in data, visible labels, panel order and interpretation; the composite preserves the final Word geometry and resolution.

## Reproducibility rules

Held-out test data never enter feature selection, hyperparameter selection or threshold calibration. AUPRC is the classification target metric; regression reports R2 on tRNA-positive plasmids. Raw replicate points are retained for three-split analyses; the temporal figure contains single estimates and no mean or SD. CCI statistics use raw CDS-level values, while display transformations are applied only where explicitly stated in the manuscript.

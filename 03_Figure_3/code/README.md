# Figure 3 CCI package code

## Code order

1. `01_rebuild_trna_supply_table.py`: rebuild `inputs_small/trna_supply_table_rebuilt_from_all.csv` from an external GBFF root.
2. `fig3b_nohypo_triad_pooled_cci.py`: recompute CDS-level pooled-supply CCI for Fig3B using package-local inputs; hypothetical proteins are retained.
3. `02_generate_cci_tables.py`: regenerate package-local downstream tables from `tables/01_gene_level_pooled_CCI_all_proteins_triad_genomes.csv`.
4. `03_plot_fig3_figS7.py`: render SVG files from package-local `tables/` into package-local `fig/`.
5. `fig3_downstream_core.py`: shared strict CCI aggregation/statistics functions.

## Current panel correspondence

| Panel | Figure file | Principal source table |
|---|---|---|
| Fig3B | `Fig3B_gene_level_CCI_chr_trnap_trnam.svg` | `FigS11C_CDS_level_CCI_group_stats.csv`; `FigS11C_CDS_pairwise_Mann_Whitney_BH.csv` |
| Fig3B 100 kb strata | `Fig3B_gene_level_CCI_chr_trnap_trnam_by_100kb.svg` | `02_Fig3B_gene_level_CCI_group_stats_no_hypothetical_triad.csv` |
| Fig3C | `Fig3C_top10_product_CCI_violin_box.svg` | `Fig3C_top10_product_gene_CCI_distribution.csv` |
| Fig3D | `Fig3D_COG_AUPRC_vs_median_CCI.svg` | `Fig3D_COG_performance_vs_product_median_CCI.csv` |
| Fig3E | `Fig3E_phylum_CCI.svg` | `Fig3E_phylum_plasmid_level_CCI_plasmid_n_ge10.csv`; `Fig3E_plasmid_level_CCI_for_phylum.csv` |
| S12 | `S12_COG_R2_vs_median_CCI.svg` | `Fig3D_COG_performance_vs_product_median_CCI.csv` |

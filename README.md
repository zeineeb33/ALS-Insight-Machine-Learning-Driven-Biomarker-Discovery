# ALS-Insight-Machine-Learning-Driven-Biomarker-Discovery



This Bioinformatics Project (L3 INFO, Paris-Saclay, 2025) targets the discovery of ALS (Amyotrophic Lateral Sclerosis) biomarkers using RNA-Seq data from post-mortem human cortex samples. We implemented an object-oriented Python pipeline to:

1. **Preprocess & Integrate Data**  
   - Load and QC raw gene-count files and sample metadata (MINiML XML).  
   - Consolidate counts into a single samples×genes matrix and annotations table.

2. **Descriptive Analysis**  
   - Compute per-sample and per-gene statistics (mean, median, SD).  
   - Visualize group distributions with bar charts, histograms, box plots and heatmaps.

3. **Dimensionality Reduction**  
   - Standardize data and perform PCA to assess variance structure and detect outliers.  
   - (Optional) Apply t-SNE and UMAP for complementary low-dimensional embeddings.

4. **Differential Expression**  
   - Use PyDESeq2 to identify genes significantly up- or downregulated between ALS and control groups, controlling the false discovery rate.  
   - Generate volcano plots to highlight top candidates.

5. **Multivariate Feature Selection**  
   - Train regularized logistic regression (Elastic-Net) and XGBoost classifiers on normalized counts.  
   - Rank genes by model coefficients and feature importances to refine biomarker candidates.

Finally, we compiled a ranked list of the top 100 candidate genes and documented all code, analyses, and results for submission and presentation.  

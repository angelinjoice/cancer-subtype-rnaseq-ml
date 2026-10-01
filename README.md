# Breast Cancer Subtype Classification via RNA-Seq & Machine Learning

A machine learning and biomarker discovery project designed to classify breast cancer molecular subtypes (PAM50 classifications) using bulk RNA-Seq transcriptomic expression profiles.

## Summary

Transcriptomic profiling via bulk RNA-Sequencing yields high-dimensional expression matrices that reflect underlying tumor biology. This project implements a machine learning and feature discovery workflow applied to The Cancer Genome Atlas Breast Cancer (**TCGA-BRCA**) dataset to classify cancer subtypes and identify key biomarker genes.

## Project Workflow & Implementation

### Phase 1: Environment & Project Setup
The workspace is configured with dedicated subfolders for data management, interactive notebooks, source modules, visual outputs, and serialized models. A virtual environment manages core data science and bioinformatic dependencies (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`).

### Phase 2: Data Ingestion & Preprocessing
RNA-Seq count tables and clinical phenotype records from TCGA-BRCA are loaded into memory. Matrix dimensions are oriented so that samples form rows and genes form columns, with duplicate gene symbols resolved by retaining maximum expression levels. An inner join aligns sample IDs between expression profiles and PAM50 subtype labels.

### Phase 3: Variance Stabilization & Leakage-Free Data Splitting
Raw counts undergo a log2(countS + 1) transformation to stabilize variance across high-expression ranges. Categorical PAM50 labels are mapped to numerical targets, and samples are split into an 80% training set and a 20% test set using stratified sampling to preserve class distributions. To avoid data leakage:
* Uninformative, low-variance genes are filtered out using a `VarianceThreshold` fit solely on training data.
* Expression values are scaled with `StandardScaler` using parameters computed exclusively from the training set.

### Phase 4: Exploratory Analysis & Subtype Visualization
Unsupervised learning methods evaluate natural groupings in the data before model training. Principal Component Analysis (PCA) maps sample relationships in lower-dimensional space, while a hierarchical bi-clustered heatmap highlights expression patterns across top dynamic features to confirm distinct molecular subtype signatures.

### Phase 5: Supervised Model Training & Hyperparameter Tuning
Multiple classifier architectures (**Random Forest**, **Support Vector Machines**, and **XGBoost**) are trained and compared. Models undergo Stratified 5-Fold Cross-Validation evaluated on Macro F1-Score to prevent bias toward dominant subtypes (like *Luminal A*). Parameter configurations (such as tree depth and regularization terms) are optimized using `GridSearchCV`.

### Phase 6: Model Evaluation & Biomarker Discovery
The final optimized classifier is evaluated against the held-out 20% test dataset. Precision, recall, macro F1-score, and confusion matrices measure classification accuracy on unseen samples. Feature importances are extracted from the trained model to rank and identify top biomarker genes driving subtype separation (e.g., *ROPN1*, *AGR3*, *VGLL1*).

### Phase 7: Model Serialization & Modular Refactoring
Fitted transformers (`StandardScaler`, `VarianceThreshold`, `LabelEncoder`) and trained model weights are saved as `.pkl` files using `joblib` to enable downstream inference on unlabelled expression matrices. Preprocessing, training, and evaluation workflows are refactored into modular Python scripts.

## Quantitative Results & Evaluation

### Test Set Performance Overview
* **Overall Accuracy:** 82.81% (159 out of 192 test samples correctly classified)
* **Macro F1-Score:** 0.8119
* **Weighted F1-Score:** 0.8184

### Per-Class Performance Breakdown
* **Basal:** Precision: 0.9655 | Recall: 0.9655 | F1-Score: 0.9655 (Support: 29)
* **Her2:** Precision: 1.0000 | Recall: 0.6154 | F1-Score: 0.7619 (Support: 13)
* **LumA:** Precision: 0.7636 | Recall: 0.9655 | F1-Score: 0.8528 (Support: 87)
* **LumB:** Precision: 0.8333 | Recall: 0.5128 | F1-Score: 0.6349 (Support: 39)
* **Normal:** Precision: 0.9048 | Recall: 0.7917 | F1-Score: 0.8444 (Support: 24)

### Exploratory Data & Dimensionality Findings
* **PCA Separation:** Clear spatial separation is observed for Basal (negative PC2) and Normal-like clusters. Luminal A and Luminal B subtypes exhibit significant spatial overlap along PC1.
* **Hierarchical Heatmap:** Bi-clustering demonstrates strong co-expression blocks across dynamic features, validating distinct subtype transcriptomic profiles.

### Top Biomarker Genes Identified
1. **ROPN1** (Importance: ~0.0165)
2. **AGR3** (Importance: ~0.0148)
3. **LOC145837** (Importance: ~0.0118)
4. **VGLL1** (Importance: ~0.0110)
5. **WNT6** (Importance: ~0.0107)
6. **GABRP** (Importance: ~0.0107)

## Key Conclusions

* **High Performance on Distinct Subtypes:** The model easily identifies **Basal** (96.55% F1) and **Normal** (84.44% F1) subtypes because of their unique gene expression patterns.
* **Luminal Subtype Overlap:** The main misclassifications happen between **Luminal A** and **Luminal B** tumors due to their similar, overlapping biological features.
* **Biologically Valid Biomarkers:** Key top genes identified (like *AGR3*, *VGLL1*, and *GABRP*) match known breast cancer markers, showing the model learns real biology.

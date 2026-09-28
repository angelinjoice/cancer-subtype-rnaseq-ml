import argparse
import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="TCGA Breast Cancer Subtype ML Training Pipeline"
    )
    parser.add_argument(
        "--expr",
        type=str,
        default=None,
        help="Path to raw gene expression matrix file",
    )
    parser.add_argument(
        "--clinical",
        type=str,
        default=None,
        help="Path to raw clinical metadata file",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=500,
        help="Number of top features to select via ANOVA F-test (default: 500)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="svm",
        choices=["svm", "rf", "xgboost"],
        help="Model architecture to train (default: svm)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained artifacts (default: models)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save metrics and biomarker CSVs (default: results)",
    )
    return parser.parse_args()


def load_raw_data(expr_path, clinical_path):
    """Auto-detects and loads raw expression and clinical datasets."""
    if expr_path is None or not os.path.exists(expr_path):
        expr_files = glob.glob("data/raw/*HiSeqV2*") or glob.glob(
            "data/raw/*counts*"
        )
        if not expr_files:
            raise FileNotFoundError(
                "Could not locate expression file in data/raw/"
            )
        expr_path = expr_files[0]

    if clinical_path is None or not os.path.exists(clinical_path):
        clinical_files = glob.glob("data/raw/*clinicalMatrix*")
        if not clinical_files:
            raise FileNotFoundError(
                "Could not locate clinical file in data/raw/"
            )
        clinical_path = clinical_files[0]

    print(f"Loading Expression: {os.path.basename(expr_path)}")
    print(f"Loading Clinical:   {os.path.basename(clinical_path)}")

    expr_kwargs = (
        {"compression": "gzip"} if expr_path.endswith(".gz") else {}
    )
    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0, **expr_kwargs)

    clin_kwargs = (
        {"compression": "gzip"} if clinical_path.endswith(".gz") else {}
    )
    clinical_df = pd.read_csv(
        clinical_path, sep="\t", index_col=0, **clin_kwargs
    )

    # Force orientation: Samples as rows, Genes as columns
    if expr_df.shape[0] > expr_df.shape[1]:
        expr_df = expr_df.T

    # Aggregate duplicate gene columns
    expr_df = expr_df.T.groupby(level=0).max().T

    return expr_df, clinical_df


def preprocess_data(expr_df, clinical_df):
    """Aligns samples, extracts PAM50 labels, and filters unclassified samples."""
    common_samples = expr_df.index.intersection(clinical_df.index)
    X_sub = expr_df.loc[common_samples]
    clinical_sub = clinical_df.loc[common_samples]

    target_col = None
    for col in ["PAM50Call_RNAseq", "subm_PAM50", "PAM50", "PAM50_mRNA"]:
        if col in clinical_sub.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            "Could not identify a valid PAM50 column in clinical metadata."
        )

    y_sub = clinical_sub[target_col]
    valid_mask = y_sub.notna() & (~y_sub.isin(["", "null", "Not Evaluated", "NA"]))

    X = X_sub[valid_mask]
    y = y_sub[valid_mask]

    # Apply log2 variance stabilization if using raw counts
    if X.max().max() > 25:
        print("Applying log2(x + 1) variance stabilization...")
        X = np.log2(X + 1)

    return X, y


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load and Preprocess Data
    expr_df, clinical_df = load_raw_data(args.expr, args.clinical)
    X, y = preprocess_data(expr_df, clinical_df)

    # 2. Encode Target Subtypes
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Feature Selection (ANOVA F-test)
    k = min(args.n_features, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y_encoded)
    selected_genes = X.columns[selector.get_support()].tolist()

    # 4. Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

# Export processed datasets to data/processed
    os.makedirs('data/processed', exist_ok=True)
    pd.DataFrame(X_scaled, columns=selected_genes).to_csv('data/processed/X_processed.csv', index=False)
    pd.Series(y_encoded, name='Subtype').to_csv('data/processed/y_processed.csv', index=False)
    print("Processed datasets saved to 'data/processed/'")

    # 5. Initialize Model
    if args.model_type == "svm":
        model = SVC(kernel="linear", C=1.0, random_state=42)
    elif args.model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.model_type == "xgboost":
        model = XGBClassifier(eval_metric="mlogloss", random_state=42)

    # 6. Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_scaled, y_encoded, cv=cv, scoring="f1_macro"
    )

    print(f"\n=== Model Training Complete ({args.model_type.upper()}) ===")
    print(
        f"5-Fold Cross-Validation Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
    )

    # 7. Fit Final Model on All Preprocessed Data
    model.fit(X_scaled, y_encoded)

    # 8. Save Model Artifacts
    joblib.dump(model, os.path.join(args.output_dir, "pam50_model.pkl"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(
        selector, os.path.join(args.output_dir, "feature_selector.pkl")
    )
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.pkl"))

    # Save Selected Biomarker Gene List
    pd.DataFrame({"Gene": selected_genes}).to_csv(
        os.path.join(args.results_dir, "selected_biomarkers.csv"), index=False
    )

    print(
        f"\nArtifacts saved to '{args.output_dir}/' and biomarker list saved to '{args.results_dir}/selected_biomarkers.csv'"
    )


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import ttest_rel, t
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words
from project_name.models.xgb_model import XGBoostClassifier
from project_name.models.logistic_regression_model import LogisticRegressionClassifier


def preprocess_features(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    parse_words_dataset(dataset)
    X = embedding_words(dataset)

    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    y = dataset['sa'].map(label_mapping).astype(np.int32).to_numpy()

    return X, y


def run_kfold_evaluation(dataset: pd.DataFrame, k: int = 5, metric: str = "f1"):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    X_all, y_all = preprocess_features(dataset)

    scores_A, scores_B = [], []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_all, y_all)):
        print(f"\n=== Fold {fold_idx + 1} ===")

        # Outer split
        X_train_full, X_test = X_all[train_index], X_all[test_index]
        y_train_full, y_test = y_all[train_index], y_all[test_index]

        # Inner split (for XGBoost validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=fold_idx
        )

        # Initialize fresh model instances per fold
        model_A = LogisticRegressionClassifier()
        model_B = XGBoostClassifier()

        # Train
        model_A.fit(X_train_full, y_train_full)  # Logistic regression sees all training data
        model_B.fit(X_train, y_train, validation_data=(X_val, y_val))  # XGBoost uses inner val

        # Evaluate
        metrics_A = model_A.evaluate_metrics(X_test, y_test)
        metrics_B = model_B.evaluate_metrics(X_test, y_test)

        print("LogReg F1:", metrics_A[metric])
        print("XGBoost F1:", metrics_B[metric])

        scores_A.append(metrics_A[metric])
        scores_B.append(metrics_B[metric])

    return np.array(scores_A), np.array(scores_B)


def main():
    dataset = pd.read_csv('project_name/data/sa_spaeng_train.csv')
    
    print("Running K-Fold Comparison...")
    scores_logreg, scores_xgb = run_kfold_evaluation(dataset, k=5, metric="f1")

    print("\n=== Cross-Validation Summary ===")
    print(f"Logistic Regression Mean F1: {np.mean(scores_logreg):.4f} ± {np.std(scores_logreg):.4f}")
    print(f"XGBoost           Mean F1: {np.mean(scores_xgb):.4f} ± {np.std(scores_xgb):.4f}")

    # Paired t-test
    t_stat, p_val = ttest_rel(scores_logreg, scores_xgb)
    print(f"\nPaired t-test result: t = {t_stat:.4f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("=> The difference is statistically significant.")
    else:
        print("=> No statistically significant difference.")

    # Compute 95% Confidence Interval for the difference in F1 scores
    diff = scores_xgb - scores_logreg
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    sem_diff = std_diff / np.sqrt(n)

    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, df=n-1)

    ci_lower = mean_diff - t_crit * sem_diff
    ci_upper = mean_diff + t_crit * sem_diff

    print(f"\n95% Confidence Interval for the difference in mean F1 scores: ({ci_lower:.4f}, {ci_upper:.4f})")

if __name__ == "__main__":
    main()

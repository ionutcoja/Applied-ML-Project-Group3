import pandas as pd
import joblib
from train_model import preprocess_features
from sklearn.metrics import f1_score 
import numpy as np


def bootstrap_test(models, X_test, y_test, metric_func, n_bootstrap=1000):
    """
    Perform bootstrap statistical test comparing two models on test data.
    
    Args:
        models: list of two trained models [baseline_model, advanced_model]
        X_test: features of test set (numpy array)
        y_test: true labels of test set (numpy array)
        metric_func: function(y_true, y_pred) -> float, e.g. f1_score
        n_bootstrap: number of bootstrap iterations
    
    Prints:
        Mean difference and 95% confidence interval of the difference.
        Whether the difference is statistically significant.
    """
    baseline_model, advanced_model = models
    n = len(y_test)
    diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_test[indices]

        y_pred_baseline = baseline_model.predict(X_test[indices])
        y_pred_advanced = advanced_model.predict(X_test[indices])

        score_baseline = metric_func(y_true_sample, y_pred_baseline, average='weighted')
        score_advanced = metric_func(y_true_sample, y_pred_advanced, average='weighted')

        diffs.append(score_advanced - score_baseline)

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    print(f"\nBootstrap mean difference (Advanced - Baseline): {mean_diff:.4f}")
    print(f"95% Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

    if ci_lower > 0 or ci_upper < 0:
        print("=> Statistically significant difference between models.")
    else:
        print("=> No statistically significant difference between models.")


def main():
    baseline_model = joblib.load("logreg_model.joblib")
    advanced_model = joblib.load("advanced_model.joblib")
    
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')
    dataset_test = pd.read_csv('project_name/data/sa_spaeng_validation.csv')

    X_train, y_train = preprocess_features(dataset_train)
    X_test, y_test = preprocess_features(dataset_test)
    
    metrics_results_training = baseline_model.evaluate(X_train, y_train)
    print("Baseline Model Training:", metrics_results_training)

    metrics_results_test = baseline_model.evaluate(X_test, y_test)
    print("Baseline Model Test:", metrics_results_test)

    metrics_results_training = advanced_model.evaluate(X_train, y_train)
    print("Advanced Model Training:", metrics_results_training)

    metrics_results_test = advanced_model.evaluate(X_test, y_test)
    print("Advanced Model Test:", metrics_results_test)

    bootstrap_test(
        models=[baseline_model, advanced_model],
        X_test=X_test,
        y_test=y_test,
        metric_func=f1_score,
        n_bootstrap=1000
    )


if __name__ == "__main__":
    main()

import pandas as pd
import joblib
from train_model import split_data, preprocess_features


def main():
    baseline_model = joblib.load("logreg_model.joblib")
    advanced_model = joblib.load("advanced_model.joblib")
    
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')
    # Replace 'lang1' with 'English' and 'lang2' with 'Spanish' in the 'lid' column
    dataset_train['lid'] = dataset_train['lid'].apply(lambda labels: ['English' if l == 'lang1' else 'Spanish' if l == 'lang2' else l for l in labels])
    
    dataset_train, dataset_test = split_data(dataset_train)

    X_train, y_train = preprocess_features(dataset_train)
    X_test, y_test = preprocess_features(dataset_test)
    
    metrics_results_training = baseline_model.evaluate(X_train, y_train)
    print("Baseline Model Training:", metrics_results_training)

    metrics_results_test = advanced_model.evaluate(X_test, y_test)
    print("Baseline Model Test:", metrics_results_test)

    metrics_results_training = baseline_model.evaluate(X_train, y_train)
    print("Advanced Model Training:", metrics_results_training)

    metrics_results_test = advanced_model.evaluate(X_test, y_test)
    print("Advanced Model Test:", metrics_results_test)


if __name__ == "__main__":
    main()

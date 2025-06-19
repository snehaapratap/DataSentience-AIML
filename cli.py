import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets
import joblib

MODEL_MAP = {
    'decision_tree': DecisionTreeClassifier,
    'svm': SVC,
    'knn': KNeighborsClassifier
}

BUILTIN_DATASETS = {
    'iris': datasets.load_iris,
    'wine': datasets.load_wine,
    'breast_cancer': datasets.load_breast_cancer
}

def load_dataset(dataset_arg):
    if dataset_arg in BUILTIN_DATASETS:
        data = BUILTIN_DATASETS[dataset_arg]()
        X, y = data.data, data.target
        return X, y, data.feature_names, data.target_names
    elif os.path.isfile(dataset_arg):
        df = pd.read_csv(dataset_arg)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y, df.columns[:-1], pd.unique(y)
    else:
        print(f"Dataset '{dataset_arg}' not found. Use a valid CSV path or one of: {list(BUILTIN_DATASETS.keys())}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ML models from the command line.")
    parser.add_argument('--model', required=True, choices=MODEL_MAP.keys(), help='Model to use: decision_tree, svm, knn')
    parser.add_argument('--dataset', required=True, help='CSV file path or built-in dataset (iris, wine, breast_cancer)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    parser.add_argument('--save', type=str, default=None, help='Path to save the trained model (.pkl)')
    args = parser.parse_args()

    print(f"\n[INFO] Loading dataset: {args.dataset}")
    X, y, feature_names, target_names = load_dataset(args.dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print(f"[INFO] Training {args.model} model...")
    model = MODEL_MAP[args.model]()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(t) for t in target_names]))

    if args.save:
        joblib.dump(model, args.save)
        print(f"\n[INFO] Model saved to {args.save}")

if __name__ == '__main__':
    main() 
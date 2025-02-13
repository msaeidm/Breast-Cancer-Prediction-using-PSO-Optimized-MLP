import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_baseline_models(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Nearest Neighbor": KNeighborsClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.3f}")
    
    return results

def main():
    processed_file = os.path.join("..", "data", "processed", "processed_data.csv")
    data = pd.read_csv(processed_file)
    results = train_baseline_models(data)
    df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    os.makedirs(os.path.join("..", "results", "tables"), exist_ok=True)
    df_results.to_csv(os.path.join("..", "results", "tables", "baseline_results.csv"), index=False)
    print("Baseline model results saved.")

if __name__ == "__main__":
    main()

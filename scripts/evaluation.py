import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    os.makedirs(os.path.join("..", "results", "figures"), exist_ok=True)
    plt.savefig(os.path.join("..", "results", "figures", f"confusion_matrix_{model_name}.png"))
    plt.close()
    return cm, report

def main():
    # For demonstration, load processed data and use one model from pso_optimization
    data = pd.read_csv(os.path.join("..", "data", "processed", "processed_data.csv"))
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For demonstration, we load the MLP model from mlp_model.py (or retrain it)
    from sklearn.neural_network import MLPClassifier
    # Here we assume the best number of hidden neurons from PSO is 20 (example)
    mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    
    cm, report = evaluate_model(mlp, X_test, y_test, "Optimized_MLP")
    
    # Save evaluation report
    os.makedirs(os.path.join("..", "results", "tables"), exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join("..", "results", "tables", "evaluation_report.csv"))
    print("Evaluation completed and results saved.")

if __name__ == "__main__":
    main()

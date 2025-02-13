import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_mlp(data):
    # Assume the last column is the target class (e.g., Healthy, Benign, Malignant)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define MLP with default parameters (to be optimized later)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"MLP Model Accuracy: {accuracy:.3f}")
    return mlp, accuracy

def main():
    processed_file = os.path.join("..", "data", "processed", "processed_data.csv")
    data = pd.read_csv(processed_file)
    model, acc = train_mlp(data)
    # Save model accuracy to results
    os.makedirs(os.path.join("..", "results", "tables"), exist_ok=True)
    with open(os.path.join("..", "results", "tables", "mlp_accuracy.txt"), "w") as f:
        f.write(f"MLP Accuracy: {acc:.3f}\n")

if __name__ == "__main__":
    main()

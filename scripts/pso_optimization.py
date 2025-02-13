import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# A simple PSO optimizer for MLP parameters
# Here we optimize only one parameter: the number of neurons in a single hidden layer
def objective(hidden_neurons, X_train, y_train, X_val, y_val):
    # Train an MLP with given hidden_neurons (rounded to an integer)
    hidden_neurons = int(round(hidden_neurons))
    if hidden_neurons < 1:
        hidden_neurons = 1
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_neurons,), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_val)
    return accuracy_score(y_val, y_pred)

def pso_optimize(X_train, y_train, X_val, y_val, num_particles=10, max_iter=20):
    # Initialize particles: each particle represents a candidate number of hidden neurons (between 5 and 50)
    dim = 1
    lb, ub = 5, 50
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective(pos[0], X_train, y_train, X_val, y_val) for pos in positions])
    gbest_index = np.argmax(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    
    for it in range(max_iter):
        for i in range(num_particles):
            # Update velocity and position
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] = positions[i] + velocities[i]
            # Keep within bounds
            positions[i] = np.clip(positions[i], lb, ub)
            score = objective(positions[i][0], X_train, y_train, X_val, y_val)
            if score > pbest_scores[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_scores[i] = score
        # Update global best
        gbest_index = np.argmax(pbest_scores)
        gbest_position = pbest_positions[gbest_index].copy()
        print(f"Iteration {it+1}, Best Accuracy: {pbest_scores[gbest_index]:.3f}, Hidden Neurons: {int(round(gbest_position[0]))}")
    
    return int(round(gbest_position[0])), pbest_scores[gbest_index]

def main():
    processed_file = os.path.join("..", "data", "processed", "processed_data.csv")
    data = pd.read_csv(processed_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_neurons, best_accuracy = pso_optimize(X_train, y_train, X_val, y_val)
    print(f"PSO Optimization Result - Best Hidden Neurons: {best_neurons}, Accuracy: {best_accuracy:.3f}")
    
    # Save results
    os.makedirs(os.path.join("..", "results", "tables"), exist_ok=True)
    with open(os.path.join("..", "results", "tables", "pso_optimization.txt"), "w") as f:
        f.write(f"Best Hidden Neurons: {best_neurons}\nBest Accuracy: {best_accuracy:.3f}\n")

if __name__ == "__main__":
    main()

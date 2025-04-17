import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, optimizer='gd',batch_size=32, regularization=0.1, early_stopping=10,lr_decay=0.0, random_state=None, verbose=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer.lower()
        self.batch_size = batch_size
        self.regularization = regularization
        self.early_stopping = early_stopping
        self.lr_decay = lr_decay
        self.random_state = random_state
        self.verbose = verbose
        self.losses = []
        self.best_loss = np.inf
        self.stopped_epoch = 0

    def sigmoid(self, z):
        # Numerically stable sigmoid with clipping
        return np.clip(1 / (1 + np.exp(-z)), 1e-15, 1 - 1e-15)

    def initialize_parameters(self, n_features):
        # Small random initialization
        np.random.seed(self.random_state)
        self.W = np.random.randn(n_features, 1) * 0.01
        self.b = 0

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_loss = -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        l2_penalty = (self.regularization/(2*m)) * np.sum(self.W**2)
        return log_loss + l2_penalty

    def predict_proba(self, X):
        z = X.dot(self.W) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def update_parameters(self, X, y, current_lr):
        m = X.shape[0]
        y_pred = self.predict_proba(X)

        # Gradient with L2 regularization
        dw = (X.T.dot(y_pred - y) / m) + (self.regularization/m)*self.W
        db = np.mean(y_pred - y)

        self.W -= current_lr * dw
        self.b -= current_lr * db

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1,1)
        self.initialize_parameters(X.shape[1])

        no_improvement = 0
        np.random.seed(self.random_state)

        for epoch in range(self.epochs):
            # Learning rate decay
            current_lr = self.learning_rate / (1 + self.lr_decay*epoch)

            # Shuffle data for SGD and mini-batch
            if self.optimizer in ['sgd', 'mini-batch']:
                shuffle_idx = np.random.permutation(len(y))
                X_shuffled, y_shuffled = X[shuffle_idx], y[shuffle_idx]
            else:
                X_shuffled, y_shuffled = X, y

            # Update parameters
            if self.optimizer == 'gd':
                self.update_parameters(X_shuffled, y_shuffled, current_lr)
            elif self.optimizer == 'sgd':
                for i in range(X_shuffled.shape[0]):
                    self.update_parameters(X_shuffled[i:i+1], y_shuffled[i:i+1], current_lr)
            elif self.optimizer == 'mini-batch':
                for i in range(0, X_shuffled.shape[0], self.batch_size):
                    end = i + self.batch_size
                    self.update_parameters(X_shuffled[i:end], y_shuffled[i:end], current_lr)
            else:
                raise ValueError("Optimizer must be 'gd', 'sgd', or 'mini-batch'")

            # Calculate and store loss
            y_pred = self.predict_proba(X)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            # Early stopping check
            if loss < self.best_loss:
                self.best_loss = loss
                no_improvement = 0
            else:
                no_improvement += 1

            if self.early_stopping and no_improvement >= self.early_stopping:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

            # Progress reporting
            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs-1):
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, LR={current_lr:.6f}")

        return self

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1': f1_score(y, y_pred)
        }

        print("Confusion Matrix:")
        print(cm)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return metrics

    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.losses, label='Training Loss')
        plt.title("Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if self.stopped_epoch > 0:
            plt.axvline(self.stopped_epoch, color='r', linestyle='--',
                       label='Early Stopping')
        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_all_losses(loss_dict):
        plt.figure(figsize=(10, 5))
        for optimizer, losses in loss_dict.items():
            plt.plot(losses, label=optimizer.upper())
        plt.title("Loss Curves for Different Optimizers")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
# -------------------------
# 🚀 MAIN EXECUTION
# -------------------------

# Generate synthetic data
X, y = make_classification(n_samples=2000, n_features=12, n_informative=8,
                           n_classes=2, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
# Optimizer comparison
results = {}
config = {
    'learning_rate': 0.001,
    'epochs': 1000,
    'regularization': 0.2,
    'early_stopping': 20,
    'lr_decay': 0.005,
    'verbose': False
}

for optimizer in ['gd', 'sgd', 'mini-batch']:
    print(f"\n=== Training with {optimizer.upper()} ===")

    model = LogisticRegression(
        optimizer=optimizer,
        batch_size=64,
        random_state=42,
        **config
    )

    model.fit(X_train, y_train)
    model.plot_loss()

    print("\nTest Set Performance:")
    metrics = model.evaluate(X_test, y_test)
    results[optimizer] = metrics
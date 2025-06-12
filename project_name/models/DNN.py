import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from project_name.models.model import Model


class DNNClassifier(Model):
    """
    Deep Neural Network (DNN) classifier as our advanced model.
    Args:
        input_dim (int): Number of input features.
        hidden_dims (list, optional): List of integers specifying the number of units in each hidden layer. Defaults to [128, 64].
        num_classes (int, optional): Number of output classes. Defaults to 3.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        device (str or torch.device, optional): Device to run the model on ('cuda' or 'cpu'). If None, automatically selects CUDA if available.
    Attributes:
        device (str): Device used for computation (for optimization)
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model (nn.Sequential): The neural network model.
        loss_fn (nn.Module): Loss function (CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (Adam).
        _parameters (dict): Dictionary of model parameters.
    """
    def __init__(self, input_dim: int, hidden_dims=[128, 64], num_classes=3, lr=1e-3, epochs=20, batch_size=32, device=None):
        super().__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.3))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self._parameters = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "num_classes": num_classes,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the DNN classifier

        Args:
            X (np.ndarray): Training feature data of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples).
        """

        self.model.train()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels as integers (0 - positive, 1 - neutral, 2 - negative) for the input data.

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).
        Returns:
            np.ndarray: Array of predicted class labels for each input sample.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Evaluates the performance of the DNN based on the provided data,
         and returns a formatted string of metrics (confusion matrix, f1-score and accuracy).

        Args:
            X (np.ndarray): Feature data of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).

        Returns:
            str: Formatted string containing accuracy, F1 score, classification report, and confusion matrix.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)

        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)

        formatted_metrics = (
            f"Evaluation Metrics\n"
            f"{'='*40}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"F1 Score: {f1:.4f}\n\n"
            f"Classification Report:\n{report}\n"
            f"Confusion Matrix:\n{matrix}\n"
        )
        return formatted_metrics

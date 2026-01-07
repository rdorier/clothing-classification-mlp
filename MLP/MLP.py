from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A Multi-Layer Percpetron model for classification tasks.
    This MLP consists of five fully connected layers with batch normalization, ReLU activations, and dropout for regularization.
    """

    def __init__(self, num_classes):
        """
        Initializes the MLP model with the given number of output classes.
        @param num_classes : the number of output classes for classification, an integer.
        """
        super().__init__()

        # Fully connected layers with decreasing number of neurons
        self.fc0 = nn.Linear(784, 512) # Input layer : 784 (28x28 flattened) -> 512 neurons
        self.bn0 = nn.BatchNorm1d(512) # Batch normalization for stability

        self.fc1 = nn.Linear(512, 256) # Hidden layer : 512 -> 256 neurons
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128) # Hidden layer : 256 -> 128 neurons
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64) # Hidden layer : 128 -> 64 neurons
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, num_classes) # Output layer : 64 -> num_classes neurons

        # Dropout layer to reduce overfitting (p=0.3 means 30% dropout)
        # The dropout randomly deactivates somes neurons during training to reduce overfitting by avoiding MLP to memorize the dataset,
        # become dependant of only a few neurons, etc
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Forward pass through the network.
        @param x : Input Tensor of shape (batch_size, 1, 28, 28) (for MNIST-like data), a torch.Tensor.
        @return log-probabilities of each class (after applying log_softmax), a torch.Tensor.
            
        """
        # Flatten the input tensor from (batch_size, 1, 28, 28) to (batch_size, 784)
        x = x.view(x.shape[0], -1) # reshaping for compatibility with fully connected layers

        # First fully connected layer : Linear -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x) # Apply dropout

        # second fully connected layer : Linear -> BatchNorm -> ReLU
        x = F.relu(self.bn1(self.fc1(x)))

        # Third fully connected layer : Linear -> BatchNorm -> RelU -> dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x) # Apply dropout

        # Fourth fully connected layer : Linear -> BatchNorm -> ReLU
        x = F.relu(self.bn3(self.fc3(x)))

        # Output layer : Linear -> Log Softmax (for classification)
        x = F.log_softmax(self.fc4(x), dim=1) # log probabilities for each class

        return x
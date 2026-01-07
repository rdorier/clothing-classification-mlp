import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import numpy as np
import random
import time

from MLP import MLP


# map class to index
class_mapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"  
}


# set seed for reproductibility when using random numbers
def set_seeds():
    # set random seed value
    SEED_VALUE = 756

    # set seed for python's random module
    random.seed(SEED_VALUE)
    # set seed for NumPy to ensure rerpoductibility in operations involving randomness
    np.random.seed(SEED_VALUE)
    # set seed for pytorch to ensure consistent initialization of weights, etc
    torch.manual_seed(SEED_VALUE)

    # if CUDA (GPU support) is avialable, set additional seed to control randomness in GPU operations
    # Fix seed to make training deterministic.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE) # for single GPU operations
        torch.cuda.manual_seed_all(SEED_VALUE) # for multi-GPU operations
        # ensures deterministic behavior of cuDNN (cuda Deep Neural Network) operations
        torch.backends.cudnn.deterministic = True
        # enbaling cuDNN benchmarking mode can improve training speed for fiex input size
        torch.backends.cudnn.benchmark = True


def train(model, trainloader, criterion, optimizer, device):
    """
    Trains the given model for one epoch using the provided training data.
    @param model : The neural network model to be trained, nn.Module.
    @param trainloader : DataLoader containing the training dataset, a DataLoader.
    @param criterion : Loss function used for optimization, a function.
    @param optimizer : Optimizer used for updating model weights (e.g., Adam, SGD), a torch.optim.Optimizer.
    @param device : Device to run the training on ("cuda" for GPU, otherwise "cpu"), a string.

    @return avg_loss : The average training loss over the entire dataset, a float.
    @return accuracy : The percentage of correctly classified samples in the training set, a float.
    """
    # set model to training mode
    model.train()
    # Move model to the specified device (GPU/CPU)
    model.to(device)

    # initialize variables to track loss and accuracy
    running_loss = 0 # accumulate the total loss for averaging
    correct_predictions = 0 # counts the number of correct predictions
    total_samples = 0 # tracks the total number of processed samples

    # iterate over the training dataset
    for images, labels in trainloader:
        # move images and labels to the specified device (GPU/CPU)
        images,labels = images.to(device),labels.to(device)
        # reset gradients to prevent accumulation from previous batches
        optimizer.zero_grad()
        # forward pass : get model predictions
        outputs = model(images)

        # compute the loss between predictions and ground truth labels
        loss = criterion(outputs, labels)
        # Backpropagation : compute gradients
        loss.backward()
        # update model weights using the optimizer
        optimizer.step()

        # accumulate loss for averaging
        running_loss += loss.item()
        # get the predicted class with the highest probability
        _, predicted = torch.max(outputs.data, dim=1)
        # update total sample count
        total_samples += labels.size(0)
        # count correctly predicted samples
        correct_predictions += (predicted == labels).sum().item()

    # compute average loss accross all batches
    avg_loss = running_loss / len(trainloader)
    # compute training accuracy as a percentage
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy # return avergae loss and accuracy after one training epoch


def validation(model, val_loader, criterion, device):
    """
    Evaluates the given model on the validation dataset.
    @param model : The trained neural network model to be evaluated, a nn.Module.
    @param val_loader : Datalader containing the validation dataset, a DataLoader.
    @param criterion : loss function used to measuer model performance (e.g., Negative Log Likelihood Loss), a function.
    @param device : The device to run the evaluation on ("cuda" for GPU, "cpu" otherwise), a string.
    @return avg_loss : The average validation loss overt the dataset, a float.
    @return accuracy : The percentage of correctly classified samples in the validation set, a float.
    """
    # set the model to evaluation mode (disables dropout, batch norm behaves differently)
    model.eval()
    # move model to the specified device (GPU/CPU)
    model.to(device)

    # Initialize variables to track loss and accuracy
    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Disable gradient computation to save memory and speed up inference
    with torch.no_grad():
        # iterate over the validation dataset
        for images, labels in val_loader:
            # move images and labels to the specified device (GPU/CPU)
            images,labels = images.to(device),labels.to(device)

            # Forward pass : get model prediction
            outputs = model(images)

            # compute loss between predictions and ground truth labels
            loss = criterion(outputs, labels)
            # accumulate loss for averaging
            running_loss += loss.item()
            # get the predicted class with the highest probability
            _, predicted = torch.max(outputs.data, 1) #(B, class_id)
            # update total sample count
            total_samples += labels.size(0)
            # count correctly predicted samples
            correct_predictions += (predicted == labels).sum().item()

    # compute average loss across all batches
    avg_loss = running_loss / len(val_loader)
    # compute validation accuracy as a percentage
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy


def main(model, training_loader, validation_loader, epochs=5, device = "cuda"):
    """
    Trains and validates the given model over multiple epochs and plots the results.
    @param model : The neural network model to be trained and validated, a nn.Module.
    @param training_loader : The Dataloader containing the training dataset, a DataLoader.
    @param validation_loader : The Dataloader containing the validation dataset, a DataLoader.
    @param epochs : The number of epochs to train the model, an integer.
    @param device : The device to run the training and validation on ("cuda" for GPU, otherwise "cpu").
    """
    # lists to store loss and accuracy values for plotting
    training_losses, validation_losses = [], []
    training_accuracies, validation_accuracies = [], []

    # training loop for multiple epochs
    for epoch in range(epochs):
        # train the model for one epoch
        training_loss, training_accuracy = train(model, training_loader, criterion, optimizer, device)
        # validate the model after training
        validation_loss, validation_accuracy = validation(model, validation_loader, criterion, device)

        # store losses and accuracies for analysis
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        # print training progress for each epoch
        print(f"Epoch {epoch+1:0>2}/{epochs} - Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}% - Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")


if __name__ == "__main__":
    print("Let start training !")
    # set random seeds for reproductibility when using random numbers
    set_seeds()

    # transforms are operations to apply to images. transforms.ToTensor() creates a transform operation that convert PIL image or numpy array to Tensors
    # and divide values by 255 to have all values between 0 and 1
    raw_transform = transforms.Compose([transforms.ToTensor()])
    # Step 1 : download the training set without normalization
    training_set_raw = datasets.FashionMNIST(root="F_MNIST_data", download=True, train=True, transform=raw_transform)

    # Step 2 : concatenate given sequence of tensors, flatten value as a 1-dimension vector and compute mean and variance from the training set
    all_pixels = torch.cat([img.view(-1) for img, _ in training_set_raw])
    mean = all_pixels.mean().item()
    variance = all_pixels.std().item()

    print(f"Computed Mean : {mean : .4f}, Computed Std : {variance:.4f}")

    # Step 3 : Define the new transform using the computed mean and variance
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (variance,))
    ])

    # Step 4 : reload datasets with proper normalization
    train_set = datasets.FashionMNIST(root="F_MNIST_data", download=True, train=True, transform=transform)
    validation_set = datasets.FashionMNIST(root="F_MNIST_data", download=True, train=False, transform=transform)

    print("Total Train Images : ", len(train_set))
    print("Total Validation Images : ", len(validation_set))

    # defining data loaders for training and testing datasets
    # To avoid the MLP network from learning the sequence pattern in the dataset we shuffle the train dataset
    train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, batch_size = 64)
    # we do not need to shuffle validation data set as it takes time (in training we want to avoid biases, but we aren't learning anymore)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle = False, batch_size = 64)

    # Instantiate the model with 10 output classes (e.g., for MNIST)
    mlp_model = MLP.MLP(num_classes=10)

    # Define the loss function
    criterion = F.nll_loss # Negative Log Likelihood Loss, commonly used for classification tasks

    # Define the optimizer : Adam optimizer with a learning rate of 0.01
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-2)

    # Number of epochs for training
    num_epochs = 40 # the model will train for 40 complete passes over the dataset

    # select the device for computation (GPU if available, otherwise CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # launch training and validation
    main(mlp_model, train_loader, validation_loader, epochs = num_epochs, device = DEVICE)
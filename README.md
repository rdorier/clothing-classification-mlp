# Clothing classification Multi-Layer Perceptron

A clothing classification project to practice PyTorch. It uses a Multi-Layer Perceptron to classify images from the Fashion MNIST dataset that you can find here on Kaggle : https://www.kaggle.com/datasets/zalando-research/fashionmnist

![Thumbnail of the Fashion MNIST dataset](doc/images/Fashion-MNIST-dataset_Q320.jpg)

## How To

To run the project, you only have to clone this repository and run the `mlp_training.py` script using Python 3. This script only trains the Multi-Layer Perceptron, so feel free to try it with some images ! You will find some usage examples inside the supplied [Jupyter Notebook](README.md#jupyter-notebook).

### Dependencies

This script requires Python 3 as well as PyTorch and NumPy in order to work.

## Results

Here is two graphs representing the loss and accuracy of the training and validation session as they progress through epochs.

![Loss and accuracy of the training progress](doc/images/loss_and_accuracy_graphs.png)

The perceptron is a pretty naive approach, but for tiny and simple images like the ones from the Fashion MNIST dataset, we can see that it gives fairly good results, as shown is the following confusion matrix :

![Confusion Matrix for the results of our MLP](doc/images/confusion_matrix.png)  

We can see that the MLP mixed up shirt and T-shirts sometimes. Indeed the confusion matrix tells us that when the target clothing is a T-shirt, it anwsers 795 times right, but it answers 133 times that it is a shirt.

### Jupyter Notebook

You can find in the `doc` folder of this repository a Jupyter Notebook with an interactive version of the script use to train this model, as well as graphs showing training progress and results. You can also find a test of the network where we try to make the MLP predict the class of an object inside the test dataset.

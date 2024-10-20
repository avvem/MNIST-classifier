import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transform for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load the MNIST dataset
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for training and testing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First layer
        self.fc2 = nn.Linear(128, 10)        # Output layer
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)          # Output layer
        return F.log_softmax(x, dim=1)  # Log softmax for output probabilities

# Initialize the model, optimizer, and loss function
model = SimpleNN().to(device)  # Move model to device
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
criterion = nn.NLLLoss()  # Negative log likelihood loss

# Function to plot training progress (loss and accuracy)
def plot_progress(epochs, train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    
    # Plot training and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')  # Save plot
    plt.close()  # Close the figure to free memory

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save confusion matrix plot
    plt.close()  # Close the figure to free memory

# Function to plot sample predictions
def plot_sample_predictions(images, labels, predictions, num_images=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)  # Create a 2x5 grid
        plt.imshow(images[i].squeeze(), cmap='gray')  # Show the image
        plt.title(f'True: {labels[i]}, Pred: {predictions[i]}')  # Show true and predicted labels
        plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig('sample_predictions.png')  # Save sample predictions plot
    plt.close()  # Close the figure to free memory

# Training the model with progress visualization
num_epochs = 10  # Number of epochs to train
epochs = []  # List to store epoch numbers
train_losses = []  # List to store training losses
train_accuracies = []  # List to store training accuracies
test_accuracies = []  # List to store test accuracies

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize loss for this epoch
    correct_train = 0  # Initialize correct predictions count for training
    total_train = 0  # Initialize total samples count for training
    
    # Iterate through the training data
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)  # Move data to device
        
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        # Update running loss and correct predictions
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total_train += labels.size(0)  # Update total samples
        correct_train += (predicted == labels).sum().item()  # Update correct predictions
    
    # Calculate and store metrics for this epoch
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct_train / total_train
    
    # Calculate test accuracy
    model.eval()  # Set model to evaluation mode
    correct_test = 0  # Initialize correct predictions count for testing
    total_test = 0  # Initialize total samples count for testing
    with torch.no_grad():  # No need to calculate gradients
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total_test += labels.size(0)  # Update total samples
            correct_test += (predicted == labels).sum().item()  # Update correct predictions
    
    test_acc = correct_test / total_test  # Calculate test accuracy
    
    # Store the metrics
    epochs.append(epoch + 1)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    test_accuracies.append(test_acc)
    
    # Plot the progress
    plot_progress(epochs, train_losses, train_accuracies, test_accuracies)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}")

print("Training complete!")

# Save the trained model
torch.save(model.state_dict(), 'mnist_model_with_visualization.pth')

# Predictions for confusion matrix and sample predictions
model.eval()  # Set model to evaluation mode
all_preds = []  # List to store all predictions
all_labels = []  # List to store all labels
with torch.no_grad():  # No need to calculate gradients
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)  # Move data to device
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        all_preds.extend(predicted.cpu().numpy())  # Store predictions
        all_labels.extend(labels.cpu().numpy())  # Store true labels

# Plot confusion matrix
plot_confusion_matrix(all_labels, all_preds, classes=range(10))

# Plot sample predictions
plot_sample_predictions(testset.data.numpy(), all_labels, all_preds)

# Final plot
plot_progress(epochs, train_losses, train_accuracies, test_accuracies)
print("Final plot saved as 'training_progress.png'")

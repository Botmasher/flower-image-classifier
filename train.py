import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict     # classifier setup
from cat_to_name import cat_labels      # map indices to labels
from devices import get_device          # select cpu or gpu
from data import setup_data             # datasets and dataloaders

# Create a dict mapping integers to image labels
cat_to_name = cat_labels()

## Set up the classifier and model

def setup_model(architecture, datasets, dataloaders, lr=0.002, hidden_units=None):
    """Build a model and model classifier for training."""
    
    # Assign datasets and dataloaders
    train_dataset = datasets['train']
    valid_dataset = datasets['valid']
    test_dataset = datasets['test']
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']
    
    # Select architecture for classifier
    architecture_settings = {
        'vgg13': [models.vgg13, 25088],
        'vgg16': [models.vgg16, 25088],
        'densenet': [models.densenet121, 1024],
        #'inception': [models.inception_v3, 2048],
        #'resnet': [models.resnet18, 512],
        'alexnet': [models.alexnet, 9216]
    }

    build_model, architecture_size = architecture_settings.get(architecture)

    # Instantiate and configure pytorch classifier
    # using classifier feature settings
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(hidden_units if hidden_units else architecture_size, 514)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(514, 102)),
        ('dropout', nn.Dropout(0.2)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))

    # Select torchvision model
    model = build_model(pretrained = True)

    # Freeze parameters so to avoid backpropagating
    for param in model.parameters():
        param.requires_grad = False

    # Attach classifier to pretrained model
    model.classifier = classifier

    # Train the classifier parameters; feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

    return (model, optimizer)

## Train and validate the model
def train_model(model, optimizer, train_loader, valid_loader, epochs=1, print_every=5, device=None):
    # Send model to GPU or CPU
    if not device:
        device = get_device('gpu')
    model.to(device)

    # Setup negative log probability for log softmax
    criterion = nn.NLLLoss()

    # Run through training epochs
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # output the loss     
                print(f"""
                    Epoch {epoch + 1}/{epochs}
                    Training loss: {running_loss / print_every:.3f}
                    Validation loss: {valid_loss / len(valid_loader):.3f}
                    Validation accuracy: {accuracy / len(valid_loader):.3f}
                """)
                running_loss = 0
                model.train()

## Test the model to verify post-training model accuracy

# Run through test dataset following pytorch tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            forward_results = model.forward(inputs)
            predicted = torch.max(forward_results.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().float().item()
        print(f"Test accuracy: {correct / total}")

## Save and rebuild the model

# Save a model checkpoint
def save_model(save_dir, model, architecture, optimizer, train_dataset, device=None):
    """Save a checkpoint for the model."""
    
    # Input size map for various architectures
    input_sizes = {
        'vgg13': 25088,
        'vgg16': 25088,
        'densenet': 1024,
        #'inception': 2048,
        #'resnet': 512,
        'alexnet': 9216
    }
    model.class_to_idx = train_dataset.class_to_idx
    model.to(device)
    checkpoint = {
        'classifier': model.classifier,
        'features': model.features,
        'optimizer': optimizer.state_dict(),
        'input_size': input_sizes[architecture],
        'output_size': 102,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)
    print(f"Trained model saved to {save_dir}")
    return checkpoint

def parse_args():
    """Read training CLI args passed on script start"""
    parser = argparse.ArgumentParser()
    
    # default arg for load directory
    parser.add_argument("data_dir", type = str, help = "a data directory for training the model")
    
    # optional args
    parser.add_argument("--save_dir", type = str, help = "a save directory for a model checkpoint")
    parser.add_argument("--arch", type = str, help = "an architecture for the model")
    parser.add_argument("--learning_rate", type = float, help = "a learning rate for the model")
    parser.add_argument("--hidden_units", type = int, help = "the input size of the model architecture")
    parser.add_argument("--epochs", type = int, help = "the number of epochs to train the model")
    parser.add_argument("--gpu", action = "store_true", help = "set to train on the GPU instead of CPU")
    
    return parser.parse_args()

def main():
    # User arguments
    args = parse_args()
    
    # Datasets and dataloaders
    img_data = setup_data(args.data_dir)
    
    # Build, train and save the model
    architecture = args.arch if args.arch else "vgg13"
    device = get_device("GPU" if args.gpu else "CPU")
    model, optimizer = setup_model(
        architecture,
        img_data['datasets'],
        img_data['dataloaders'],
        lr = args.learning_rate if args.learning_rate else 0.002,
        hidden_units = args.hidden_units
    )
    train_model(
        model,
        optimizer,
        img_data['dataloaders']['train'],
        img_data['dataloaders']['valid'],
        epochs = args.epochs if args.epochs else 1,
        device = device
    )
    if args.save_dir:
        save_model(args.save_dir, model, architecture, optimizer, img_data['datasets']['train'], device = device)

main()

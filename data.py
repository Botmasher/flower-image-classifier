import torch
from torchvision import datasets, transforms

# Set up training, testing and validation data
def setup_data(data_dir):
    """Create image datasets and dataloaders from training, validation
    and testing folders from the given data directory."""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    # NOTE: the pre-trained networks require 224 x 224 pixels input

    # Randomize scale, rotation, flip to help network generalize
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Scale, crop, normalize new testing and validation data
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)

    # Use the image datasets and transforms to define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)

    return {
        'transforms': {
            'train': train_transforms,
            'test': test_transforms,
            'valid': test_transforms
        },
        'datasets': {
            'train': train_dataset,
            'test': test_dataset,
            'valid': valid_dataset
        },
        'dataloaders': {
            'train': train_loader,
            'test': test_loader,
            'valid': valid_loader
        }
    }
import argparse
import re
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image                  # for image processing
from devices import get_device         # for selecting cpu or gpu
from cat_to_name import cat_labels     # JSON image labels

# Load a saved checkpoint
def load_checkpoint(file_path, device=None):
    """Rebuild a model from a saved checkpoint."""
    if not device:
        device = get_device('gpu')
    model_data = torch.load(file_path)
    model = models.vgg16(pretrained = True)
    model.classifier = model_data.get('classifier', {
        ('fc1', nn.Linear(25088, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 102)),
        ('dropout', nn.Dropout(0.2)),
        ('output', nn.LogSoftmax(dim = 1))
    })
    model.class_to_idx = model_data.get('class_to_idx', model_data.get('idx_to_class'))
    model.load_state_dict(model_data['state_dict'])
    model.to(device)
    print("Model loaded.")
    return (model, model_data)

## Preprocess image to use as model input
## This is used when making predictions below.
def process_image(image, size=256):
    """Scales, crops, and normalizes a PIL image for a PyTorch model.
    Returns a Numpy array.
    """
    # Load and set up image
    pil_image = Image.open(image)

    # # NOTE: below attempted to follow instructions on using and adjusting PIL image.
    # # These attempts were all unsuccessful. Just using transforms.Compose
    # # seemed to work fine, but does not avoid the issues raised in the instructions!
    #
    # "You'll want to use PIL to load the image (documentation). It's best to write a
    # function that preprocesses the image so it can be used as input for the model.
    # This function should process the images in the same manner used for training.
    # "First, resize the images where the shortest side is 256 pixels, keeping the
    # aspect ratio. This can be done with the thumbnail or resize methods. Then you'll
    # need to crop out the center 224x224 portion of the image.
    # Color channels of images are typically encoded as integers 0-255, but the model
    # expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy
    # array, which you can get from a PIL image like so np_image = np.array(pil_image).
    # As before, the network expects the images to be normalized in a specific way.
    # For the means, it's [0.485, 0.456, 0.406] and for the standard deviations
    # [0.229, 0.224, 0.225]. You'll want to subtract the means from each color channel,
    # then divide by the standard deviation.
    # And finally, PyTorch expects the color channel to be the first dimension but
    # it's the third dimension in the PIL image and Numpy array. You can reorder
    # dimensions using ndarray.transpose. The color channel needs to be first and retain
    # the order of the other two dimensions."
    #
    # # resize image
    # width, height = pil_image.size
    # if width > height:
    #     height = size
    #     width *= (size / height)
    # else:
    #     width = size
    #     height *= (size / height)
    # pil_image.thumbnail((width, height))
    # # normalize color values
    # np_image = np.array(pil_image)
    # np_image = np_image.astype('float32')
    # np_image /= 255.0
    # print(np_image.size)
    #
    # # PyTorch expects the color channel to be the first dimension.
    # # It's the third dimension in the PIL image and Numpy array.
    # # Reorder dimensions using ndarray.transpose. The color channel needs
    # # to be first; retain the order of the other two dimensions.
    # image_a = image_a.transpose(2,0,1)

    # Transform into image tensor
    transformed_image = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])(pil_image) #(np_image)

    return transformed_image

# check the preprocessor function
def imshow(image, ax=None, title=None):
    """Imshow for Tensor. Checks that preprocessing worked by
    reversing the process and returning original (cropped) image"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes it is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


## Predict with the model
## This is where you pass in an image and make predictions!

# Predict the top 5 or so (top-𝐾) most probable classes
# in the tensor (x.topk(k))
def predict(image_tensor, model, k=5):
    """Returns the top 𝐾 most likely classes for an image along with the
    probabilities. This method returns both the highest probabilities and
    the indices of those probabilities corresponding to the classes.
    Args:
        image_path: path to an image file
        model:      saved model checkpoint
        k:          number of top predictions to return
    Returns:
        k highest probabilities and their class labels
    """
    # Calculate the class probabilities then find the 𝐾 largest values
    # Turn off gradients to speed up this part
    with torch.no_grad():
        log_ps = model.forward(image_tensor)
        ps = torch.exp(log_ps)
        top_ks = ps.topk(k, dim = 1)
    top_ps = top_ks[0][0]
    top_idx = top_ks[1][0]
    
    # Build a list of classes and probabilities
    # Convert indices to classes using model.class_to_idx (added earlier)
    # Invert the dict to get a mapping from index to class
    classes = [
        k for k, v in model.class_to_idx.items()
        if v in top_idx
    ]
    
    # Return classes and predictions
    return np.array([classes, top_ps])

def parse_args():
    """Read prediction CLI args passed on script start"""
    parser = argparse.ArgumentParser()
    
    # default arg for image load directory
    parser.add_argument("path_to_image", type = str, help = "path to an image to predict")
    parser.add_argument("checkpoint", type = str, help = "path to a model checkpoint to load")
    
    # optional args
    parser.add_argument("--top_k", type = int, help = "return top K most likely classes")
    parser.add_argument("--category_names", type = str, help = "use a mapping of categories to real names")
    parser.add_argument("--gpu", action = "store_true", help = "use the GPU instead of CPU for inference")
    
    return parser.parse_args()

def main():
    # Read user arguments
    args = parse_args()
    
    # Load the model
    device = get_device("GPU" if args.gpu else "CPU")
    model, model_data = load_checkpoint(args.checkpoint, device = device)

    # Load map of labels and indices optionally passing in custom json
    cat_to_name = cat_labels(args.category_names) if args.category_names else cat_labels()

    # Get image name and label for image
    image_path = args.path_to_image
    image_class = image_path.split("/")[-2]
    image_label = cat_to_name.get(image_class, "Undefined")
    
    # Load image tensor and calculate top-k predictions
    image = process_image(args.path_to_image)
    unsqueezed_image = image.unsqueeze(0)
    #unsqueezed_image.requires_grad = False
    #unsqueezed_image.to(device)
    model.to('cpu')
    model.eval()
    # Get the class labels and prediction probabilities
    k = args.top_k if args.top_k else 5
    predictions = predict(unsqueezed_image, model, k = k)

    # Top-k image labels and likelihoods
    k = args.top_k if args.top_k else 1
    print(f"\nTop {k} predictions for {image_path}:")
    # Get the predicted classes and probabilities
    predictions = predict(unsqueezed_image, model, k = k)
    # Log out the top labels and probabilities
    for i in range(len(predictions[0])):
        print(f"{cat_to_name[predictions[0][i]]}: {float(predictions[1][i])}")
    # Log the real image label
    print(f"True classification: {image_label}\n")  

main()

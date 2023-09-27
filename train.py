import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

def input_args():
    parser = argparse.ArgumentParser(description="Train network on a dataset")
    parser.add_argument("--data_dir", type=str, default="flowers", help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default='flower_classifier_checkpoint.pth', help="Directory to save the model checkpoint")
    parser.add_argument("--arch", default="vgg16", help="model architecture")
    parser.add_argument("--hidden_units", type=int, default=4096, help="Number of hidden units in the classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", default='GPU', help="Use GPU or CPU for training")

    return parser.parse_args()
    
def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, epochs, gpu):
    #load dataset
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load model to allow users choose from two different archs
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Freeze feature parameters
        for param in model.parameters():
            param.requires_grad = False
        # Classifier for VGG16
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, len(image_datasets['train'].classes)))]))
        
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        # Freeze feature parameters
        for param in model.parameters():
            param.requires_grad = False
        # Classifier for Densenet
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, len(image_datasets['train'].classes)))]))
    else:
        raise ValueError("Unsupported architecture. Please choose from 'VGG' or 'Densenet'.")

    model.classifier = classifier  
    
    # Set device 
    device = torch.device("cuda" if gpu.lower() == 'gpu' and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0
        
        model.train()
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)
                
                output = model(images)
                loss = criterion(output, labels)
                
                valid_loss += loss.item()
                
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        # Print training and validation stats
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training loss: {train_loss/len(dataloaders['train']):.2f}.. "
              f"Validation loss: {valid_loss/len(dataloaders['valid']):.2f}.. "
              f"Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

        print("Training completed!")
        
        print("Model testing in progress...")
        model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in dataloaders['test']:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        print(f"Test accuracy: {accuracy/len(dataloaders['test'])*100:.2f}%")
        
        print("Testing completed!")
        
    # Save the model checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)
    print("Model checkpoint saved!")


if __name__ == "__main__":
    args = input_args()
    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description='Predict class of input image')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('--checkpoint', type=str, default='flower_classifier_checkpoint.pth', help='Path to the model checkpoint')
    parser.add_argument('--gpu', default='GPU', help="Use GPU or CPU")
    parser.add_argument('--top_k', default=5, type=int, help="Top likely classes")
    parser.add_argument('--cat_to_name', default='cat_to_name.json', help='Path to category name mapping')
    
    return parser.parse_args()

def process_image(image_path):
    #Load and process the image
    image = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = preprocess(image)
    
    return image

def predict(image_path, checkpoint_path, topk, gpu):
    #Load pre-trained model
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    classifier = checkpoint['classifier']
    class_to_idx = checkpoint['class_to_idx']
    
    model = getattr(models, arch)(pretrained=True)
    model.classifier = classifier
    model.class_to_idx = class_to_idx

    #Set device
    device = torch.device("cuda" if gpu.lower() == 'gpu' and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #Process image
    processed_image = process_image(image_path)
    image_tensor = processed_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)
        
    top_probs, top_indices = probabilities.topk(topk)

    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]

    return top_probs[0].tolist(), top_classes

if __name__ == "__main__":
    args = get_args()
    top_probs, top_classes = predict(args.image_path, args.checkpoint, args.top_k, args.gpu)

    #Category names
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    category_names = [cat_to_name[class_] for class_ in top_classes]

    #Print results
    for prob, class_, category_name in zip(top_probs, top_classes, category_names):
        print(f"Class: {class_}, Category: {category_name}, Probability: {prob:.4f}")

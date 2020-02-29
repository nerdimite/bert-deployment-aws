import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import BERTClassifier

from utils import preprocess

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = "cpu"
    model = BERTClassifier()

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location='cpu'))

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = "cpu"
    
    # Preprocess the input as per BERT
    tokens_ids, attn_mask = preprocess(input_data)
    tokens_ids, attn_mask = tokens_ids.to(device), attn_mask.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    with torch.no_grad():
        
        # Getting the output and passing through sigmoid
        result = model(tokens_ids, attn_mask)
        probs = torch.sigmoid(result.unsqueeze(-1))
        output = (probs > 0.5).long().squeeze(0)
        
        print(output.item())

        return output.item()

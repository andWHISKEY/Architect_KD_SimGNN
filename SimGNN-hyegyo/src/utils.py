"""Data processing utilities."""

import json
import math
import numpy as np
import random
from texttable import Texttable
from torch.nn import CrossEntropyLoss
from torch.nn.functional import mse_loss,l1_loss
import torch

def set_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(random_seed)
    random.seed(random_seed)

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data

def print_evaluation(ground_truth,scores):
    """
    Printing the error rates.
    """
    norm_ged_mean = np.mean(ground_truth)
    base_error = np.mean([(n-norm_ged_mean)**2 for n in ground_truth])
    model_error = np.mean(scores)
    print("\nBaseline error: " +str(round(base_error, 5))+".")
    print("\nModel test error: " +str(round(model_error, 5))+".")
    return model_error

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    score=mse_loss(prediction,target)

    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/100.0 
    return norm_ged


def create_batches(train_dataloader,BATCH_SIZE):
    """
    Creating batches from the training graph list.
    :return batches: List of lists with batches.
    """
    batches = []
    temp=[]
    for i,graph in enumerate(train_dataloader):
        graph=graph.squeeze()
        if i%(BATCH_SIZE-1)==0:
            temp.append(graph)
            batches.append(temp)
            temp=[]
        elif i==len(train_dataloader)-1:
            temp.append(graph)
            batches.append(temp)
            del temp
        else:
            temp.append(graph)

    return batches

def transfer_to_torch(data,global_labels):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        edges_1 = data["graph_1"]

        edges_2 = data["graph_2"]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1, features_2 = [], []

        for n in data["labels_1"]:
            features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_ged = data["ged"]/100.0#(0.5*(len(data["labels_1"])+len(data["labels_2"])))

        new_data["target"] = torch.from_numpy(1 - np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        return new_data
import json
import glob
from tqdm import tqdm
import numpy as np
from utils import *
from torch.utils.data import Dataset,DataLoader
import torch
from param_parser import parameter_parser


def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    
    return data


class GraphDataset(Dataset):
    def __init__(self, is_train=True):
        # global label의 개수를 알기위해 수행하는 과정
        self.args = parameter_parser()
        self.is_train=is_train
        self.number_of_labels,self.global_labels=self.initial_label_enumeration()
        
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = sorted(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        print(len(self.global_labels))
        return len(self.global_labels),self.global_labels

    def __getitem__(self, index):
        #transfer to torch 함수가 여기
        if self.is_train:
            data=self.training_graphs[index]
        else:
            data=self.testing_graphs[index]    
        # dataname=data
        data = process_pair(data)
        # print(f'get item data:{data} ')
        
        transfer_data=dict()
        edges_1 = data["graph_1"]
        edges_2 = data["graph_2"]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1, features_2 = [], [] 

        for n in data["labels_1"]:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        transfer_data["edge_index_1"] = edges_1
        transfer_data["edge_index_2"] = edges_2

        transfer_data["features_1"] = features_1
        transfer_data["features_2"] = features_2

        norm_ged = data["ged"]/100.0#(0.5*(len(data["labels_1"])+len(data["labels_2"])))

        transfer_data["target"] = torch.from_numpy(1 - np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        return transfer_data,norm_ged

        
    def __len__(self):
        if self.is_train:
            return len(self.training_graphs)
        else:
            return len(self.testing_graphs)


if __name__ == '__main__':
    train_data=GraphDataset(is_train=True)
    print(train_data.__getitem__(4))
    # train_dl=DataLoader(train_data,1,shuffle=True)
    # for data in train_dl:
    #     print(data)
    



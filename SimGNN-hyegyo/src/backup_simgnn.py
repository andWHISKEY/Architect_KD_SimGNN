import torch
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from torch.nn import Conv1d
import time


class Student(torch.nn.Module): 
    def __init__(self, args, number_of_labels,device):
        super().__init__()
        self.args = args
        self.device=device
        self.number_labels = number_of_labels
        print(f'number_labels:{number_of_labels}')
        self.convolution_1 = GCNConv(number_of_labels, self.args.filters_1)
        self.convolution_2 = Conv1d(in_channels=self.args.filters_1, out_channels=self.args.filters_2,kernel_size=1)
        self.convolution_3 = Conv1d(in_channels=self.args.filters_2, out_channels=self.args.filters_3,kernel_size=1)
        
    
    def layermodule(self,features,edge_index):
        features=self.convolution_1(features,edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                            p=self.args.dropout,
                            training=self.training)
        features=torch.transpose(features, 0, 1)
        features=features.unsqueeze(0)
        features=self.convolution_2(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                            p=self.args.dropout,
                            training=self.training)
        features=self.convolution_3(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                            p=self.args.dropout,
                            training=self.training)
        features=features.mean(dim=-1)
        features=torch.transpose(features, 0, 1)

        return features

    def forward(self, data):
        edge_index_1 = data["edge_index_1"].to(self.device)
        edge_index_2 = data["edge_index_2"].to(self.device)
        features_1 = data["features_1"].to(self.device)
        features_2 = data["features_2"].to(self.device)

        features_vector_1=self.layermodule(features_1,edge_index_1)
        features_vector_2=self.layermodule(features_2,edge_index_2)
        return features_vector_1,features_vector_2
        
# --------------------------------------------------------------------------------
# SimGNN 
# --------------------------------------------------------------------------------

class SimGNN(torch.nn.Module):
    def __init__(self,args,number_of_labels,device):
        super(SimGNN,self).__init__()
        self.device=device
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
            
    def setup_layers(self):
        """
        Creating the layers.
        """
        self.feature_count = self.args.tensor_neurons
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        self.sigmoid = torch.nn.Sigmoid()    


    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        # print(features,edge_index)
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].to(self.device)
        edge_index_2 = data["edge_index_2"].to(self.device)
        features_1 = data["features_1"].to(self.device)
        features_2 = data["features_2"].to(self.device)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = self.sigmoid(self.scoring_layer(scores))
        return score,pooled_features_1,pooled_features_2
    
    def embedded_forward(self, embedding_vector1, embedding_vector2):
        scores = self.tensor_network(embedding_vector1, embedding_vector2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = self.sigmoid(self.scoring_layer(scores))
        return score

class SimGNN_student(torch.nn.Module):
    def __init__(self,args,number_of_labels,device):
        super(SimGNN_student,self).__init__()
        self.teacher = SimGNN.__init__(self,args,number_of_labels,device)
        self.student = Student.__init__(self,args,number_of_labels,device)
        self.device=device
        self.args = args
        self.number_labels = number_of_labels
    
    def layermodule(self, features, edge_index):
        return self.student.layermodule(features, edge_index)
            
    def setup_layers(self):
        return self.teacher.setup_layers()

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        return self.teacher.convolutional_pass(edge_index, features)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        return self.teacher.forward(data)
        
    def student_forward(self, data):
        return self.student.forward(data)   
    
    def embedded_forward(self, embedding_vector1, embedding_vector2):
        return self.teacher.embedded_forward(embedding_vector1, embedding_vector2)

    def student_forward_total(self,data):
        fv1,fv2=self.student_forward(data)
        return self.embedded_forward(fv1,fv2)
    
    def teacher_load_state(self, path):
        self.teacher.load_state_dict(torch.load(path))

    def student_load_state(self, path):
        self.student.load_state_dict(torch.load(path))

class Student2(torch.nn.Module): 
    def __init__(self, args, number_of_labels,device):
        super().__init__()
        self.args = args
        self.device=device
        self.number_labels = number_of_labels
        self.feature_count = self.args.tensor_neurons
        self.convolution_1 = GCNConv(number_of_labels, self.args.filters_1)
        self.convolution_2 = Conv1d(in_channels=self.args.filters_1, out_channels=self.args.filters_2,kernel_size=1)
        self.convolution_3 = Conv1d(in_channels=self.args.filters_2, out_channels=self.args.filters_3,kernel_size=1)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        self.sigmoid = torch.nn.Sigmoid()    
        
    
    def layermodule(self,features,edge_index):
        features=self.convolution_1(features,edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                            p=self.args.dropout,
                            training=self.training)
        features=torch.transpose(features, 0, 1)
        features=features.unsqueeze(0)
        features=self.convolution_2(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                            p=self.args.dropout,
                            training=self.training)
        features=self.convolution_3(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                            p=self.args.dropout,
                            training=self.training)
        features=features.mean(dim=-1)
        features=torch.transpose(features, 0, 1)

        return features

    def forward(self, data):
        edge_index_1 = data["edge_index_1"].to(self.device)
        edge_index_2 = data["edge_index_2"].to(self.device)
        features_1 = data["features_1"].to(self.device)
        features_2 = data["features_2"].to(self.device)

        features_vector_1=self.layermodule(features_1,edge_index_1)
        features_vector_2=self.layermodule(features_2,edge_index_2)
        return features_vector_1,features_vector_2

    def embedded_forward(self, embedding_vector1, embedding_vector2):
        scores = self.tensor_network(embedding_vector1, embedding_vector2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = self.sigmoid(self.scoring_layer(scores))
        return score    
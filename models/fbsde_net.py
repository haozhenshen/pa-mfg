
import torch as torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_outputs):
        super(Network, self).__init__()
        #Pass input parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_out = n_outputs
        #Construct network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_out)
        nn.init.xavier_uniform_(self.fc3.weight)
        

    #def forward(self, x0, sigma, rho, dB, N, NT)
    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x= torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output
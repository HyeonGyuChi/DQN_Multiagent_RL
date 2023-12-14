import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Qnet(nn.Module):
    def __init__(self, state_size, action_size, num_nodes):
        super(Qnet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_nodes =  num_nodes

        # a series of fully connected layer for estimating Q(s,a)
        self.fc1 = nn.Linear(self.state_size, self.num_nodes)
        self.fc2 = nn.Linear(self.num_nodes, self.num_nodes)
        self.fc3 = nn.Linear(self.num_nodes, self.action_size)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()        
    
    def forward(self, x):
        x = self.relu1(self.fc1(x)) # state
        x = self.relu2(self.fc2(x))
        out = self.fc3(x) # action

        return out
        
class Brain():
    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.optimizer_model = arguments['optimizer']

        # model
        self.model = Qnet(self.state_size, self.action_size, self.num_nodes) # main network

        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                self.model.load_state_dict(self.weight_backup)
                
        self.model_ = Qnet(self.state_size, self.action_size, self.num_nodes) # target network
        self.update_target_model()

        # optimizer
        if self.optimizer_model == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_model == 'RMSProp':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        # loss
        self.loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)
        
        # to GPU
        self.model.to(DEVICE) # main network
        self.model_.to(DEVICE) # target network
    
    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output
        self.model.train()
        self.model_.train()
        
        # forward
        x = torch.from_numpy(x).float().to(DEVICE)
        y = torch.from_numpy(y).float().to(DEVICE)
        sample_weight = torch.from_numpy(sample_weight).float().to(DEVICE)
        
        y_pred = self.model(x) # state -> Q value

        # print(x.size()) # [32, 8]
        # print(y.size()) # [32, 5]
        # print(sample_weight.size()) # [32,]
        # print(y_pred.size()) # [32, 5]
        
        # calc loss
        # sample_weight on Pytorch // https://stackoverflow.com/questions/77299691/how-to-set-sample-weights-in-torch-loss-functions
        loss = self.loss_fn(y_pred, y)
        
        # print(loss.size()) # [32, ]

        if sample_weight is not None:
            loss = loss * sample_weight
        
        loss = loss.mean()

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def predict(self, state, target=False):
        self.model.eval()
        self.model_.eval()
        
        state = torch.from_numpy(state).float().to(DEVICE)
        if target:  # get prediction from target network
            return self.model_(state).to('cpu')
        else:  # get prediction from local network
            return self.model(state).to('cpu')

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,self.state_size), target=target).flatten()

    def update_target_model(self):
        self.model_.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.model.state_dict(), self.weight_backup)
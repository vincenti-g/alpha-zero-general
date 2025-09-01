import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MutsNNet(nn.Module):
    def __init__(self, game, args):
        super(MutsNNet, self).__init__()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv1 = nn.Conv2d(1, args.num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * self.board_x * self.board_y, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.action_size)  
        self.fc4 = nn.Linear(256, 1)

    def forward(self, s):
        # Input shape: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        
        # Convolutional layers
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        
        # Flatten
        s = s.view(-1, self.args.num_channels * self.board_x * self.board_y)
        
        # Fully connected layers
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        # Output heads
        pi = self.fc3(s)  # Policy
        v = self.fc4(s)   # Value
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)
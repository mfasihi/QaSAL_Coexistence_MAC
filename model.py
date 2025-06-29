import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.bn3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        return self.fc4(x)


class MultiHeadQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MultiHeadQNetwork, self).__init__()
        # Shared encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim)
        # Two separate output heads
        self.fc4_delay = nn.Linear(hidden_dim, output_dim)  # Q-values for delay objective
        self.fc4_fair = nn.Linear(hidden_dim, output_dim)  # Q-values for fairness objective

    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        # Two heads
        q_delay = self.fc4_delay(x)  # Q-values targeting delay minimization
        q_fair = self.fc4_fair(x)  # Q-values targeting fairness maximization
        return q_delay, q_fair

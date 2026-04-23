import torch
import torch.nn as nn

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable gate scores (same shape as weights)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Convert gate scores to (0,1)
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gating (pruning effect)
        pruned_weights = self.weight * gates
        
        # Linear transformation
        return torch.matmul(x, pruned_weights.t()) + self.bias


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = PrunableLinear(3072, 512)  # 32*32*3 = 3072
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        # Flatten image
        x = x.view(x.size(0), -1)
        
        # Forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

import torch.nn as nn
import torch.nn.functional as F

class tianqi_2NN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(13, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, inputs):
        tensor = F.sigmoid(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor
    
    
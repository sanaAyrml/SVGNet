import torch.nn as nn



class ResNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU()
        )

    def forward(self, z):
        z = z + self.linear1(z)
        z = z + self.linear2(z)
        z = z + self.linear3(z)
        z = z + self.linear4(z)

        return z
